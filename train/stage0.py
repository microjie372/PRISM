# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Warm-up: latent regression over the full training set, which gives stage 1 a
stable starting point instead of a random one.
"""

import argparse
import logging
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_lib
from skimage.metrics import structural_similarity as skimage_ssim
from omegaconf import OmegaConf, ListConfig
from tqdm.auto import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from prism.encoder import WarpResidualEncoder
from src.models.recon.model_latent_recon import LatentRecon
from src.models.data import get_multi_dataloader
from src.models.utils.model import load_vae, encode_plucker_vae, encode_multi_view_video
from src.models.utils.render import get_plucker_embedding_and_rays
from src.models.utils.misc import dtype_map, load_and_merge_configs
from src.models.utils.train import get_most_recent_checkpoint

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# LatentRecon loader (frozen, used for eval only)
# ---------------------------------------------------------------------------

def load_latent_recon(cfg_lr, device, dtype) -> LatentRecon:
    model = LatentRecon(cfg_lr)
    ckpt_path = cfg_lr.get("resume_pretrained_model_ckpt", None)
    assert ckpt_path, "config must have resume_pretrained_model_ckpt set"
    data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data["module"], strict=False)
    logger.info(f"Loaded LatentRecon from {ckpt_path}")
    model.requires_grad_(False)
    for module in model.modules():
        module.to(device, dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Warp file helpers
# ---------------------------------------------------------------------------

def load_warp_data(file_names, warp_dir, device, dtype):
    warped_list, mask_list, ref_list = [], [], []
    for name in file_names:
        path = os.path.join(warp_dir, f"{name}.pt")
        if not os.path.isfile(path):
            return None, None, None
        data = torch.load(path, map_location="cpu")
        warped_list.append(data["warp_latents"].to(device, dtype=dtype))
        mask_list.append(data["warp_mask"].to(device, dtype=dtype))
        if "ref_latent" in data:
            ref_list.append(data["ref_latent"].to(device, dtype=dtype))
    ref = torch.stack(ref_list, dim=0) if ref_list else None
    return torch.stack(warped_list, dim=0), torch.stack(mask_list, dim=0), ref


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_encoder(encoder, accelerator, output_dir, global_step,
                 checkpoints_total_limit, permanent_checkpointing_steps):
    if not accelerator.is_main_process:
        return
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(encoder)
    torch.save(unwrapped.state_dict(), os.path.join(save_path, "encoder.pt"))
    logger.info(f"Saved encoder checkpoint to {save_path}")

    if checkpoints_total_limit is not None:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        non_perm = [c for c in checkpoints
                    if int(c.split("-")[1]) % permanent_checkpointing_steps != 0]
        while len(non_perm) >= checkpoints_total_limit:
            victim = non_perm.pop(0)
            shutil.rmtree(os.path.join(output_dir, victim))
            logger.info(f"Removed old checkpoint {victim}")


def load_checkpoint(encoder, cfg):
    is_cross_run_init = bool(cfg.get("init_ckpt", None))
    init_ckpt = cfg.get("init_ckpt", None)
    if not init_ckpt:
        ckpt_dir = cfg.get("resume_from_checkpoint", "latest")
        if ckpt_dir == "latest":
            ckpt_dir = get_most_recent_checkpoint(cfg.output_dir)
        if ckpt_dir:
            init_ckpt = os.path.join(ckpt_dir, "encoder.pt")

    if not init_ckpt or not os.path.isfile(init_ckpt):
        logger.warning("No encoder checkpoint found - training from random init")
        return 0

    data = torch.load(init_ckpt, map_location="cpu")
    state_dict = data["module"] if isinstance(data, dict) and "module" in data else data
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading the encoder: {missing[:5]}")
    logger.info(f"Loaded encoder weights from {init_ckpt}")

    if is_cross_run_init:
        logger.info("Cross-run init (init_ckpt set) - resetting step to 0")
        return 0

    parent = os.path.basename(os.path.dirname(init_ckpt))
    step = 0
    if "-" in parent:
        try:
            step = int(parent.split("-")[1])
        except ValueError:
            pass
    return step


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def compute_eval_psnr(images_pred, images_gt, num_input_multi_views):
    _, VT, C, H, W = images_pred.shape
    T    = VT // num_input_multi_views
    pred = images_pred[0].float().reshape(num_input_multi_views, T, C, H, W)
    gt   = images_gt[0].float().reshape(num_input_multi_views, T, C, H, W)
    psnr_list = [-10 * torch.log10(((pred[v] - gt[v]) ** 2).mean() + 1e-8).item()
                 for v in range(num_input_multi_views)]
    return sum(psnr_list) / len(psnr_list)


def compute_eval_ssim_lpips(images_pred, images_gt, eval_lpips_fn, eval_size=256):
    N = images_pred.shape[0] * images_pred.shape[1]
    pred_flat = images_pred.reshape(N, *images_pred.shape[2:]).float()
    gt_flat   = images_gt.reshape(N, *images_gt.shape[2:]).float()
    pred_r = F.interpolate(pred_flat, size=(eval_size, eval_size), mode='bilinear', align_corners=False)
    gt_r   = F.interpolate(gt_flat,   size=(eval_size, eval_size), mode='bilinear', align_corners=False)
    with torch.no_grad():
        lpips_val = eval_lpips_fn(pred_r * 2 - 1, gt_r * 2 - 1).mean().item()
    pred_np = (pred_r.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    gt_np   = (gt_r.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    ssim_val = float(np.mean([
        skimage_ssim(p, g, channel_axis=2, data_range=255)
        for p, g in zip(pred_np, gt_np)
    ]))
    return ssim_val, lpips_val


def run_latent_recon_eval(latent_embed, batch, latent_recon, num_input_multi_views, eval_lpips_fn):
    batch_eval = dict(batch)
    batch_eval["images_input_embed"] = latent_embed
    with torch.no_grad():
        model_out = latent_recon(batch_eval)
    psnr = compute_eval_psnr(
        model_out["images_pred"], batch["images_output"], num_input_multi_views)
    ssim, lpips_val = compute_eval_ssim_lpips(
        model_out["images_pred"], batch["images_output"].float(), eval_lpips_fn)
    return psnr, ssim, lpips_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg, cfg_lr):
    output_dir = cfg.get("output_dir", "outputs/training/prism_init")
    os.makedirs(output_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.get("find_unused_parameters", False)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        mixed_precision="no",
        log_with=cfg.get("log_with", None),
        project_config=ProjectConfiguration(project_dir=output_dir,
                                            logging_dir=cfg.get("logging_dir", "logs")),
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if cfg.get("seed") is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        OmegaConf.save(cfg,    os.path.join(output_dir, "config_prism_init.yaml"))
        OmegaConf.save(cfg_lr, os.path.join(output_dir, "config_lr.yaml"))

    weight_dtype = torch.bfloat16

    # ---- VAE (frozen) ----
    vae = load_vae(cfg_lr.vae_backbone, cfg_lr.vae_path)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    # ---- LatentRecon (frozen, eval only) ----
    cfg_lr_mut = OmegaConf.to_container(cfg_lr, resolve=True)
    cfg_lr_mut["gradient_checkpoint_transformer"] = True
    cfg_lr_mut["gradient_checkpoint_conv"]        = True
    cfg_lr_obj = OmegaConf.create(cfg_lr_mut)
    for k, v in cfg_lr_obj.items():
        if isinstance(v, (list, ListConfig)):
            cfg_lr_obj[k] = tuple(v)
    cfg_lr_obj = OmegaConf.structured(cfg_lr_obj)
    latent_recon = load_latent_recon(cfg_lr_obj, accelerator.device, weight_dtype)

    # ---- Eval LPIPS ----
    eval_lpips_fn = lpips_lib.LPIPS(net="vgg").to(accelerator.device)
    eval_lpips_fn.requires_grad_(False)

    # ---- WarpResidualEncoder (trainable) ----
    encoder = WarpResidualEncoder(
        latent_channels     = cfg.get("prism_latent_channels",    16),
        plucker_channels    = cfg.get("prism_plucker_channels",   32),
        embed_dim           = cfg.get("prism_embed_dim",          512),
        depth               = cfg.get("prism_depth",              8),
        num_heads           = cfg.get("prism_num_heads",          8),
        mlp_ratio           = cfg.get("prism_mlp_ratio",          4.0),
        patch_size          = cfg.get("prism_patch_size",         4),
        use_flow_correction = cfg.get("use_flow_correction",    True),
        flow_hidden         = cfg.get("flow_hidden",            64),
        flow_max_flow       = cfg.get("flow_max_flow",          4.0),
        drop_rate           = cfg.get("prism_drop_rate",          0.0),
    )
    encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"WarpResidualEncoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    initial_step = load_checkpoint(encoder, cfg)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()),
        lr=cfg.get("learning_rate", 1e-4),
        betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
        weight_decay=cfg.get("adam_weight_decay", 1e-2),
        eps=cfg.get("adam_epsilon", 1e-8),
    )

    max_steps    = cfg.get("max_train_steps", 6000)
    sched_name   = cfg.get("lr_scheduler", "cosine")
    warmup_steps = cfg.get("lr_warmup_steps", 100)
    num_cycles   = cfg.get("lr_num_cycles", 1)

    if sched_name == "cyclic_constant_with_warmup":
        steps_per_cycle = max(1, max_steps // num_cycles)
        def cyclic_constant_lr_lambda(current_step: int) -> float:
            cycle_step = current_step % steps_per_cycle
            if cycle_step < warmup_steps:
                return float(cycle_step) / float(max(1, warmup_steps))
            return 1.0
        from torch.optim.lr_scheduler import LambdaLR
        lr_scheduler = LambdaLR(optimizer, cyclic_constant_lr_lambda)
    else:
        lr_scheduler = get_scheduler(
            sched_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps * accelerator.num_processes,
            num_training_steps=max_steps * accelerator.num_processes,
            num_cycles=num_cycles,
        )

    # ---- Data ----
    cfg_data = OmegaConf.merge(cfg_lr, OmegaConf.create({
        "batch_size":  cfg.get("batch_size", cfg_lr.get("batch_size", 1)),
        "num_workers": cfg.get("num_workers", cfg_lr.get("num_workers", 2)),
        "load_latents": True,
        "num_input_multi_views":        cfg.get("num_input_multi_views", 6),
        "sample_num_input_multi_views": False,
    }))

    # Honour dataset_root so the dataset can live outside the repo.
    if cfg.get("dataset_root", None):
        from src.models.data.registry import dataset_registry
        _root = os.path.abspath(os.path.expanduser(cfg.dataset_root))
        dataset_registry[cfg.data_mode[0][0]]['kwargs']['root_path'] = _root

    train_loader, _ = get_multi_dataloader(cfg_data, accelerator)

    encoder, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        encoder, optimizer, train_loader, lr_scheduler
    )

    # ---- Hyperparameters ----
    num_input_multi_views = cfg_lr.get("num_input_multi_views", 6)
    C_lat      = cfg.get("prism_latent_channels", 16)
    T_lat      = cfg.get("prism_latent_T", 16)
    V          = num_input_multi_views
    warp_dir   = cfg.get("warp_precompute_dir", "outputs/warp_precompute")

    lambda_lat_mse = cfg.get("lambda_lat_mse", 1.0)

    logger.info(f"Warm-up: lat_mse={lambda_lat_mse}  max_steps={max_steps}")

    checkpointing_steps      = cfg.get("checkpointing_steps", 500)
    permanent_ckpt_steps     = cfg.get("permanent_checkpointing_steps", 1000)
    checkpoints_total_limit  = cfg.get("checkpoints_total_limit", 5)
    warp_missing = 0

    global_step = initial_step
    progress_bar = tqdm(
        range(global_step, max_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    break_loop = False
    while True:
        for batch in train_loader:
            if global_step >= max_steps:
                break_loop = True
                break

            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    if k not in ["intrinsics_input", "c2ws_input", "cam_view", "intrinsics"]:
                        batch[k] = batch[k].to(accelerator.device, dtype=weight_dtype)
                    else:
                        batch[k] = batch[k].to(accelerator.device)

            if "num_input_multi_views" in batch:
                v = batch["num_input_multi_views"]
                num_input_multi_views = int(v) if isinstance(v, int) else int(v[0].item())
            batch["num_input_multi_views"] = num_input_multi_views
            V = num_input_multi_views

            # ---- Plucker embeddings ----
            batch["plucker_embedding"], batch["rays_os"], batch["rays_ds"] = \
                get_plucker_embedding_and_rays(
                    batch["intrinsics_input"],
                    batch["c2ws_input"],
                    cfg_lr.img_size,
                    cfg_lr.patch_size_out_factor,
                    batch["flip_flag"],
                    get_batch_index=False,
                    dtype=dtype_map[cfg_lr.get("compute_plucker_dtype", "bfloat16")],
                    out_dtype=weight_dtype,
                )
            if cfg_lr.get("plucker_embedding_vae", False):
                batch = encode_plucker_vae(
                    batch,
                    lambda x: encode_multi_view_video(
                        vae, x, num_input_multi_views, cfg_lr.vae_backbone
                    ),
                )

            # ---- Load warp file ----
            file_names = batch["file_name"] if isinstance(batch["file_name"], (list, tuple)) \
                         else [batch["file_name"]]
            warp_latents_raw, warp_mask_raw, ref_latent_saved = load_warp_data(
                file_names, warp_dir, accelerator.device, weight_dtype
            )

            if warp_latents_raw is None:
                warp_missing += 1
                if warp_missing <= 3 or warp_missing % 50 == 0:
                    logger.warning(f"Warp file missing for '{file_names[0]}' - skipping step")
                continue

            B = warp_latents_raw.shape[0]
            H_lat, W_lat = warp_latents_raw.shape[-2], warp_latents_raw.shape[-1]
            T_raw = warp_latents_raw.shape[2]
            warp_vt = warp_latents_raw.reshape(B, V, C_lat, T_raw, H_lat, W_lat) \
                                       .permute(0, 1, 3, 2, 4, 5)   # (B, V, T, C, H, W)
            mask_vt = warp_mask_raw.unsqueeze(3)                     # (B, V, T, 1, H, W)

            if ref_latent_saved is not None:
                ref_latent = ref_latent_saved
            else:
                ref_latent = warp_vt[:, 0, 0]

            plucker_raw  = batch["plucker_embedding"]   # (B, V*T, C_p, H, W)
            C_p          = plucker_raw.shape[2]
            pluckers_all = plucker_raw.reshape(B, V, T_lat, C_p, H_lat, W_lat)

            # ---- GT latents ----
            assert "rgb_latents" in batch, "load_latents must be true - rgb_latents required"
            rgb_lat       = batch["rgb_latents"]
            gt_latents_vt = rgb_lat.reshape(B, V, T_lat, C_lat, H_lat, W_lat).float()

            # ---- Encoder forward, latent regression loss ----
            with accelerator.accumulate(encoder):
                delta = encoder(warp_vt, mask_vt, ref_latent, pluckers_all)

                ref_exp      = ref_latent[:, None, None].expand_as(warp_vt)
                blended_base = warp_vt * mask_vt + ref_exp * (1 - mask_vt)
                pred_latents = blended_base + delta      # (B, V, T, C, H, W)

                loss_latmse = F.mse_loss(pred_latents.float(), gt_latents_vt)
                loss        = lambda_lat_mse * loss_latmse

                accelerator.backward(loss)

                if global_step == initial_step and accelerator.is_main_process:
                    grad_ok = False
                    for name, p in encoder.named_parameters():
                        if p.grad is not None and p.grad.abs().max() > 0:
                            logger.info(f"Gradient check OK: {name} grad max={p.grad.abs().max():.6f}")
                            grad_ok = True
                            break
                    if not grad_ok:
                        logger.error("GRADIENT CHECK FAILED!")
                    logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(encoder.parameters()),
                                                cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    scene=file_names[0][-6:],
                    latmse=f"{loss_latmse.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                if global_step % checkpointing_steps == 0:
                    save_encoder(encoder, accelerator, output_dir, global_step,
                                 checkpoints_total_limit, permanent_ckpt_steps)

                do_eval = (global_step == 1) or \
                          (global_step > 1 and global_step % checkpointing_steps == 0)
                if do_eval and accelerator.is_main_process:
                    with torch.no_grad():
                        images_embed = pred_latents.detach().reshape(B, V * T_lat, C_lat, H_lat, W_lat)
                        gt_embed_ev  = gt_latents_vt.to(weight_dtype).reshape(B, V * T_lat, C_lat, H_lat, W_lat)

                        psnr_pred, ssim_pred, lpips_pred = run_latent_recon_eval(
                            images_embed, batch, latent_recon, num_input_multi_views, eval_lpips_fn)
                        psnr_gt, _, _ = run_latent_recon_eval(
                            gt_embed_ev, batch, latent_recon, num_input_multi_views, eval_lpips_fn)
                        del images_embed, gt_embed_ev

                    tqdm.write(
                        f"[step {global_step:5d}] "
                        f"psnr_pred={psnr_pred:.2f}  ssim={ssim_pred:.4f}  lpips={lpips_pred:.4f}  "
                        f"psnr_gt={psnr_gt:.2f}  "
                        f"latmse={loss_latmse.item():.4f}  "
                        f"scene={file_names[0][-6:]}"
                    )

            del warp_latents_raw, warp_mask_raw, ref_latent_saved, warp_vt, mask_vt, delta
            del ref_exp, blended_base, pred_latents, gt_latents_vt, loss_latmse, loss
            torch.cuda.empty_cache()

        if break_loop:
            break

    save_encoder(encoder, accelerator, output_dir, global_step,
                 checkpoints_total_limit, permanent_ckpt_steps)
    accelerator.end_training()
    logger.info("Warm-up initialization training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if unknown:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(unknown))

    assert "config_path" in cfg, "config must contain config_path list for LatentRecon"
    cfg_lr = load_and_merge_configs(list(cfg.config_path))

    try:
        main(cfg, cfg_lr)
    finally:
        if is_wandb_available():
            import wandb
            wandb.finish()
