# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2: perceptual fine-tuning.

LPIPS on the rendered image, plus an MSE term whose weight is rescaled each
step to hold a fixed share of the total loss, which keeps PSNR from drifting
down as perceptual quality improves.
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
from kiui.lpips import LPIPS

from prism.encoder import WarpResidualEncoder
from src.models.utils.loss import resize_images_lpips
from src.models.recon.model_latent_recon import LatentRecon
from src.models.data import get_multi_dataloader
from src.models.utils.model import load_vae, encode_plucker_vae, encode_multi_view_video
from src.models.utils.render import get_plucker_embedding_and_rays
from src.models.utils.misc import dtype_map, load_and_merge_configs
from src.models.utils.train import get_most_recent_checkpoint

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


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


def fill_warp_holes(warp, mask, kernel_size=15):
    """Fill mask=0 holes with mask-weighted average of neighboring valid pixels.
    warp: (B, V, T, C, H, W),  mask: (B, V, T, 1, H, W)
    Returns:
        filled:     (B, V, T, C, H, W)
        fill_valid: (B, V, T, 1, H, W)  smooth weight [0,1]; 0=no neighbors, 1=fully covered
    """
    B, V, T, C, H, W = warp.shape
    w_flat = warp.reshape(B * V * T, C, H, W)
    m_flat = mask.reshape(B * V * T, 1, H, W)
    pad    = kernel_size // 2
    w_sum  = F.avg_pool2d(w_flat * m_flat, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)
    m_sum  = F.avg_pool2d(m_flat,          kernel_size, stride=1, padding=pad) * (kernel_size ** 2)
    filled     = (w_sum / (m_sum + 1e-6)).reshape(B, V, T, C, H, W)
    fill_valid = (m_sum / (kernel_size ** 2)).clamp(0, 1).reshape(B, V, T, 1, H, W)
    return filled, fill_valid


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
    init_ckpt = cfg.get("init_ckpt", None)
    if not init_ckpt:
        ckpt_dir = cfg.get("resume_from_checkpoint", "latest")
        if ckpt_dir == "latest":
            ckpt_dir = get_most_recent_checkpoint(cfg.output_dir)
        if ckpt_dir:
            init_ckpt = os.path.join(ckpt_dir, "encoder.pt")

    if cfg.get("init_ckpt", None) and not os.path.isfile(cfg["init_ckpt"]):
        raise FileNotFoundError(
            f"init_ckpt is set to {cfg['init_ckpt']} but that file does not "
            "exist. Point it at a checkpoint the previous stage actually wrote."
        )

    if not init_ckpt or not os.path.isfile(init_ckpt):
        logger.warning("No encoder checkpoint found - training from random init")
        return

    data = torch.load(init_ckpt, map_location="cpu")
    state_dict = data["module"] if isinstance(data, dict) and "module" in data else data
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading the encoder: {missing[:5]}")
    logger.info(f"Loaded encoder weights from {init_ckpt}")


def subset_batch_for_rendering(batch, view_indices):
    batch = dict(batch)
    batch["cam_view"]      = batch["cam_view"][:, view_indices]
    batch["intrinsics"]    = batch["intrinsics"][:, view_indices]
    batch["images_output"] = batch["images_output"][:, view_indices]
    if "depths_output" in batch:
        batch["depths_output"] = batch["depths_output"][:, view_indices]
    return batch


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


def main(cfg, cfg_lr):
    output_dir = cfg.get("output_dir", "outputs/training/prism_stage2")
    os.makedirs(output_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
        OmegaConf.save(cfg,    os.path.join(output_dir, "config_prism_stage2.yaml"))
        OmegaConf.save(cfg_lr, os.path.join(output_dir, "config_lr.yaml"))

    weight_dtype = torch.bfloat16

    vae = load_vae(cfg_lr.vae_backbone, cfg_lr.vae_path)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    cfg_lr_mut = OmegaConf.to_container(cfg_lr, resolve=True)
    cfg_lr_mut["gradient_checkpoint_transformer"] = True
    cfg_lr_mut["gradient_checkpoint_conv"]        = True
    cfg_lr_obj = OmegaConf.create(cfg_lr_mut)
    for k, v in cfg_lr_obj.items():
        if isinstance(v, (list, ListConfig)):
            cfg_lr_obj[k] = tuple(v)
    cfg_lr_obj = OmegaConf.structured(cfg_lr_obj)
    latent_recon = load_latent_recon(cfg_lr_obj, accelerator.device, weight_dtype)

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


    )
    encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"WarpResidualEncoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    lambda_lpips = cfg.get("lambda_lpips", 0.5)
    mse_ratio    = cfg.get("lambda_mse_ratio", 0.5)
    lpips_fn = LPIPS(net="vgg").to(accelerator.device)
    lpips_fn.requires_grad_(False)
    logger.info(f"Stage 2: fixed LPIPS={lambda_lpips} + dynamic MSE (MSE share={mse_ratio*100:.0f}%)")

    eval_lpips_fn = lpips_lib.LPIPS(net="vgg").to(accelerator.device)
    eval_lpips_fn.requires_grad_(False)

    load_checkpoint(encoder, cfg)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()),
        lr=cfg.get("learning_rate", 1e-4),
        betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
        weight_decay=cfg.get("adam_weight_decay", 1e-2),
        eps=cfg.get("adam_epsilon", 1e-8),
    )

    max_steps = cfg.get("max_train_steps", 6000)
    lr_scheduler = get_scheduler(
        cfg.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=cfg.get("lr_warmup_steps", 100) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )

    cfg_data = OmegaConf.merge(cfg_lr, OmegaConf.create({
        "batch_size":  cfg.get("batch_size", 1),
        "num_workers": cfg.get("num_workers", 2),
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

    num_input_multi_views = cfg_lr.get("num_input_multi_views", 6)
    C_lat      = cfg.get("prism_latent_channels", 16)
    T_lat      = cfg.get("prism_latent_T", 16)
    V          = num_input_multi_views
    warp_dir   = cfg.get("warp_precompute_dir", "outputs/warp_precompute")
    num_render_views    = cfg.get("e2e_num_render_views", 6)
    hard_view_sampling  = cfg.get("hard_view_sampling", False)
    checkpointing_steps = cfg.get("checkpointing_steps", 100)
    permanent_ckpt_steps    = cfg.get("permanent_checkpointing_steps", 1000)
    checkpoints_total_limit = cfg.get("checkpoints_total_limit", 5)
    warp_missing = 0
    global_step  = 0

    progress_bar = tqdm(range(0, max_steps), desc="Steps",
                        disable=not accelerator.is_local_main_process)

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

            batch["plucker_embedding"], batch["rays_os"], batch["rays_ds"] = \
                get_plucker_embedding_and_rays(
                    batch["intrinsics_input"], batch["c2ws_input"],
                    cfg_lr.img_size, cfg_lr.patch_size_out_factor,
                    batch["flip_flag"], get_batch_index=False,
                    dtype=dtype_map[cfg_lr.get("compute_plucker_dtype", "bfloat16")],
                    out_dtype=weight_dtype,
                )
            if cfg_lr.get("plucker_embedding_vae", False):
                batch = encode_plucker_vae(
                    batch,
                    lambda x: encode_multi_view_video(vae, x, num_input_multi_views, cfg_lr.vae_backbone),
                )

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
            warp_vt = warp_latents_raw.reshape(B, V, C_lat, T_raw, H_lat, W_lat).permute(0, 1, 3, 2, 4, 5)
            mask_vt = warp_mask_raw.unsqueeze(3)
            ref_latent = ref_latent_saved if ref_latent_saved is not None else warp_vt[:, 0, 0]

            plucker_raw  = batch["plucker_embedding"]
            C_p          = plucker_raw.shape[2]
            pluckers_all = plucker_raw.reshape(B, V, T_lat, C_p, H_lat, W_lat)

            with accelerator.accumulate(encoder):
                delta = encoder(warp_vt, mask_vt, ref_latent, pluckers_all)

                ref_exp      = ref_latent[:, None, None].expand_as(warp_vt)
                blended_base = warp_vt * mask_vt + ref_exp * (1 - mask_vt)
                pred_latents = blended_base + delta

                images_input_embed = pred_latents.reshape(B, V * T_lat, C_lat, H_lat, W_lat)

                total_views = batch["cam_view"].shape[1]
                if hard_view_sampling:
                    n_per_view = total_views // V
                    coverage = mask_vt[0].float().mean(dim=(1, 2, 3, 4))  # (V,) on device
                    view_assignments = torch.arange(total_views, device=accelerator.device) // n_per_view
                    target_weights = 1.0 / (coverage[view_assignments] + 1e-2)
                    target_weights = target_weights / target_weights.sum()
                    view_idx = torch.multinomial(target_weights, num_render_views, replacement=False)
                else:
                    view_idx = torch.randperm(total_views, device=accelerator.device)[:num_render_views]
                batch_render = subset_batch_for_rendering(batch, view_idx)
                batch_render["num_input_multi_views"] = num_input_multi_views
                batch_render["images_input_embed"]    = images_input_embed

                model_output = latent_recon(batch_render)
                pred_images  = model_output["images_pred"].float()
                gt_images    = batch_render["images_output"].float()

                # Stage 2 loss: fixed LPIPS + dynamic MSE
                loss_mse  = F.mse_loss(pred_images, gt_images)
                loss_lpips = lpips_fn(
                    resize_images_lpips(pred_images, tuple(cfg_lr.img_size), cfg_lr.lpips_img_size_min),
                    resize_images_lpips(gt_images,   tuple(cfg_lr.img_size), cfg_lr.lpips_img_size_min),
                ).mean()
                lambda_mse_dyn = lambda_lpips * (mse_ratio / (1.0 - mse_ratio)) * \
                                 loss_lpips.detach() / (loss_mse.detach() + 1e-8)
                loss_render    = lambda_lpips * loss_lpips + lambda_mse_dyn * loss_mse

                accelerator.backward(loss_render)

                # Gradient check at first step
                if global_step == 0 and accelerator.is_main_process:
                    grad_ok = False
                    for name, p in encoder.named_parameters():
                        if p.grad is not None and p.grad.abs().max() > 0:
                            logger.info(f"Gradient check OK: {name} grad max={p.grad.abs().max():.6e}")
                            grad_ok = True
                            break
                    if not grad_ok:
                        logger.error("GRADIENT CHECK FAILED!")
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9
                    logger.info(f"Stage2 peak GPU memory: {mem_gb:.2f} GB")

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(encoder.parameters()), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                with torch.no_grad():
                    psnr_train = -10 * torch.log10(loss_mse.detach() + 1e-8).item()

                progress_bar.set_postfix(
                    scene=file_names[0][-6:],
                    psnr=f"{psnr_train:.1f}",
                    lpips=f"{loss_lpips.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                if global_step % checkpointing_steps == 0:
                    save_encoder(encoder, accelerator, output_dir, global_step,
                                 checkpoints_total_limit, permanent_ckpt_steps)

                do_eval = (global_step == 1) or (global_step % checkpointing_steps == 0)
                if do_eval:
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        batch_eval = dict(batch)
                        batch_eval["images_input_embed"] = images_input_embed.detach()
                        model_out_eval = latent_recon(batch_eval)
                        psnr_eval = compute_eval_psnr(
                            model_out_eval["images_pred"], batch["images_output"], num_input_multi_views,
                        )
                        ssim_eval, lpips_eval = compute_eval_ssim_lpips(
                            model_out_eval["images_pred"], batch["images_output"].float(), eval_lpips_fn,
                        )
                        # GT latent baseline
                        psnr_gt = None
                        gt_latents_vt = None
                        if "rgb_latents" in batch:
                            rgb_lat = batch["rgb_latents"]
                            gt_latents_vt = rgb_lat.reshape(B, V, T_lat, C_lat, H_lat, W_lat).float()
                        if gt_latents_vt is not None:
                            gt_embed = gt_latents_vt.to(weight_dtype).reshape(
                                B, num_input_multi_views * T_lat, C_lat, H_lat, W_lat)
                            batch_gt = dict(batch)
                            batch_gt["images_input_embed"] = gt_embed
                            model_out_gt = latent_recon(batch_gt)
                            psnr_gt = compute_eval_psnr(
                                model_out_gt["images_pred"], batch["images_output"], num_input_multi_views,
                            )
                            del model_out_gt, batch_gt
                    if accelerator.is_main_process:
                        gt_str = f"  psnr_gt={psnr_gt:.2f}" if psnr_gt is not None else ""
                        tqdm.write(
                            f"[step {global_step:5d}] "
                            f"psnr={psnr_eval:.2f}  ssim={ssim_eval:.4f}  lpips={lpips_eval:.4f}  "
                            f"mse_w={lambda_mse_dyn.item():.2f}{gt_str}  scene={file_names[0][-6:]}"
                        )
                    del model_out_eval, batch_eval
                    if gt_latents_vt is not None:
                        del gt_latents_vt, gt_embed

            del warp_latents_raw, warp_mask_raw, ref_latent_saved, warp_vt, mask_vt, delta
            del ref_exp, blended_base, pred_latents, images_input_embed
            del model_output, batch_render, pred_images, gt_images
            del loss_mse, loss_lpips, lambda_mse_dyn, loss_render
            torch.cuda.empty_cache()

        if break_loop:
            break

    save_encoder(encoder, accelerator, output_dir, global_step,
                 checkpoints_total_limit, permanent_ckpt_steps)
    accelerator.end_training()
    logger.info("Stage 2 complete.")


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
