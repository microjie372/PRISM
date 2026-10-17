# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from NVIDIA Lyra
# (https://github.com/nv-tlabs/lyra), Copyright (c) 2025 NVIDIA CORPORATION,
# licensed under the Apache License, Version 2.0.

"""
Run PRISM on a single image, with no dataset.

Monocular depth and a generated camera trajectory give the warp prior, the
encoder predicts the residual, and the frozen decoder renders the result as a
video.
"""

import argparse
import json
import os
import re

import torch
from PIL import Image
from torchvision import transforms
from accelerate import PartialState
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from tqdm import tqdm

from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from src.models.utils.misc import load_and_merge_configs, seed_everything
from src.models.utils.model import encode_video, load_vae
from src.models.utils.render import save_ply
from src.utils.visu import generate_wave_video, save_video

logger = get_logger(__name__, log_level="INFO")

from prism.pipeline import (
    MULTI_TRAJECTORIES,
    load_encoder,
    load_latent_recon,
    run_encoder_all_views,
    run_moge,
    weight_dtype,
)


def main(args):
    seed_everything(args.seed)
    distributed_state = PartialState()
    device = distributed_state.device

    inf_config  = load_and_merge_configs([args.config_default, args.config])
    cli         = OmegaConf.from_dotlist(args.overrides)
    inf_config  = OmegaConf.merge(inf_config, cli)

    main_config = load_and_merge_configs(list(inf_config.config_path))
    main_config.load_latents = False

    img_size              = tuple(main_config.get("img_size", [704, 1280]))
    patch_size_out_factor = tuple(main_config.get("patch_size_out_factor", [1, 8, 8]))

    encoder_ckpt_path = args.encoder_ckpt or inf_config.get("encoder_ckpt_path")
    recon_ckpt_path   = inf_config.get("recon_ckpt_path")
    assert encoder_ckpt_path and os.path.isfile(encoder_ckpt_path), \
        f"encoder_ckpt_path not found: {encoder_ckpt_path}"
    assert recon_ckpt_path and os.path.isfile(recon_ckpt_path), \
        f"recon_ckpt_path not found: {recon_ckpt_path}"

    scene_name = os.path.splitext(os.path.basename(args.image))[0]
    enc_tag    = re.search(r"(checkpoint-\d+)", encoder_ckpt_path)
    enc_tag    = enc_tag.group(1) if enc_tag else "encoder"
    outdir     = os.path.join(
        inf_config.get("out_dir_inference_single", "outputs/inference_single"),
        enc_tag, scene_name,
    )

    vae         = load_vae(main_config.vae_backbone, main_config.vae_path)
    vae.to(device=device, dtype=weight_dtype).eval()
    transformer = load_latent_recon(recon_ckpt_path, main_config, device)
    encoder     = load_encoder(encoder_ckpt_path, inf_config, device)

    print("Running MoGe depth estimation...")
    moge_image, moge_depth, moge_mask, moge_w2c, moge_intrinsics = \
        run_moge(args.image, img_size[0], img_size[1], device)
    initial_w2c        = moge_w2c[0, 0]
    initial_intrinsics = moge_intrinsics[0, 0]

    ref_depth_hw = (moge_depth if isinstance(moge_depth, torch.Tensor)
                    else torch.tensor(moge_depth, dtype=torch.float32)).float().to(device)
    if ref_depth_hw.dim() == 2:   ref_depth_hw = ref_depth_hw.unsqueeze(0).unsqueeze(0)
    elif ref_depth_hw.dim() == 3: ref_depth_hw = ref_depth_hw.unsqueeze(0)

    print(f"Encoding reference image: {args.image}")
    preprocess = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = preprocess(Image.open(args.image).convert("RGB")) \
        .unsqueeze(0).unsqueeze(0).to(device, dtype=weight_dtype)
    with torch.no_grad():
        ref_latent = encode_video(vae, img_tensor, main_config.vae_backbone)[:, 0]

    outdir_raw  = os.path.join(outdir, "raw")
    outdir_meta = os.path.join(outdir, "meta")
    outdir_grid = os.path.join(outdir, "grid")
    outdir_full = os.path.join(outdir, "full_output")
    outdir_3dgs = os.path.join(outdir, "main_gaussians_renderings")
    for d in [outdir_raw, outdir_meta, outdir_grid, outdir_full, outdir_3dgs]:
        os.makedirs(d, exist_ok=True)

    wave_color_dict = {
        'wave_color_front':   [255, 230, 200],
        'wave_color_back':    [200, 220, 255],
        'use_gradient_color': True,
    }
    num_grid_samples = inf_config.get("num_grid_samples", 4)

    # Always build all 6 trajectory cameras for V=6 encoder input (matches training)
    traj_names = list(MULTI_TRAJECTORIES.keys())
    print("Building camera trajectories...")
    cameras_list = []
    for traj_name, traj_cfg in MULTI_TRAJECTORIES.items():
        cam_gen_kwargs = traj_cfg.get("camera_gen_kwargs", {})
        w2cs, intrinsics_3x3 = generate_camera_trajectory(
            trajectory_type    = traj_name,
            initial_w2c        = initial_w2c,
            initial_intrinsics = initial_intrinsics,
            num_frames         = 121,
            movement_distance  = traj_cfg["movement_distance"] * args.movement_distance_factor,
            camera_rotation    = "center_facing",
            device             = device,
            **cam_gen_kwargs,
        )
        cameras_list.append((w2cs, intrinsics_3x3))

    # Run encoder once with V=6
    print("Running WarpResidualEncoder (V=6)...")
    pred_latents_all, batch_list = run_encoder_all_views(
        cameras_list, ref_latent, ref_depth_hw, vae, encoder,
        main_config, inf_config, img_size, patch_size_out_factor, device,
    )

    # Determine which views to render
    if args.multi_trajectory:
        render_indices = list(range(len(traj_names)))
    else:
        assert args.trajectory in traj_names, \
            f"--trajectory {args.trajectory} not in MULTI_TRAJECTORIES: {traj_names}"
        render_indices = [traj_names.index(args.trajectory)]

    # Build full V=6 multi-view input for LatentRecon (must match training)
    # Training always passed (B, V*T, C, H, W) with num_input_multi_views=6.
    # Passing V=1 disables cross-view attention and produces garbage.
    V_enc = pred_latents_all.shape[1]   # 6
    T_lat = pred_latents_all.shape[2]
    C_lat = pred_latents_all.shape[3]
    H_lat = pred_latents_all.shape[4]
    W_lat = pred_latents_all.shape[5]
    images_input_embed_full = pred_latents_all.reshape(1, V_enc * T_lat, C_lat, H_lat, W_lat)
    plucker_full = torch.cat([b["plucker_embedding"] for b in batch_list], dim=1)
    rays_os_full = torch.cat([b["rays_os"]           for b in batch_list], dim=1)
    rays_ds_full = torch.cat([b["rays_ds"]           for b in batch_list], dim=1)

    step_test_sum         = 0
    step_test_sum_dataset = 0
    test_video_out        = []
    test_video_out_rgb    = []

    for out_idx, v_idx in enumerate(tqdm(render_indices, desc="Rendering")):
        traj_name = traj_names[v_idx]
        print(f"\n--- Rendering: {traj_name} ---")

        b_v = batch_list[v_idx]
        batch = {
            "images_input_embed":    images_input_embed_full,
            "plucker_embedding":     plucker_full,
            "rays_os":               rays_os_full,
            "rays_ds":               rays_ds_full,
            "num_input_multi_views": V_enc,
            "cam_view":              b_v["cam_view"],
            "intrinsics":            b_v["intrinsics"],
            "time_embeddings":       b_v["time_embeddings"],
            "time_embeddings_target": b_v["time_embeddings_target"],
            "flip_flag":             b_v["flip_flag"],
        }

        with torch.no_grad():
            model_output = transformer(batch)

        pred_images = model_output['images_pred'].float().cpu()

        if args.save_gaussians and model_output.get('gaussians') is not None:
            out_dir_gaussians = os.path.join(outdir, 'gaussians')
            os.makedirs(out_dir_gaussians, exist_ok=True)
            save_ply(model_output['gaussians'],
                     os.path.join(out_dir_gaussians, f'gaussians_{out_idx}.ply'))
        model_output.pop('gaussians', None)

        pred_images_wave = generate_wave_video(
            model_output['images_pred'].float(), model_output['depths_pred'].float(),
            wave_length=0.4, wave_color=None,
            use_gradient_color=wave_color_dict['use_gradient_color'],
            wave_color_front=wave_color_dict['wave_color_front'],
            wave_color_back=wave_color_dict['wave_color_back'],
        )
        pred_images_rgb = torch.cat((pred_images_wave, pred_images), 1)
        save_video(pred_images_rgb,  outdir_3dgs, name=f'rgb_{out_idx}',            fps=args.out_fps)
        save_video(pred_images_wave, outdir_raw,  name=f'rgb_wave_{out_idx}',       fps=args.out_fps)
        save_video(pred_images,      outdir_raw,  name=f'rgb_{out_idx}_view_idx_0', fps=args.out_fps)
        save_video(pred_images,      outdir_full, name=f'sample_{out_idx}',         fps=args.out_fps)

        with open(os.path.join(outdir_meta, f'sample_{out_idx}.json'), 'w') as f:
            json.dump({'file_name': traj_name, 'trajectory': traj_name}, f)
        print(f"Saved: full_output/sample_{out_idx}.mp4  ({traj_name})")

        step_test_sum += 1
        test_video_out.append(pred_images)
        test_video_out_rgb.append(pred_images_rgb)

        if step_test_sum >= num_grid_samples:
            save_video(torch.cat(test_video_out, 0),     outdir_grid,
                       name=f'sample_grid_{step_test_sum_dataset}', fps=args.out_fps)
            save_video(torch.cat(test_video_out_rgb, 0), outdir_grid,
                       name=f'rgb_grid_{step_test_sum_dataset}',    fps=args.out_fps)
            step_test_sum         = 0
            step_test_sum_dataset += 1
            test_video_out        = []
            test_video_out_rgb    = []

        del model_output
        torch.cuda.empty_cache()

    if test_video_out:
        save_video(torch.cat(test_video_out, 0),     outdir_grid,
                   name=f'sample_grid_{step_test_sum_dataset}', fps=args.out_fps)
        save_video(torch.cat(test_video_out_rgb, 0), outdir_grid,
                   name=f'rgb_grid_{step_test_sum_dataset}',    fps=args.out_fps)

    print(f"\nDone. Results saved to: {outdir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single image -> WarpResidualEncoder -> 3DGS")
    parser.add_argument("--image",           type=str, required=True)
    parser.add_argument("--config",          type=str,
                        default="configs/inference/prism.yaml")
    parser.add_argument("--encoder_ckpt",    type=str, default=None,
                        help="Encoder checkpoint; overrides encoder_ckpt_path "
                             "in the config.")
    parser.add_argument("--config_default",  type=str,
                        default="configs/inference/default.yaml")
    parser.add_argument("--multi_trajectory", action="store_true")
    parser.add_argument("--trajectory", type=str, default="left",
                        choices=sorted(MULTI_TRAJECTORIES))
    parser.add_argument("--movement_distance_factor", type=float, default=1.0)
    parser.add_argument("--out_fps",         type=int,   default=24)
    parser.add_argument("--save_gaussians",  action="store_true")
    parser.add_argument("--seed",            type=int,   default=42)

    args, unknown = parser.parse_known_args()
    args.overrides = unknown
    main(args)
