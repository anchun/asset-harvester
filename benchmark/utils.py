#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Load PLY and camera data from ./example_data, render the PLY assets,
compute metrics with original frames, and visualize results.
"""

import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torchmetrics
from gsplat.rendering import rasterization
from kiui.cam import orbit_camera

from asset_harvester.tokengs.ply_io import read_ply


def apply_y_rotation_gaussians(gaussians, degrees):
    """
    Rotate 3D Gaussians around the Y axis by the given angle in degrees.
    gaussians: [1, N, 14] tensor (xyz, opacity, scale, quat_wxyz, rgb).
    Rotates positions and composes rotation with the per-splat quaternion.
    """
    rad = np.deg2rad(float(degrees))
    c, s = np.cos(rad), np.sin(rad)
    # R_y: x' = x*c - z*s, y' = y, z' = x*s + z*c
    R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)
    device = gaussians.device
    g = gaussians.float().to(device).clone()
    xyz = g[0, :, 0:3].detach().cpu().numpy()
    xyz_rot = xyz @ R.T
    g[0, :, 0:3] = torch.from_numpy(xyz_rot).float().to(device)
    # Quaternion for Y rotation: (w, x, y, z) = (cos(θ/2), 0, sin(θ/2), 0)
    half = rad / 2.0
    qw, qx, qy, qz = np.cos(half), 0.0, np.sin(half), 0.0
    quats = g[0, :, 7:11].cpu().numpy()
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    new_w = qw * w - qx * x - qy * y - qz * z
    new_x = qw * x + qx * w + qy * z - qz * y
    new_y = qw * y - qx * z + qy * w + qz * x
    new_z = qw * z + qx * y - qy * x + qz * w
    g[0, :, 7:11] = torch.from_numpy(np.stack([new_w, new_x, new_y, new_z], axis=-1)).float().to(device)
    return g


def load_ply(path):
    """
    Load PLY file and convert to gaussians tensor [1, N, 14].
    Format: (xyz 3, opacity 1, scale 3, rot 4, rgb 3)
    """
    vertex = read_ply(path)

    # Extract xyz
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    # Extract opacity (stored as inverse sigmoid)
    opacity = vertex["opacity"][:, np.newaxis]
    opacity = 1.0 / (1.0 + np.exp(-opacity))  # sigmoid activation

    # Extract scales (stored as log)
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1)
    scales = np.exp(scales)  # exp activation

    # Extract rotations
    rotations = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1)

    # Extract colors (SH DC component, convert back from SH to RGB)
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1)
    rgb = f_dc * 0.28209479177387814 + 0.5  # Convert SH to RGB
    rgb = np.clip(rgb, 0, 1)

    # Stack all: [N, 14] = xyz(3) + opacity(1) + scale(3) + rot(4) + rgb(3)
    gaussians = np.concatenate([xyz, opacity, scales, rotations, rgb], axis=-1)
    gaussians = torch.from_numpy(gaussians).float().unsqueeze(0)  # [1, N, 14]

    return gaussians


def render(opt, gaussians, cam_view, intrinsics):
    """
    Render gaussians from given camera views.
    Borrowed from batch_inference_benchformat_1frame.py

    Args:
        opt: Options with output_size, znear, zfar
        gaussians: [B, N, 14] tensor
        cam_view: [B, V, 4, 4] camera view matrices (pre-transposed)
        intrinsics: [B, V, 4] intrinsics (fx, fy, cx, cy)

    Returns:
        dict with 'image', 'alpha', 'depth' tensors
    """
    B, V = cam_view.shape[:2]

    images = []
    alphas = []
    depths = []
    bg_rgb = getattr(opt, "bg_color", (1.0, 1.0, 1.0))
    if isinstance(bg_rgb, torch.Tensor):
        bg_color = bg_rgb.float().flatten()[:3].to("cuda")
    else:
        bg_color = torch.tensor(list(bg_rgb)[:3], dtype=torch.float32, device="cuda")

    for b in range(B):
        # pos, opacity, scale, rotation, shs
        means3D = gaussians[b, :, 0:3].contiguous().float()
        opacity = gaussians[b, :, 3:4].contiguous().float()
        scales = gaussians[b, :, 4:7].contiguous().float()
        rotations = gaussians[b, :, 7:11].contiguous().float()
        rgbs = gaussians[b, :, 11:].contiguous().float()  # [N, 3]

        # render novel views
        view_matrix = cam_view[b].float()
        viewmat = view_matrix.transpose(2, 1)  # [V, 4, 4]
        view_intrinsics = intrinsics[b]

        Ks = [
            torch.tensor(
                [
                    [view_intrinsic[0], 0.0, view_intrinsic[2]],
                    [0.0, view_intrinsic[1], view_intrinsic[3]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device="cuda",
            )
            for view_intrinsic in view_intrinsics
        ]

        rendered_image_all, rendered_alpha_all, info = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=rgbs,
            viewmats=viewmat,
            Ks=torch.stack(Ks),
            width=opt.output_size,
            height=opt.output_size,
            near_plane=opt.znear,
            far_plane=opt.zfar,
            packed=False,
            backgrounds=torch.stack([bg_color for _ in range(V)]),
            render_mode="RGB+ED",
        )

        for rendered_image, rendered_alpha in zip(rendered_image_all, rendered_alpha_all):
            depths.append(rendered_image[..., 3:].permute(2, 0, 1))
            rendered_image = rendered_image[..., :3].permute(2, 0, 1)
            images.append(rendered_image)
            alphas.append(rendered_alpha.permute(2, 0, 1))

    images = torch.stack(images, dim=0).view(B, V, 3, opt.output_size, opt.output_size)
    alphas = torch.stack(alphas, dim=0).view(B, V, 1, opt.output_size, opt.output_size)
    depths = torch.stack(depths, dim=0).view(B, V, 1, opt.output_size, opt.output_size)

    return {
        "image": images,  # [B, V, 3, H, W]
        "alpha": alphas,  # [B, V, 1, H, W]
        "depth": depths,
    }


def render_from_cameras(gaussians, cam_poses, dists, fovs, lwh, output_size=512, bg_color=(1.0, 1.0, 1.0)):
    """
    Render gaussians from the loaded camera parameters.

    Args:
        gaussians: [1, N, 14] tensor on CUDA
        cam_poses: [N, 3] camera positions (x, y, z) - normalized direction vectors
        dists: [N] distances
        fovs: [N] field of view in degrees (can be single value or per-frame)
        lwh: [3] object dimensions [length, width, height]
        output_size: Output image resolution
        bg_color: RGB in [0, 1] behind the splats (default white).

    Returns:
        rendered_images: [N, 3, H, W] tensor (0-1 range)
        rendered_alphas: [N, 1, H, W] tensor (0-1 range), for alignment to GT masks.
    """
    opt = SimpleNamespace(output_size=output_size, znear=0.1, zfar=500, bg_color=bg_color)
    num_views = len(cam_poses)

    # Normalize distances by object size (same as reference)
    max_lwh = max(lwh)

    images = []
    alphas = []

    for i in range(num_views):
        # Get camera parameters for this view
        x = cam_poses[i][1] * -1
        y = cam_poses[i][2]
        z = cam_poses[i][0]
        dist_normalized = dists[i] / max_lwh
        fov = fovs[i] if hasattr(fovs, "__len__") and len(fovs) > 1 else float(fovs)

        # Compute azimuth and elevation from camera position
        azimuth = np.arctan2(z, x) * 180 / np.pi + 90
        elevation = np.arcsin(y) * 180 / np.pi * -1

        # Compute intrinsics for this view
        tan_half_fov = np.tan(0.5 * np.deg2rad(fov))
        f = output_size / (2 * tan_half_fov)
        intrinsics = torch.tensor([f, f, output_size / 2.0, output_size / 2.0])[None, ...].cuda()

        # Create camera pose using orbit_camera (no opengl=True, matching reference)
        cam_pose = (
            torch.from_numpy(orbit_camera(elevation, azimuth, radius=dist_normalized)).unsqueeze(0).cuda().float()
        )
        cam_pose[:, :3, 1:3] *= -1  # invert up & forward direction

        # Convert to view matrix for rasterizer
        # cam_view = inverse(c2w), transposed for gsplat
        cam_view = torch.inverse(cam_pose).transpose(1, 2)  # [1, 4, 4]

        # Render
        result = render(opt, gaussians, cam_view[None], intrinsics=intrinsics[None])
        images.append(result["image"].squeeze(0).squeeze(0))  # [3, H, W]
        alphas.append(result["alpha"].squeeze(0).squeeze(0))  # [1, H, W]

    rendered_images = torch.stack(images, dim=0)  # [N, 3, H, W]
    rendered_alphas = torch.stack(alphas, dim=0)  # [N, 1, H, W]
    return rendered_images, rendered_alphas


def _weighted_center_and_area(mask_2d):
    """
    Return (center_x, center_y, area) for foreground from a binary mask.
    Center and area are from the axis-aligned bounding box of non-zero pixels.
    mask_2d: [H, W] (or squeezable to 2D) numpy or tensor; binary (foreground = non-zero).
    Returns (cx, cy, area) or None if no foreground.
    """
    if hasattr(mask_2d, "cpu"):
        mask_np = np.squeeze(mask_2d.detach().cpu().numpy())
    else:
        mask_np = np.squeeze(np.asarray(mask_2d))
    if mask_np.ndim != 2:
        return None
    binary = mask_np != 0
    if not np.any(binary):
        return None
    rows_inds, cols_inds = np.where(binary)
    x0 = int(cols_inds.min())
    x1 = int(cols_inds.max()) + 1
    y0 = int(rows_inds.min())
    y1 = int(rows_inds.max()) + 1
    cx = (x0 + x1 - 1) / 2.0
    cy = (y0 + y1 - 1) / 2.0
    area = float((x1 - x0) * (y1 - y0))
    return (float(cx), float(cy), area)


def align_rendered_to_gt_masks(
    rendered_images,
    rendered_alphas,
    gt_masks,
    output_size,
    quiet=False,
    border_rgb=(255, 255, 255),
):
    """
    Translate and scale each rendered image so that the rendered foreground (alpha)
    matches the GT mask in weighted center and area. Uses OpenCV warpAffine per frame.

    Args:
        rendered_images: [N, 3, H, W] tensor (0-1), H=W=output_size.
        rendered_alphas: [N, 1, H, W] tensor (0-1).
        gt_masks: list of numpy [H_gt, W_gt] (0-255, foreground=255).
        output_size: int, size of rendered and target grid.
        quiet: if True, do not print per-frame alignment stats.
        border_rgb: (R, G, B) 0-255 for warpAffine fill where alpha is undefined.

    Returns:
        aligned_images: [N, 3, H, W] tensor, same shape as rendered_images.
        aligned_alphas: [N, 1, H, W] tensor, same shape as rendered_alphas.
    """
    import cv2

    N, _, H, W = rendered_images.shape
    device = rendered_images.device
    aligned_list = []
    aligned_alphas_list = []
    for i in range(N):
        gt_mask = np.asarray(gt_masks[i])
        assert gt_mask.shape[0] == output_size and gt_mask.shape[1] == output_size, (
            f"GT mask shape {gt_mask.shape} does not match output_size {output_size}"
        )
        gt_info = _weighted_center_and_area(gt_mask > 127)
        alpha_i = rendered_alphas[i, 0]
        rend_info = _weighted_center_and_area(alpha_i > 0.3)
        if gt_info is None or rend_info is None:
            aligned_list.append(rendered_images[i : i + 1])
            a = rendered_alphas[i : i + 1]
            while a.dim() > 4:
                a = a[:, :, 0]
            aligned_alphas_list.append(a)
            continue
        cx_gt, cy_gt, area_gt = gt_info
        cx_r, cy_r, area_rend = rend_info
        area_ratio = area_gt / max(area_rend, 1e-6)
        scale = float(np.sqrt(max(area_ratio, 1e-6)))
        # Forward affine (src -> dst): dst_x = scale*(src_x - cx_r) + cx_gt. warpAffine expects dst->src, so we pass forward to try.
        M = np.array(
            [
                [scale, 0.0, cx_gt - scale * cx_r],
                [0.0, scale, cy_gt - scale * cy_r],
            ],
            dtype=np.float32,
        )
        # Rendered image: [1, 3, H, W] 0-1 -> (H, W, 3) 0-255 for OpenCV
        img_np = rendered_images[i].permute(1, 2, 0).cpu().numpy()
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        warped_img = cv2.warpAffine(
            img_u8,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(int(round(c)) for c in border_rgb),
        )
        aligned_list.append(torch.from_numpy(warped_img).permute(2, 0, 1).float().to(device=device) / 255.0)
        # Alpha: ensure 2D [H, W] then warp
        alpha_np = rendered_alphas[i, 0].cpu().numpy()
        alpha_np = np.squeeze(alpha_np)
        while alpha_np.ndim > 2:
            alpha_np = alpha_np[0]
        alpha_u8 = (np.clip(alpha_np, 0, 1) * 255).astype(np.uint8)
        warped_alpha = cv2.warpAffine(
            alpha_u8,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # warped_alpha is (H, W); output (1, 1, H, W)
        warped_alpha = np.asarray(warped_alpha)
        if warped_alpha.ndim > 2:
            warped_alpha = warped_alpha.squeeze()
            while warped_alpha.ndim > 2:
                warped_alpha = warped_alpha[0]
        aligned_alphas_list.append(
            torch.from_numpy(warped_alpha).float().to(device=device).unsqueeze(0).unsqueeze(0) / 255.0
        )
    aligned = torch.stack(aligned_list, dim=0)
    aligned_alphas = torch.cat(aligned_alphas_list, dim=0)

    # Compare aligned alpha vs GT mask: center and area per frame
    if not quiet:
        for i in range(N):
            alpha_np = aligned_alphas[i, 0].cpu().numpy().squeeze()
            aligned_info = _weighted_center_and_area(alpha_np > 0.3)
            gt_info = _weighted_center_and_area(np.asarray(gt_masks[i]) > 127)
            if aligned_info is not None and gt_info is not None:
                cx_a, cy_a, area_a = aligned_info
                cx_g, cy_g, area_g = gt_info
                center_dist = np.hypot(cx_a - cx_g, cy_a - cy_g)
                area_ratio = area_a / max(area_g, 1e-6)
                print(
                    f"  frame {i}: aligned_alpha center=({cx_a:.1f},{cy_a:.1f}) area={area_a:.0f}  "
                    f"gt center=({cx_g:.1f},{cy_g:.1f}) area={area_g:.0f}  "
                    f"center_dist={center_dist:.2f} area_ratio={area_ratio:.3f}"
                )
            else:
                print(f"  frame {i}: aligned_info={aligned_info} gt_info={gt_info}")

    return aligned, aligned_alphas


def compute_metrics(rendered_images, gt_frames_masked):
    """
    Compute PSNR, LPIPS, and SSIM between rendered images and pre-masked GT frames.
    GT frames are already masked with white background (0-255); no mask argument.

    Args:
        rendered_images: [N, 3, H, W] rendered images (0-1 range) with white background
        gt_frames_masked: list of numpy arrays [H, W, 3] (0-255), already masked (white bg)
    Returns:
        dict with psnr, lpips, ssim, mean_*, rendered_list, gt_masked_list
    """
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).cuda()
    lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    psnr_values = []
    lpips_values = []
    ssim_values = []
    rendered_list = []
    gt_masked_list = []

    for i, gt_frame in enumerate(gt_frames_masked):
        gt_tensor = torch.from_numpy(gt_frame).float().permute(2, 0, 1) / 255.0
        gt_tensor = gt_tensor.unsqueeze(0).cuda()

        rendered = rendered_images[i : i + 1]
        if rendered.shape[-2:] != gt_tensor.shape[-2:]:
            rendered = F.interpolate(rendered, size=gt_tensor.shape[-2:], mode="bilinear", align_corners=False)

        rendered = rendered.clamp(0, 1)
        gt_tensor = gt_tensor.clamp(0, 1)

        psnr = psnr_metric(rendered, gt_tensor)
        psnr_values.append(psnr.item())
        lpips = lpips_metric(rendered, gt_tensor)
        lpips_values.append(lpips.item())
        ssim = ssim_metric(rendered, gt_tensor)
        ssim_values.append(ssim.item())

        rendered_list.append(rendered.detach().clone())
        gt_masked_list.append(gt_tensor.detach().clone())

    return {
        "psnr": psnr_values,
        "lpips": lpips_values,
        "ssim": ssim_values,
        "mean_psnr": np.mean(psnr_values),
        "mean_lpips": np.mean(lpips_values),
        "mean_ssim": np.mean(ssim_values),
        "rendered_list": rendered_list,
        "gt_masked_list": gt_masked_list,
    }


def create_comparison_image_from_metrics(metrics, output_size=512):
    """
    Create side-by-side comparison images from compute_metrics output (GT | Rendered),
    using the same masked images used for PSNR/SSIM.

    Args:
        metrics: dict from compute_metrics with "rendered_list" and "gt_masked_list"
                 (lists of [1, 3, H, W] tensors in 0-1 range)
        output_size: int, size for resizing each half

    Returns:
        list of PIL Images: RGB row (GT masked | Rendered masked).
    """
    rendered_list = metrics["rendered_list"]
    gt_masked_list = metrics["gt_masked_list"]
    comparisons = []
    for i in range(len(rendered_list)):
        gt_np = gt_masked_list[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = (gt_np * 255).clip(0, 255).astype(np.uint8)
        ren_np = rendered_list[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        ren_np = (ren_np * 255).clip(0, 255).astype(np.uint8)
        if gt_np.shape[0] != output_size or gt_np.shape[1] != output_size:
            gt_np = np.array(Image.fromarray(gt_np).resize((output_size, output_size), Image.BILINEAR))
        if ren_np.shape[0] != output_size or ren_np.shape[1] != output_size:
            ren_np = np.array(Image.fromarray(ren_np).resize((output_size, output_size), Image.BILINEAR))
        row = np.concatenate([gt_np, ren_np], axis=1)
        comparisons.append(Image.fromarray(row))
    return comparisons
