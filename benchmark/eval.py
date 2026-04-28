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
Evaluate run_inference.py output: load gaussians from each sample folder,
render at reserved_views cameras, and compare with reserved_views images using
PSNR, LPIPS, SSIM, and patch embedding metrics.

Usage:
  python eval.py --output_dir /path/to/run_inference_output
  (e.g. outputs/multiview)

Expects: output_dir/class_name/sample_id/gaussians.ply, reserved_views.json.
reserved_views.json points to the folder with camera.json and frames.

Before metrics, the gaussians are Y-rotated by 0/90/180/270° (whichever minimizes
mean LPIPS vs GT after mask alignment). Use --rotation_align_views 1 to score
only the first reserved view (faster). Use --no_rotation_align to skip this
search and evaluate at the model's native orientation (0°).

The render/GT background defaults to mid-grey (0.5, 0.5, 0.5); pass
--background white to use white instead.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from embedding_metrics import (
    DINOv3PatchEmbedder,
    KeypointBodyPartSegmenter,
    SAM3DBodyKeypointDetector,
    compute_embedding_metrics_patch,
    load_dinov3_backbone,
)
from PIL import Image
from tqdm import tqdm
from utils import (
    align_rendered_to_gt_masks,
    apply_y_rotation_gaussians,
    compute_metrics,
    create_comparison_image_from_metrics,
    load_ply,
    render_from_cameras,
)


def _safe_mean(values):
    """Mean of a sequence, ignoring inf and nan. Returns np.nan if no finite values."""
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.nan
    return np.mean(arr[finite])


def _safe_fmt(x):
    """Format a scalar for display; handle inf/nan."""
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    if np.isnan(x):
        return "nan"
    return x


def find_best_rotation(
    data,
    output_size,
    view_indices=None,
    bg_color=(1.0, 1.0, 1.0),
    border_rgb=(255, 255, 255),
):
    """
    Find the Y-axis rotation in {0, 90, 180, 270} degrees that minimizes
    misalignment with GT by rendering at reserved views, aligning to GT masks,
    and comparing. Uses mean LPIPS (lower is better) over the specified views.

    Args:
        data: sample dict from load_sample_from_run_inference_output (3DGS).
        output_size: int, render resolution.
        view_indices: list of int or None. If None, use all reserved views; else
            use only these view indices for the alignment score (e.g. [0]).
        bg_color: RGB in [0, 1] used for renders and GT compositing.
        border_rgb: (R, G, B) 0-255 used for warpAffine border fill.

    Returns:
        best_angle_deg: int in {0, 90, 180, 270}.
    """
    n_views = len(data["frames"])
    if view_indices is None:
        view_indices = list(range(n_views))
    else:
        view_indices = [i for i in view_indices if 0 <= i < n_views]
    if not view_indices:
        return 0

    best_angle = 0
    best_lpips = float("inf")
    bg_u8 = np.array(bg_color, dtype=np.float32) * 255.0

    for angle_deg in (0, 90, 180, 270):
        with torch.no_grad():
            g = apply_y_rotation_gaussians(data["gaussians"].clone(), angle_deg)
            rendered_images, rendered_alphas = render_from_cameras(
                g,
                cam_poses=data["cam_poses"],
                dists=data["dists"],
                fovs=data["fovs"],
                lwh=data["lwh"],
                output_size=output_size,
                bg_color=bg_color,
            )
            rendered_images, rendered_alphas = align_rendered_to_gt_masks(
                rendered_images,
                rendered_alphas,
                data["masks"],
                output_size,
                quiet=True,
                border_rgb=border_rgb,
            )
        masked_gt_frames = []
        for frame, mask in zip(data["frames"], data["masks"]):
            m = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            masked = (frame.astype(np.float32) * m + bg_u8 * (1.0 - m)).clip(0, 255).astype(np.uint8)
            masked_gt_frames.append(masked)
        metrics = compute_metrics(rendered_images, masked_gt_frames)
        lpips_vals = [metrics["lpips"][i] for i in view_indices]
        mean_lpips = float(np.nanmean(lpips_vals)) if lpips_vals else float("inf")
        if mean_lpips < best_lpips:
            best_lpips = mean_lpips
            best_angle = angle_deg

    return best_angle


def load_sample_from_run_inference_output(sample_dir):
    """
    Load gaussians from a run_inference output sample folder and GT from
    reserved_views (path from reserved_views.json). Same dict shape as load_sample.

    sample_dir: path to output_dir/class_name/sample_00000/ (contains gaussians.ply, reserved_views.json)
    """
    ply_path = os.path.join(sample_dir, "gaussians.ply")
    if not os.path.isfile(ply_path):
        print(f"No gaussians.ply in {sample_dir}")
        return None
    json_path = os.path.join(sample_dir, "reserved_views.json")
    if not os.path.isfile(json_path):
        print(f"No reserved_views.json in {sample_dir}")
        return None
    with open(json_path) as f:
        data = json.load(f)
    reserved_views_path = data.get("reserved_views")
    if not reserved_views_path or not os.path.isdir(reserved_views_path):
        print(f"Invalid reserved_views path: {reserved_views_path}")
        return None
    camera_path = os.path.join(reserved_views_path, "camera.json")
    if not os.path.isfile(camera_path):
        print(f"No camera.json in {reserved_views_path}")
        return None
    with open(camera_path) as f:
        cam_data = json.load(f)
    frame_filenames = cam_data["frame_filenames"]
    mask_filenames = cam_data.get("mask_filenames", [])
    cam_poses_raw = cam_data["normalized_cam_positions"]
    dists_list = cam_data["cam_dists"]
    fov_list = cam_data["cam_fovs"]
    lwh_list = cam_data["object_lwh"]
    n = len(frame_filenames)
    if n == 0:
        print(f"No frames in reserved_views: {reserved_views_path}")
        return None
    frames = []
    masks = []
    for i, fn in enumerate(frame_filenames):
        path = os.path.join(reserved_views_path, fn)
        if not os.path.isfile(path):
            print(f"Missing frame: {path}")
            return None
        img = Image.open(path).convert("RGB")
        frames.append(np.array(img))
        if i < len(mask_filenames):
            mask_path = os.path.join(reserved_views_path, mask_filenames[i])
            if os.path.isfile(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                masks.append(mask)
            else:
                masks.append(np.ones((frames[-1].shape[0], frames[-1].shape[1]), dtype=np.uint8) * 255)
        else:
            masks.append(np.ones((frames[-1].shape[0], frames[-1].shape[1]), dtype=np.uint8) * 255)
    cam_poses = np.array(cam_poses_raw, dtype=np.float64).reshape(n, 3)
    dists = np.array(dists_list, dtype=np.float64).reshape(n)
    fovs = np.array(fov_list, dtype=np.float64).reshape(n)
    lwh = np.asarray(lwh_list, dtype=np.float64).reshape(3)
    gaussians = load_ply(ply_path).cuda()
    return {
        "gaussians": gaussians,
        "cam_poses": cam_poses,
        "dists": dists,
        "fovs": fovs,
        "lwh": lwh,
        "frames": frames,
        "masks": masks,
        "sample_folder": sample_dir,
        "ply_path": ply_path,
        "reserved_views_path": reserved_views_path,
    }


def process_run_inference_sample(
    sample_dir,
    output_folder,
    output_size=512,
    save_comparisons=True,
    body_part_segmenter=None,
    patch_embedder=None,
    vis_debug=False,
    enable_rotation_alignment=True,
    rotation_align_view_indices=None,
    bg_color=(0.5, 0.5, 0.5),
    border_rgb=(128, 128, 128),
):
    """
    Process one run_inference output sample: load gaussians and reserved_views,
    render at reserved_views cameras, compute metrics vs reserved_views images.

    sample_dir: path to output_dir/class_name/sample_00000/
    output_folder: where to write metrics and comparison images
    """
    print(f"\nProcessing: {sample_dir}")
    data = load_sample_from_run_inference_output(sample_dir)
    if data is None:
        return None
    print(f"  PLY: {data['ply_path']}")
    print(f"  Reserved views: {data['reserved_views_path']} ({len(data['frames'])} frames)")
    num_views = len(data["frames"])

    # Y-rotation alignment: pick the angle in {0, 90, 180, 270} that minimizes
    # mean LPIPS vs GT after mask alignment.
    if enable_rotation_alignment:
        rot_deg = find_best_rotation(
            data,
            output_size,
            view_indices=rotation_align_view_indices,
            bg_color=bg_color,
            border_rgb=border_rgb,
        )
        if rot_deg != 0:
            print(f"  Rotation alignment: using {rot_deg}° (best of 0, 90, 180, 270)")
    else:
        rot_deg = 0
        print("  Rotation alignment: disabled (using 0°)")

    with torch.no_grad():
        gaussians_render = (
            apply_y_rotation_gaussians(data["gaussians"].clone(), rot_deg)
            if rot_deg != 0
            else data["gaussians"]
        )
        rendered_images, rendered_alphas = render_from_cameras(
            gaussians_render,
            cam_poses=data["cam_poses"],
            dists=data["dists"],
            fovs=data["fovs"],
            lwh=data["lwh"],
            output_size=output_size,
            bg_color=bg_color,
        )
        # Align rendered images to GT masks (scale + translate so rendered foreground matches GT bbox)
        rendered_images, rendered_alphas = align_rendered_to_gt_masks(
            rendered_images,
            rendered_alphas,
            data["masks"],
            output_size,
            border_rgb=border_rgb,
        )

    print("  Computing metrics...")
    # Mask GT frames with the chosen background so we pass pre-masked images
    masked_gt_frames = []
    bg_u8 = np.array(bg_color, dtype=np.float32) * 255.0
    for frame, mask in zip(data["frames"], data["masks"]):
        m = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        masked = (frame.astype(np.float32) * m + bg_u8 * (1.0 - m)).clip(0, 255).astype(np.uint8)
        masked_gt_frames.append(masked)
    metrics = compute_metrics(rendered_images, masked_gt_frames)
    mean_psnr = _safe_mean(metrics["psnr"])
    mean_lpips = _safe_mean(metrics["lpips"])
    mean_ssim = _safe_mean(metrics["ssim"])
    metrics["mean_psnr"] = mean_psnr
    metrics["mean_lpips"] = mean_lpips
    metrics["mean_ssim"] = mean_ssim
    metrics["rotation_align_deg"] = rot_deg
    metrics["rotation_alignment_searched"] = bool(enable_rotation_alignment)
    if np.isfinite(mean_psnr):
        print(f"  Mean PSNR: {mean_psnr:.2f}")
    else:
        print(f"  Mean PSNR: {_safe_fmt(mean_psnr)}")
    if np.isfinite(mean_lpips):
        print(f"  Mean LPIPS: {mean_lpips:.4f}")
    else:
        print(f"  Mean LPIPS: {_safe_fmt(mean_lpips)}")
    if np.isfinite(mean_ssim):
        print(f"  Mean SSIM: {mean_ssim:.4f}")
    else:
        print(f"  Mean SSIM: {_safe_fmt(mean_ssim)}")

    os.makedirs(output_folder, exist_ok=True)

    is_person = "pedestrian" in sample_dir.lower()
    if patch_embedder is not None:
        if is_person and body_part_segmenter is None:
            print("  Skipping patch embedding metrics (person sample but no body part segmenter).")
        else:
            if is_person:
                label = type(body_part_segmenter).__name__
            else:
                label = "foreground mask"
            print(f"  Computing patch embedding metrics with {label}...")
            patch_metrics = compute_embedding_metrics_patch(
                rendered_images,
                rendered_alphas,
                data["frames"],
                data["masks"],
                patch_embedder,
                body_part_segmenter=body_part_segmenter,
                is_person=is_person,
                output_dir=output_folder if vis_debug else None,
                bg_color=bg_color,
            )
            metrics.update(patch_metrics)
            if "mean_emb_dist" in patch_metrics:
                v = patch_metrics["mean_emb_dist"]
                print(
                    f"  Mean embedding distance: {v:.4f}"
                    if np.isfinite(v)
                    else f"  Mean embedding distance: {_safe_fmt(v)}"
                )
            if "mean_emb_dist_part" in patch_metrics:
                v = patch_metrics["mean_emb_dist_part"]
                print(
                    f"  Mean embedding distance (parts): {v:.4f}"
                    if np.isfinite(v)
                    else f"  Mean embedding distance (parts): {_safe_fmt(v)}"
                )

    if save_comparisons:
        comparisons = create_comparison_image_from_metrics(metrics, output_size)
        for i, comp in enumerate(comparisons):
            comp_path = os.path.join(output_folder, f"render_comparison_{i:02d}.jpeg")
            comp.save(comp_path, quality=95)
        print(f"  Saved {len(comparisons)} comparison images to {output_folder}")
    return {
        "sample_folder": sample_dir,
        "output_folder": output_folder,
        "metrics": metrics,
        "num_frames": num_views,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate run_inference.py output: render gaussians at reserved_views cameras and compare with reserved_views images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory from run_inference.py (e.g. outputs/multiview). Expects class_name/sample_id/gaussians.ply and reserved_views.json.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="If set, only evaluate samples under this class (one subdir of output_dir). Used by launch_job.sh.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to evaluate per run (default: all). Used with --sample_seed for subsampling.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Random seed for subsampling when --max_samples is set (default: 0).",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default=None,
        help="Directory to write eval results (metrics, comparison images). Default: <output_dir>/eval",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=512,
        help="Rendering resolution (default: 512)",
    )
    parser.add_argument(
        "--no_comparisons",
        action="store_true",
        help="Do not save comparison images",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Only process samples matching this substring (e.g. '02837')",
    )
    parser.add_argument(
        "--vis_debug",
        action="store_true",
        default=False,
        help="Save intermediate debug visualizations (keypoints, patch overlays).",
    )
    parser.add_argument(
        "--rotation_align_views",
        type=str,
        default="all",
        choices=("all", "1"),
        help="Views to use for Y-rotation alignment search: 'all' (default) or '1' (first view only, faster). Ignored if --no_rotation_align.",
    )
    parser.add_argument(
        "--no_rotation_align",
        action="store_true",
        help="Do not search over Y rotations (0/90/180/270°); evaluate at native orientation (0°).",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="grey",
        choices=("grey", "white"),
        help="Background colour for renders, GT compositing, and alignment warp borders (default: grey).",
    )
    args = parser.parse_args()

    if args.background == "white":
        bg_color = (1.0, 1.0, 1.0)
    else:
        bg_color = (0.5, 0.5, 0.5)
    border_rgb = tuple(int(round(c * 255.0)) for c in bg_color)
    rotation_align_view_indices = None if args.rotation_align_views == "all" else [0]

    output_dir = os.path.abspath(args.output_dir)
    eval_output_dir = args.eval_output_dir
    if eval_output_dir is None:
        eval_output_dir = os.path.join(output_dir, "eval")
    eval_output_dir = os.path.abspath(eval_output_dir)

    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        return

    # Find sample folders: output_dir/class_name/<any>/ with gaussians.ply and reserved_views.json
    sample_folders = []
    for class_name in sorted(os.listdir(output_dir)):
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for sub in sorted(os.listdir(class_dir)):
            f = os.path.join(class_dir, sub)
            if not os.path.isdir(f):
                continue
            has_ply = os.path.isfile(os.path.join(f, "gaussians.ply"))
            has_rv = os.path.isfile(os.path.join(f, "reserved_views.json"))
            if has_ply and has_rv:
                if args.sample is not None and args.sample not in f:
                    continue
                sample_folders.append(f)
            elif has_ply or has_rv:
                missing = "reserved_views.json" if not has_rv else "gaussians.ply"
                print(f"   Skipping {f}: missing {missing}")
    sample_folders = sorted(sample_folders)
    print(f"Scanning: {output_dir}")
    if args.class_name is not None:
        sample_folders = [f for f in sample_folders if os.path.basename(os.path.dirname(f)) == args.class_name]
        print(f"Filtered to class '{args.class_name}': {len(sample_folders)} samples")
    if args.max_samples is not None and args.max_samples > 0 and len(sample_folders) > args.max_samples:
        rng = np.random.default_rng(args.sample_seed)
        indices = rng.choice(len(sample_folders), size=args.max_samples, replace=False)
        sample_folders = [sample_folders[i] for i in sorted(indices)]
        print(f"Subsampled to {len(sample_folders)} samples (max_samples={args.max_samples}, seed={args.sample_seed})")
    print(f"Found {len(sample_folders)} samples with gaussians.ply and reserved_views.json")
    print(f"Eval results will be saved to: {eval_output_dir}")

    if not sample_folders:
        print("No samples found. Each sample must have gaussians.ply and reserved_views.json.")
        return

    # Optional: keypoint detector and embedders for embedding metrics
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sam3d_ckpt = os.environ.get(
        "SAM3D_CKPT",
        os.path.join(script_dir, "checkpoints", "sam-3d-body-dinov3", "model.ckpt"),
    )
    sam3d_mhr = os.environ.get(
        "SAM3D_MHR_PATH",
        os.path.join(script_dir, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt"),
    )
    body_part_segmenter = None
    patch_embedder = None
    if os.path.isfile(sam3d_ckpt):
        try:
            logging.getLogger("sam_3d_body.utils.checkpoint").setLevel(logging.ERROR)
            keypoint_detector = SAM3DBodyKeypointDetector(
                checkpoint_path=sam3d_ckpt,
                mhr_path=sam3d_mhr,
            )
            pretrained_backbone = load_dinov3_backbone()
            patch_embedder = DINOv3PatchEmbedder(
                backbone=pretrained_backbone,
                input_size=512,
            )
            print(f"Using SAM 3D Body detector: {sam3d_ckpt}")
            body_part_segmenter = KeypointBodyPartSegmenter(
                keypoint_detector=keypoint_detector,
                patch_size=16,
                confidence_thr=0.3,
                max_patch_dist=2,
            )
            print("Using keypoint skeleton-proximity body-part segmentation.")
        except Exception as e:
            print(f"SAM 3D Body / DINOv3 not available, skipping embedding metrics: {e}")
    else:
        print(f"SAM 3D Body checkpoint not found at {sam3d_ckpt}, skipping embedding metrics.")

    os.makedirs(eval_output_dir, exist_ok=True)
    all_results = []
    all_psnr = []
    all_lpips = []
    all_ssim = []
    all_emb_dists = {}

    for sample_folder in tqdm(sample_folders, desc="Samples", unit="sample"):
        try:
            rel_path = os.path.relpath(sample_folder, output_dir)
            out_folder = os.path.join(eval_output_dir, rel_path)
            result = process_run_inference_sample(
                sample_folder,
                out_folder,
                output_size=args.output_size,
                save_comparisons=not args.no_comparisons,
                body_part_segmenter=body_part_segmenter,
                patch_embedder=patch_embedder,
                vis_debug=args.vis_debug,
                enable_rotation_alignment=not args.no_rotation_align,
                rotation_align_view_indices=rotation_align_view_indices,
                bg_color=bg_color,
                border_rgb=border_rgb,
            )
            if result is not None:
                all_results.append(result)
                all_psnr.extend(result["metrics"]["psnr"])
                all_lpips.extend(result["metrics"]["lpips"])
                all_ssim.extend(result["metrics"]["ssim"])
                if "emb_dist" in result["metrics"]:
                    all_emb_dists.setdefault("emb", []).extend(result["metrics"]["emb_dist"])
                if "emb_dist_part" in result["metrics"]:
                    all_emb_dists.setdefault("emb_part", []).extend(result["metrics"]["emb_dist_part"])
        except Exception as e:
            print(f"  Error processing {sample_folder}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed {len(all_results)} samples")
    if all_psnr:
        m_psnr, m_lpips, m_ssim = _safe_mean(all_psnr), _safe_mean(all_lpips), _safe_mean(all_ssim)
        if np.isfinite(m_psnr):
            print(f"Overall Mean PSNR: {m_psnr:.2f} dB")
        else:
            print(f"Overall Mean PSNR: {_safe_fmt(m_psnr)}")
        if np.isfinite(m_lpips):
            print(f"Overall Mean LPIPS: {m_lpips:.4f}")
        else:
            print(f"Overall Mean LPIPS: {_safe_fmt(m_lpips)}")
        if np.isfinite(m_ssim):
            print(f"Overall Mean SSIM: {m_ssim:.4f}")
        else:
            print(f"Overall Mean SSIM: {_safe_fmt(m_ssim)}")
    for emb_name, dists in all_emb_dists.items():
        if dists:
            m = _safe_mean(dists)
            if np.isfinite(m):
                print(f"Overall Mean {emb_name} Embedding Distance: {m:.4f}")
            else:
                print(f"Overall Mean {emb_name} Embedding Distance: {_safe_fmt(m)}")

    summary_path = os.path.join(eval_output_dir, "rendering_metrics_summary.txt")
    if args.class_name is not None:
        safe_name = args.class_name.replace(os.path.sep, "_")
        summary_path = os.path.join(eval_output_dir, f"rendering_metrics_summary_{safe_name}.txt")
    with open(summary_path, "w") as f:
        f.write("Rendering Metrics Summary (run_inference output vs reserved_views)\n")
        f.write("=" * 60 + "\n\n")
        for result in all_results:
            m = result["metrics"]
            f.write(f"Sample: {result['sample_folder']}\n")
            f.write(f"Output: {result['output_folder']}\n")
            f.write(f"  Frames: {result['num_frames']}\n")
            f.write(f"  Rotation align (deg): {m.get('rotation_align_deg', 0)}\n")
            v = m["mean_psnr"]
            f.write(f"  Mean PSNR: {v:.2f} dB\n" if np.isfinite(v) else f"  Mean PSNR: {_safe_fmt(v)}\n")
            v = m["mean_lpips"]
            f.write(f"  Mean LPIPS: {v:.4f}\n" if np.isfinite(v) else f"  Mean LPIPS: {_safe_fmt(v)}\n")
            v = m["mean_ssim"]
            f.write(f"  Mean SSIM: {v:.4f}\n" if np.isfinite(v) else f"  Mean SSIM: {_safe_fmt(v)}\n")
            f.write(f"  Per-frame PSNR: {[_safe_fmt(x) for x in m['psnr']]}\n")
            f.write(f"  Per-frame LPIPS: {[_safe_fmt(x) for x in m['lpips']]}\n")
            f.write(f"  Per-frame SSIM: {[_safe_fmt(x) for x in m['ssim']]}\n")
            if "emb_dist" in m:
                v = m["mean_emb_dist"]
                f.write(f"  Mean Emb Dist: {v:.4f}\n" if np.isfinite(v) else f"  Mean Emb Dist: {_safe_fmt(v)}\n")
                f.write(f"  Per-frame Emb Dist: {[_safe_fmt(x) for x in m['emb_dist']]}\n")
            if "emb_dist_part" in m:
                v = m["mean_emb_dist_part"]
                f.write(
                    f"  Mean Emb Dist (parts): {v:.4f}\n"
                    if np.isfinite(v)
                    else f"  Mean Emb Dist (parts): {_safe_fmt(v)}\n"
                )
                f.write(f"  Per-frame Emb Dist (parts): {[_safe_fmt(x) for x in m['emb_dist_part']]}\n")
            f.write("\n")
        f.write("=" * 60 + "\n")
        if all_psnr:
            m_psnr = _safe_mean(all_psnr)
            m_lpips = _safe_mean(all_lpips)
            m_ssim = _safe_mean(all_ssim)
            f.write(
                f"Overall Mean PSNR: {m_psnr:.2f} dB\n"
                if np.isfinite(m_psnr)
                else f"Overall Mean PSNR: {_safe_fmt(m_psnr)}\n"
            )
            f.write(
                f"Overall Mean LPIPS: {m_lpips:.4f}\n"
                if np.isfinite(m_lpips)
                else f"Overall Mean LPIPS: {_safe_fmt(m_lpips)}\n"
            )
            f.write(
                f"Overall Mean SSIM: {m_ssim:.4f}\n"
                if np.isfinite(m_ssim)
                else f"Overall Mean SSIM: {_safe_fmt(m_ssim)}\n"
            )
        for emb_name, dists in all_emb_dists.items():
            if dists:
                m = _safe_mean(dists)
                f.write(
                    f"Overall Mean {emb_name} Embedding Distance: {m:.4f}\n"
                    if np.isfinite(m)
                    else f"Overall Mean {emb_name} Embedding Distance: {_safe_fmt(m)}\n"
                )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
