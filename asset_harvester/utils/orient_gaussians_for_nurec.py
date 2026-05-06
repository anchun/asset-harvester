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

"""Post-process Asset Harvester Gaussian PLYs for NuRec insertion."""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from asset_harvester.tokengs.ply_io import read_ply, write_ply

log = logging.getLogger(__name__)

PLY_FILENAME = "gaussians.ply"
SH_C0 = 0.28209479177387814


def apply_y_rotation_gaussians(gaussians: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate [1, N, 14] Gaussians around the Y axis."""
    rad = np.deg2rad(float(degrees))
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)
    device = gaussians.device
    g = gaussians.float().to(device).clone()

    xyz = g[0, :, 0:3].detach().cpu().numpy()
    g[0, :, 0:3] = torch.from_numpy(xyz @ R.T).float().to(device)

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


def load_gaussians_ply(path: Path) -> torch.Tensor:
    """Load a compatible Gaussian PLY file into an activated [1, N, 14] tensor."""
    vertex = read_ply(str(path))

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    opacity = vertex["opacity"][:, np.newaxis]
    opacity = 1.0 / (1.0 + np.exp(-opacity))
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1)
    scales = np.exp(scales)
    rotations = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1)
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1)
    rgb = np.clip(f_dc * SH_C0 + 0.5, 0, 1)

    gaussians = np.concatenate([xyz, opacity, scales, rotations, rgb], axis=-1)
    return torch.from_numpy(gaussians).float().unsqueeze(0)


def find_gaussian_plys(input_dir: Path) -> list[Path]:
    """Return generated Gaussian PLY files below an Asset Harvester output directory."""
    return sorted(input_dir.rglob(PLY_FILENAME))


def write_gaussians_ply(path: Path, gaussians: torch.Tensor) -> None:
    """Write an activated [1, N, 14] Gaussian tensor back to compatible PLY attributes."""
    if gaussians.shape[0] != 1 or gaussians.shape[-1] != 14:
        raise ValueError(f"Expected gaussians with shape [1, N, 14], got {tuple(gaussians.shape)}")

    g = gaussians[0].detach().cpu().float()
    xyz = g[:, 0:3].numpy()
    opacity = torch.logit(g[:, 3:4].clamp(1e-8, 1 - 1e-8)).numpy()
    scales = torch.log(g[:, 4:7] + 1e-8).numpy()
    rotations = g[:, 7:11].numpy()
    f_dc = ((g[:, 11:14] - 0.5) / SH_C0).numpy()

    props = OrderedDict()
    props["x"] = xyz[:, 0].astype(np.float32)
    props["y"] = xyz[:, 1].astype(np.float32)
    props["z"] = xyz[:, 2].astype(np.float32)
    for i in range(f_dc.shape[1]):
        props[f"f_dc_{i}"] = f_dc[:, i].astype(np.float32)
    props["opacity"] = opacity[:, 0].astype(np.float32)
    for i in range(scales.shape[1]):
        props[f"scale_{i}"] = scales[:, i].astype(np.float32)
    for i in range(rotations.shape[1]):
        props[f"rot_{i}"] = rotations[:, i].astype(np.float32)

    write_ply(str(path), xyz.shape[0], props)


def prepare_output_dir(input_dir: Path, output_dir: Path | None, in_place: bool) -> Path:
    """Create or select the directory whose PLY files will be transformed."""
    if in_place:
        return input_dir
    if output_dir is None:
        raise ValueError("Pass either --output-dir to create a transformed copy or --in-place to overwrite input PLYs")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    shutil.copytree(input_dir, output_dir)
    return output_dir


def orient_gaussians_for_nurec(
    input_dir: Path,
    output_dir: Path | None,
    degrees: float,
    in_place: bool,
) -> int:
    """Rotate generated Gaussian PLYs around Y for NuRec insertion."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    target_dir = prepare_output_dir(input_dir, output_dir, in_place)
    ply_paths = find_gaussian_plys(target_dir)
    if not ply_paths:
        log.warning(f"No {PLY_FILENAME} files found under {target_dir}")
        return 0

    for ply_path in ply_paths:
        gaussians = load_gaussians_ply(ply_path)
        rotated = apply_y_rotation_gaussians(gaussians, degrees)
        write_gaussians_ply(ply_path, rotated)
        log.info(f"Rotated {ply_path.relative_to(target_dir)} by {degrees:g} degrees around Y")

    return len(ply_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate Asset Harvester Gaussian PLYs for NuRec insertion.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root Asset Harvester output directory from run_inference.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to create as a transformed copy of --input-dir.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite gaussians.ply files in --input-dir instead of writing a transformed copy.",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=90.0,
        help="Y-axis rotation to apply.",
    )
    args = parser.parse_args()

    if args.output_dir is not None and args.in_place:
        raise ValueError("Use either --output-dir or --in-place, not both")

    count = orient_gaussians_for_nurec(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        degrees=args.degrees,
        in_place=args.in_place,
    )
    log.info(f"Processed {count} Gaussian PLY file(s)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
