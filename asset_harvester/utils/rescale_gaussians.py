#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert 3DGS PLY assets between the NuRec-normalized frame and simulation frame.

Forward mode:
  gaussians.ply -> gaussians_sim.ply
  Rescales normalized 3DGS assets to real-world dimensions using lwh.txt and
  converts coordinates from (x-right, y-down, z-forward) to
  (x-forward, y-left, z-up).

Reverse mode:
  gaussians_sim.ply -> gaussians_nurec.ply
  Applies the inverse coordinate transform and re-normalizes the asset using
  the axis-aligned extent of gaussians_sim.ply itself, without depending on
  lwh.txt or any other sidecar file.

Usage:
    python utils/rescale_gaussians.py --output-dir <harvest_output_dir>
    python utils/rescale_gaussians.py --output-dir <harvest_output_dir> --mode reverse
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Re-use the project's own PLY I/O to avoid extra dependencies.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "tokengs"))
from asset_harvester.tokengs.ply_io import read_ply, write_ply

# Coordinate transform: (x-right, y-down, z-forward) -> (x-forward, y-left, z-up)
#   new_x = old_z,  new_y = -old_x,  new_z = -old_y
_COORD_TRANSFORM = np.array(
    [[0, 0, 1],
     [-1, 0, 0],
     [0, -1, 0]],
    dtype=np.float64,
)
_COORD_TRANSFORM_INV = _COORD_TRANSFORM.T

# Unit quaternion for _COORD_TRANSFORM in (w, x, y, z) order.
_COORD_TRANSFORM_QUAT_WXYZ = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)
_COORD_TRANSFORM_INV_QUAT_WXYZ = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float64)


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in (w, x, y, z) order."""
    lw, lx, ly, lz = lhs
    rw = rhs[:, 0]
    rx = rhs[:, 1]
    ry = rhs[:, 2]
    rz = rhs[:, 3]
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=1,
    )


def _transform_quaternions(props: dict[str, np.ndarray], transform_quat_wxyz: np.ndarray) -> None:
    """Left-multiply Gaussian quaternions by a frame-transform quaternion."""
    quats_wxyz = np.stack(
        [props["rot_0"], props["rot_1"], props["rot_2"], props["rot_3"]], axis=1
    ).astype(np.float64)
    norms = np.linalg.norm(quats_wxyz, axis=1, keepdims=True)
    quats_wxyz = quats_wxyz / np.maximum(norms, 1e-12)
    new_quats_wxyz = _quat_multiply_wxyz(transform_quat_wxyz, quats_wxyz)
    new_quats_wxyz = new_quats_wxyz / np.maximum(np.linalg.norm(new_quats_wxyz, axis=1, keepdims=True), 1e-12)

    props["rot_0"] = new_quats_wxyz[:, 0].astype(np.float32)
    props["rot_1"] = new_quats_wxyz[:, 1].astype(np.float32)
    props["rot_2"] = new_quats_wxyz[:, 2].astype(np.float32)
    props["rot_3"] = new_quats_wxyz[:, 3].astype(np.float32)


def _load_scale_factor(lwh_path: str) -> float:
    lwh = np.loadtxt(lwh_path, dtype=np.float64)
    if lwh.size != 3:
        raise ValueError(f"Expected 3 values in {lwh_path}, got {lwh.size}")

    scale_factor = float(np.max(lwh))
    if scale_factor <= 0:
        raise ValueError(f"Invalid max(lwh) = {scale_factor} from {lwh_path}")
    return scale_factor


def _infer_scale_factor_from_sim_ply(props: dict[str, np.ndarray]) -> float:
    """Infer the normalization scale from a simulation-frame asset's AABB size."""
    xyz = np.stack([props["x"], props["y"], props["z"]], axis=1).astype(np.float64)
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    scale_factor = float(np.max(extents))
    if scale_factor <= 0:
        raise ValueError(f"Invalid max AABB extent = {scale_factor} inferred from gaussians_sim.ply")
    return scale_factor


def _apply_coord_transform(props: dict[str, np.ndarray], transform: np.ndarray) -> None:
    """Apply a 3x3 coordinate transform to Gaussian centers in-place."""
    xyz = np.stack([props["x"], props["y"], props["z"]], axis=1).astype(np.float64)
    xyz = np.einsum("ij,nj->ni", transform, xyz).astype(np.float32)
    props["x"] = xyz[:, 0]
    props["y"] = xyz[:, 1]
    props["z"] = xyz[:, 2]


def rescale_ply(ply_path: str, lwh_path: str, out_path: str) -> None:
    """Convert gaussians.ply to gaussians_sim.ply."""
    scale_factor = _load_scale_factor(lwh_path)

    props = read_ply(ply_path)
    count = len(next(iter(props.values())))

    # 1. Rescale positions to metric units.
    for k in ("x", "y", "z"):
        props[k] = (props[k].astype(np.float64) * scale_factor).astype(np.float32)

    # 2. Rescale Gaussian scales in log-space: log(s * f) = log(s) + log(f).
    log_sf = np.float32(np.log(scale_factor))
    for k in ("scale_0", "scale_1", "scale_2"):
        props[k] = props[k] + log_sf

    # 3. Convert coordinates: (x-right, y-down, z-forward) -> (x-forward, y-left, z-up).
    _apply_coord_transform(props, _COORD_TRANSFORM)

    # 4. Rotate Gaussian orientations into the simulation frame.
    _transform_quaternions(props, _COORD_TRANSFORM_QUAT_WXYZ)

    write_ply(out_path, count, props)


def inverse_rescale_ply(ply_path: str, out_path: str) -> None:
    """Convert gaussians_sim.ply back to gaussians_nurec.ply."""
    props = read_ply(ply_path)
    count = len(next(iter(props.values())))
    scale_factor = _infer_scale_factor_from_sim_ply(props)

    # 1. Undo the simulation-frame coordinate transform.
    _apply_coord_transform(props, _COORD_TRANSFORM_INV)

    # 2. Rotate Gaussian orientations back into the normalized NuRec frame.
    _transform_quaternions(props, _COORD_TRANSFORM_INV_QUAT_WXYZ)

    # 3. Undo metric scaling on positions.
    for k in ("x", "y", "z"):
        props[k] = (props[k].astype(np.float64) / scale_factor).astype(np.float32)

    # 4. Undo metric scaling on Gaussian scales in log-space.
    log_sf = np.float32(np.log(scale_factor))
    for k in ("scale_0", "scale_1", "scale_2"):
        props[k] = props[k] - log_sf

    write_ply(out_path, count, props)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 3DGS assets between normalized NuRec and simulation coordinate frames"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Harvest output directory (contains <class>/<id>/ sub-folders).",
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "reverse"),
        default="forward",
        help="forward: gaussians.ply -> gaussians_sim.ply using lwh.txt; reverse: gaussians_sim.ply -> gaussians_nurec.ply using inferred AABB scale",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    input_name = "gaussians.ply" if args.mode == "forward" else "gaussians_sim.ply"
    output_name = "gaussians_sim.ply" if args.mode == "forward" else "gaussians_nurec.ply"
    convert_fn = rescale_ply if args.mode == "forward" else inverse_rescale_ply

    ply_files = sorted(root.rglob(input_name))
    if not ply_files:
        print(f"No {input_name} files found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ply_files)} {input_name} file(s) under {root}")

    failures = 0
    for ply_path in ply_files:
        asset_dir = ply_path.parent
        lwh_path = asset_dir / "multiview" / "lwh.txt"
        out_path = asset_dir / output_name

        if args.mode == "forward" and not lwh_path.is_file():
            print(f"  SKIP {asset_dir.relative_to(root)}: missing {lwh_path.name}")
            continue

        try:
            if args.mode == "forward":
                convert_fn(str(ply_path), str(lwh_path), str(out_path))
            else:
                convert_fn(str(ply_path), str(out_path))
            print(f"  OK   {out_path.relative_to(root)}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {asset_dir.relative_to(root)}: {e}", file=sys.stderr)

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()