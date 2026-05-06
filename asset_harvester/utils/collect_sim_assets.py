#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect simulation-ready Gaussian assets into a single directory with thumbnails.

This script can consume ``confidence.json`` files produced by
``asset_harvester.utils.postprocess_clip_confidence`` and apply CLIP-based
filtering while collecting simulation assets.

Typical usage
-------------

        python asset_harvester/utils/collect_sim_assets.py \
                --input-dir ./outputs/ncore_harvest/<clip_uuid> \
                --output-dir ./outputs/assets \
                --confidence-threshold 0.7

Behavior with ``--confidence-threshold`` enabled:

- keep a sample if ``mean_target_confidence >= threshold``
- keep a sample in its original class if ``majority_predicted_class`` still
    matches the original class and ``mean_predicted_confidence > threshold``
- redirect a sample to the predicted class directory if
    ``majority_predicted_class`` differs from the original class and
    ``mean_predicted_confidence > threshold``
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asset_harvester.tokengs.ply_io import read_ply


_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_CONFIDENCE_METRIC = "mean_target_confidence"


class ConfidenceEntry(dict):
    pass


def _infer_group_name(ply_path: Path, input_dir: Path) -> str:
    asset_dir = ply_path.parent
    if asset_dir.parent != input_dir and asset_dir.parent.name:
        return asset_dir.parent.name
    if input_dir.name:
        return input_dir.name
    return "ungrouped"


def _extract_uuid_prefix(ply_path: Path) -> str | None:
    for part in ply_path.parts:
        if _UUID_PATTERN.fullmatch(part):
            return part[:8].lower()
    return None


def _build_output_stem(ply_path: Path) -> str:
    uuid_prefix = _extract_uuid_prefix(ply_path)
    asset_id = ply_path.parent.name or "asset"
    if uuid_prefix:
        return f"{uuid_prefix}_{asset_id}"
    return asset_id


def _resize_with_padding(frame_rgb: np.ndarray, image_size: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid frame size")

    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    background = np.full((image_size, image_size, 3), 247, dtype=np.uint8)
    y_offset = (image_size - resized_height) // 2
    x_offset = (image_size - resized_width) // 2
    background[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return background


def extract_video_thumbnail(video_path: Path, thumbnail_path: Path, image_size: int, frame_number: int) -> bool:
    if not video_path.is_file():
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_index = max(frame_number - 1, 0)
    if frame_count > 0:
        target_index = min(target_index, frame_count - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return False

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    thumb = _resize_with_padding(frame_rgb, image_size)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(thumbnail_path), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect gaussians_sim.ply simulation assets into type-grouped output folders and generate thumbnails"
    )
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "outputs" / "ncore_harvest"),
        help="Directory to scan for gaussians_sim.ply files (default: outputs/ncore_harvest)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "assets"),
        help="Directory that will receive type-grouped collected simulation assets (default: outputs/assets)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Collect only the first N assets after sorting by relative path",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=512,
        help="Thumbnail size in pixels (default: 512)",
    )
    parser.add_argument(
        "--thumbnail-frame",
        type=int,
        default=50,
        help="Frame number to extract from 3d_lifted.mp4 for thumbnails (1-based, default: 50)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Maximum number of points used per thumbnail (default: 120000)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="If set, only collect assets whose sample entry in confidence.json meets or exceeds this threshold.",
    )
    return parser.parse_args()


def _require_positive(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _require_unit_interval(name: str, value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _find_clip_dir(asset_dir: Path, input_dir: Path) -> Path | None:
    for parent in (asset_dir, *asset_dir.parents):
        if parent == input_dir.parent:
            break
        if (parent / "confidence.json").is_file():
            return parent
    return None


def _compute_majority_share(prediction_histogram: dict[str, object]) -> tuple[str | None, float]:
    histogram: dict[str, int] = {}
    total = 0
    for label, value in prediction_histogram.items():
        if not isinstance(label, str) or not isinstance(value, int):
            continue
        histogram[label] = value
        total += value

    if not histogram or total <= 0:
        return None, 0.0

    majority_label, majority_count = max(histogram.items(), key=lambda item: item[1])
    return majority_label, majority_count / total


def _load_confidence_index(report_path: Path) -> dict[Path, ConfidenceEntry]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {report_path}")

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Missing 'samples' list in {report_path}")

    index: dict[Path, ConfidenceEntry] = {}
    clip_dir = report_path.parent.resolve()
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        class_name = sample.get("class_name")
        sample_id = sample.get("sample_id")
        metric_raw = sample.get(_CONFIDENCE_METRIC)
        predicted_metric_raw = sample.get("mean_predicted_confidence")
        majority_predicted_class = sample.get("majority_predicted_class")
        prediction_histogram = sample.get("prediction_histogram")
        if not isinstance(class_name, str) or not isinstance(sample_id, str):
            continue
        if not isinstance(metric_raw, (int, float)):
            continue
        if not isinstance(predicted_metric_raw, (int, float)):
            predicted_metric_raw = 0.0
        if not isinstance(prediction_histogram, dict):
            prediction_histogram = {}
        majority_label, majority_share = _compute_majority_share(prediction_histogram)
        if not isinstance(majority_predicted_class, str):
            majority_predicted_class = majority_label
        index[(clip_dir / class_name / sample_id).resolve()] = ConfidenceEntry(
            metric_value=float(metric_raw),
            predicted_metric_value=float(predicted_metric_raw),
            original_class=class_name,
            majority_predicted_class=majority_predicted_class,
            majority_share=majority_share,
        )
    return index


def _filter_candidates_by_confidence(
    candidates: list[Path],
    input_dir: Path,
    threshold: float,
) -> tuple[list[Path], list[tuple[Path, str]], int, dict[Path, str]]:
    filtered: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    report_cache: dict[Path, dict[Path, ConfidenceEntry]] = {}
    missing_report_count = 0
    redirect_groups: dict[Path, str] = {}

    for ply_path in candidates:
        asset_dir = ply_path.parent.resolve()
        clip_dir = _find_clip_dir(asset_dir, input_dir)
        if clip_dir is None:
            skipped.append((ply_path, "missing confidence.json in asset ancestry"))
            missing_report_count += 1
            continue

        if clip_dir not in report_cache:
            report_cache[clip_dir] = _load_confidence_index(clip_dir / "confidence.json")

        entry = report_cache[clip_dir].get(asset_dir)
        if entry is None:
            skipped.append((ply_path, f"sample not found in {clip_dir / 'confidence.json'}"))
            continue
        score = float(entry["metric_value"])
        majority_predicted_class = entry.get("majority_predicted_class")
        predicted_score = float(entry.get("predicted_metric_value", 0.0))
        original_class = entry.get("original_class")
        should_redirect = (
            isinstance(majority_predicted_class, str)
            and majority_predicted_class
            and predicted_score > threshold
            and majority_predicted_class != original_class
        )
        should_keep_original = (
            isinstance(majority_predicted_class, str)
            and majority_predicted_class == original_class
            and predicted_score > threshold
        )
        if should_redirect:
            redirect_groups[asset_dir] = majority_predicted_class

        if score < threshold and not should_redirect and not should_keep_original:
            skipped.append(
                (
                    ply_path,
                    f"{_CONFIDENCE_METRIC}={score:.3f} < {threshold:.3f} and mean_predicted_confidence={predicted_score:.3f}",
                )
            )
            continue

        filtered.append(ply_path)

    return filtered, skipped, missing_report_count, redirect_groups


def _extract_xyz(props: dict[str, np.ndarray]) -> np.ndarray:
    required = ("x", "y", "z")
    missing = [key for key in required if key not in props]
    if missing:
        raise ValueError(f"Missing coordinates in PLY: {', '.join(missing)}")
    return np.stack([props["x"], props["y"], props["z"]], axis=1).astype(np.float32)


def _extract_rgb(props: dict[str, np.ndarray], count: int) -> np.ndarray:
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(props):
        sh_dc = np.stack([props["f_dc_0"], props["f_dc_1"], props["f_dc_2"]], axis=1).astype(np.float32)
        rgb = sh_dc * np.float32(0.28209479177387814) + np.float32(0.5)
        return np.clip(rgb, 0.0, 1.0)

    if {"red", "green", "blue"}.issubset(props):
        rgb = np.stack([props["red"], props["green"], props["blue"]], axis=1).astype(np.float32)
        if rgb.max(initial=0.0) > 1.0:
            rgb /= 255.0
        return np.clip(rgb, 0.0, 1.0)

    return np.full((count, 3), 0.35, dtype=np.float32)


def _extract_weights(props: dict[str, np.ndarray], count: int) -> np.ndarray:
    weights = np.ones(count, dtype=np.float32)

    if "opacity" in props:
        opacity = props["opacity"].astype(np.float32)
        weights *= 1.0 / (1.0 + np.exp(-opacity))

    if {"scale_0", "scale_1", "scale_2"}.issubset(props):
        scales = np.stack([props["scale_0"], props["scale_1"], props["scale_2"]], axis=1).astype(np.float32)
        scale_weight = np.exp(np.clip(scales.mean(axis=1), -8.0, 8.0))
        scale_norm = np.percentile(scale_weight, 95) if scale_weight.size else 1.0
        if scale_norm <= 0:
            scale_norm = 1.0
        weights *= np.clip(scale_weight / scale_norm, 0.2, 4.0)

    weight_norm = np.percentile(weights, 95) if weights.size else 1.0
    if weight_norm <= 0:
        weight_norm = 1.0
    return np.clip(weights / weight_norm, 0.05, 1.0)


def _downsample(*arrays: np.ndarray, max_points: int) -> tuple[np.ndarray, ...]:
    if not arrays:
        return tuple()
    count = len(arrays[0])
    if count <= max_points:
        return arrays

    rng = np.random.default_rng(0)
    indices = rng.choice(count, size=max_points, replace=False)
    return tuple(arr[indices] for arr in arrays)


def _project_points(xyz: np.ndarray) -> np.ndarray:
    centered = xyz - np.median(xyz, axis=0, keepdims=True)
    if centered.shape[0] < 3:
        return centered[:, :2]

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T
        projected = centered @ basis
    except np.linalg.LinAlgError:
        projected = centered[:, :2]

    if np.allclose(projected.std(axis=0), 0.0):
        projected = centered[:, :2]
    return projected.astype(np.float32)


def render_thumbnail(
    ply_path: Path,
    thumbnail_path: Path,
    image_size: int,
    max_points: int,
) -> None:
    props = read_ply(str(ply_path))
    xyz = _extract_xyz(props)
    rgb = _extract_rgb(props, len(xyz))
    weights = _extract_weights(props, len(xyz))
    xyz, rgb, weights = _downsample(xyz, rgb, weights, max_points=max_points)

    projected = _project_points(xyz)
    half_extent = np.max(np.abs(projected), axis=0)
    span = float(np.max(half_extent))
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    padding = 0.08
    projected = projected / span
    projected = projected * (0.5 - padding) + 0.5
    projected = np.clip(projected, 0.0, 1.0)

    px = np.rint(projected[:, 0] * (image_size - 1)).astype(np.int32)
    py = np.rint((1.0 - projected[:, 1]) * (image_size - 1)).astype(np.int32)

    color_accum = np.zeros((image_size, image_size, 3), dtype=np.float32)
    weight_accum = np.zeros((image_size, image_size), dtype=np.float32)

    np.add.at(weight_accum, (py, px), weights)
    for channel in range(3):
        np.add.at(color_accum[..., channel], (py, px), rgb[:, channel] * weights)

    nonzero = weight_accum > 0
    avg_color = np.zeros_like(color_accum)
    avg_color[nonzero] = color_accum[nonzero] / weight_accum[nonzero, None]

    alpha = np.zeros_like(weight_accum)
    if np.any(nonzero):
        alpha_scale = float(np.percentile(weight_accum[nonzero], 99))
        if alpha_scale <= 0:
            alpha_scale = 1.0
        alpha = np.clip(weight_accum / alpha_scale, 0.0, 1.0) ** 0.7

    background = np.array([250.0, 250.0, 247.0], dtype=np.float32) / 255.0
    image = background[None, None, :] * (1.0 - alpha[..., None]) + avg_color * alpha[..., None]
    image = np.clip(image, 0.0, 1.0)

    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    thumb = cv2.GaussianBlur(np.uint8(image * 255.0), ksize=(0, 0), sigmaX=0.6)
    cv2.imwrite(str(thumbnail_path), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))


def collect_assets(
    input_dir: Path,
    output_dir: Path,
    limit: int | None,
    image_size: int,
    max_points: int,
    thumbnail_frame: int,
    confidence_threshold: float | None,
) -> int:
    candidates = sorted(input_dir.rglob("gaussians_sim.ply"))
    if not candidates:
        raise FileNotFoundError(f"No gaussians_sim.ply files found under {input_dir}")

    skipped_by_confidence: list[tuple[Path, str]] = []
    redirected_groups: dict[Path, str] = {}
    if confidence_threshold is not None:
        candidates, skipped_by_confidence, missing_report_count, redirected_groups = _filter_candidates_by_confidence(
            candidates,
            input_dir=input_dir,
            threshold=confidence_threshold,
        )
        print(
            f"Confidence filter: kept {len(candidates)} asset(s) with {_CONFIDENCE_METRIC} >= {confidence_threshold:.3f}"
        )
        if skipped_by_confidence:
            print(f"Confidence filter: skipped {len(skipped_by_confidence)} asset(s)")
        if missing_report_count:
            print(f"Confidence filter: {missing_report_count} asset(s) skipped because confidence.json was not found")
        if redirected_groups:
            print(
                f"Confidence filter: redirecting {len(redirected_groups)} asset(s) to their majority predicted class directories"
            )
        if not candidates:
            print("No assets remain after confidence filtering.")
            return 0

    if limit is not None:
        candidates = candidates[:limit]

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Assets selected:  {len(candidates)}")
    print("")

    output_dir.mkdir(parents=True, exist_ok=True)
    collected = 0
    for index, ply_path in enumerate(candidates, start=1):
        asset_dir = ply_path.parent.resolve()
        group_name = redirected_groups.get(asset_dir, _infer_group_name(ply_path, input_dir))
        output_stem = _build_output_stem(ply_path)
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        target_ply = group_dir / f"{output_stem}.ply"
        target_thumbnail = group_dir / f"{output_stem}.png"
        video_path = ply_path.with_name("3d_lifted.mp4")

        shutil.copy2(ply_path, target_ply)
        thumbnail_from_video = extract_video_thumbnail(
            video_path,
            target_thumbnail,
            image_size=image_size,
            frame_number=thumbnail_frame,
        )
        if not thumbnail_from_video:
            render_thumbnail(target_ply, target_thumbnail, image_size=image_size, max_points=max_points)
        collected += 1
        print(f"[{index}/{len(candidates)}] Collected {group_name}/{output_stem}")

    print("")
    print(f"Done. Collected {collected} assets into {output_dir}")
    if skipped_by_confidence:
        preview = skipped_by_confidence[:5]
        print("Confidence filter examples:")
        for ply_path, reason in preview:
            print(f"  SKIP {ply_path}: {reason}")
        if len(skipped_by_confidence) > len(preview):
            print(f"  ... {len(skipped_by_confidence) - len(preview)} more filtered asset(s)")
    if redirected_groups:
        preview = list(redirected_groups.items())[:5]
        print("Confidence redirect examples:")
        for asset_dir, target_class in preview:
            print(f"  MOVE {asset_dir} -> {target_class}/")
        if len(redirected_groups) > len(preview):
            print(f"  ... {len(redirected_groups) - len(preview)} more redirected asset(s)")
    return collected


def main() -> None:
    args = parse_args()
    _require_positive("--limit", args.limit)
    _require_positive("--thumbnail-size", args.thumbnail_size)
    _require_positive("--thumbnail-frame", args.thumbnail_frame)
    _require_positive("--max-points", args.max_points)
    _require_unit_interval("--confidence-threshold", args.confidence_threshold)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    collect_assets(
        input_dir=input_dir,
        output_dir=output_dir,
        limit=args.limit,
        image_size=args.thumbnail_size,
        max_points=args.max_points,
        thumbnail_frame=args.thumbnail_frame,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()