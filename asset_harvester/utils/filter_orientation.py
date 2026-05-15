#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter multiview samples whose orientation is incorrect.

Two independent checks are applied (both must pass):

1. **LWH aspect-ratio check** (fast, no GPU):
   Compares the expected horizontal aspect ratio (L/W from ``multiview/lwh.txt``)
   with the actual extent ratio (``extent_z / extent_x``) of the ``gaussians.ply``
   bounding box (2nd–98th percentile).  Detects yaw-rotation errors (45°, 90°,
   etc.) because they distort the horizontal footprint of the reconstruction.

   In the normalized PLY frame (before ``rescale_gaussians``):
     - z-axis = object length (forward/backward)
     - x-axis = object width  (left/right)
     - y-axis = object height (up/down)

   A correctly oriented reconstruction satisfies::

       |actual_ratio - expected_ratio| / expected_ratio  <=  threshold  (default 30%)

2. **CLIP pairwise check** (requires GPU, optional via ``--skip-clip``):
   Uses CLIP to compare rear-group views (0, 1, 15 — near azimuth 0°) with
   front-group views (7, 8, 9 — near azimuth 180°) via relative scoring.
   Detects 180° orientation flips where front/rear are swapped.

   Pass criteria:
     - avg_rear_score(rear_group) > avg_rear_score(front_group)
     - avg_front_score(front_group) > avg_front_score(rear_group)

   The relative (pairwise) approach avoids absolute-classification bias
   inherent in CLIP's text matching.

Typical usage
-------------

LWH-only check (fast, no GPU):

    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest/<clip_uuid> \\
        --skip-clip

Full check (LWH + CLIP):

    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest/<clip_uuid>

Delete bad samples (prompts for confirmation):

    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest \\
        --delete --skip-clip

Adjust LWH threshold (stricter):

    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest \\
        --skip-clip --lwh-threshold 0.2

Save a JSON report:

    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest \\
        --skip-clip --output-json ./outputs/orientation_report.json

Integration with collect_sim_assets.py
--------------------------------------

The ``_compute_lwh_check()`` function is used by ``collect_sim_assets.py``
(via ``--lwh-threshold``) to filter samples during asset collection without
requiring a separate pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14-336"

# Views near azimuth 0° (rear) and azimuth 180° (front).
_DEFAULT_REAR_VIEWS = [0, 1, 15]   # azimuth 0°, 22.5°, 337.5°
_DEFAULT_FRONT_VIEWS = [7, 8, 9]   # azimuth 157.5°, 180°, 202.5°

# LWH aspect-ratio check: max allowed relative deviation between expected
# and actual horizontal aspect ratio (L/W from lwh.txt vs extent_z/extent_x
# from gaussians.ply).  Default 0.3 = 30%.
_DEFAULT_LWH_THRESHOLD = 0.3

# Per-orientation CLIP prompt aliases (3-class: rear / front / side).
_VEHICLE_CLASSES = {"automobile", "truck", "heavy_truck", "bus", "trailer", "van"}
_CYCLIST_CLASSES = {"bicycle", "motorcycle", "rider"}
_PERSON_CLASSES = {"person", "pedestrian"}

_PROMPTS: dict[str, dict[str, list[str]]] = {
    "vehicle": {
        "rear": [
            "rear of a car",
            "back of a car",
            "car from behind",
            "vehicle taillights",
            "car tail lights",
            "back bumper of a car",
        ],
        "front": [
            "front of a car",
            "front of a vehicle",
            "car headlights",
            "vehicle front bumper",
            "car grille",
        ],
        "side": [
            "side of a car",
            "car profile view",
            "vehicle from the side",
            "car door",
        ],
    },
    "cyclist": {
        "rear": [
            "cyclist from behind",
            "bicycle from behind",
            "motorcycle rear view",
        ],
        "front": [
            "cyclist from the front",
            "bicycle front view",
            "motorcycle front view",
        ],
        "side": [
            "cyclist from the side",
            "bicycle side view",
            "motorcycle profile",
        ],
    },
    "person": {
        "rear": [
            "person from behind",
            "pedestrian rear view",
            "back of a person",
        ],
        "front": [
            "person facing the camera",
            "pedestrian front view",
            "person face",
        ],
        "side": [
            "person from the side",
            "pedestrian profile",
            "side of a person",
        ],
    },
}

ORIENTATIONS = ("rear", "front", "side")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt_category(class_name: str) -> str:
    """Map a class name to one of the prompt categories."""
    normalized = class_name.strip().lower()
    if normalized in _VEHICLE_CLASSES:
        return "vehicle"
    if normalized in _CYCLIST_CLASSES:
        return "cyclist"
    if normalized in _PERSON_CLASSES:
        return "person"
    # Default to vehicle prompts for unknown classes
    return "vehicle"


# ---------------------------------------------------------------------------
# Orientation scorer
# ---------------------------------------------------------------------------


class OrientationScorer:
    """CLIP-based orientation scorer (rear / front / side).

    Returns per-view softmax scores over the three orientation categories.
    Uses the same loading pattern as ``ClipClassifier`` in
    ``postprocess_clip_confidence.py``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "This script requires torch and transformers. "
                "Install the repo with the multiview_diffusion or camera-estimator extras."
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Pre-encode all prompts for each category x orientation.
        self._text_features: dict[str, dict[str, "torch.Tensor"]] = {}
        for category, orientation_prompts in _PROMPTS.items():
            self._text_features[category] = {}
            for orientation, prompts in orientation_prompts.items():
                inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    feats = self.model.get_text_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                self._text_features[category][orientation] = feats

    def score_view(self, image_path: Path, class_name: str) -> dict[str, float]:
        """Return softmax probabilities ``{"rear": p, "front": p, "side": p}``."""
        from PIL import Image

        category = _prompt_category(class_name)
        text_feats = self._text_features[category]

        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        img.close()

        scale = self.model.logit_scale.exp()
        with self._torch.inference_mode():
            img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            logits = []
            for orientation in ORIENTATIONS:
                feats = text_feats[orientation]
                alias_logits = scale * (img_feat @ feats.T)
                logits.append(alias_logits.amax(dim=-1))

            logit_tensor = self._torch.stack(logits, dim=-1)
            probs = logit_tensor.softmax(dim=-1).squeeze(0)

        return {orientation: float(probs[i].item()) for i, orientation in enumerate(ORIENTATIONS)}


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleEntry:
    sample_dir: Path
    class_name: str
    clip_dir: Path

    @property
    def track_id(self) -> str:
        return f"{self.class_name}/{self.sample_dir.name}"


def _find_samples(input_dir: Path) -> list[SampleEntry]:
    """Walk *input_dir* and return every sample that has a multiview/ subdirectory."""
    entries: list[SampleEntry] = []
    seen: set[Path] = set()

    for multiview_dir in sorted(input_dir.rglob("multiview")):
        if not multiview_dir.is_dir():
            continue
        sample_dir = multiview_dir.parent
        if sample_dir in seen:
            continue
        seen.add(sample_dir)

        # Expected layout: <clip_dir>/<class_name>/<sample_id>/multiview/
        class_dir = sample_dir.parent
        clip_dir = class_dir.parent

        if sample_dir == input_dir or class_dir == input_dir:
            continue
        class_name = class_dir.name.strip().lower()
        if not class_name:
            continue

        entries.append(SampleEntry(sample_dir=sample_dir, class_name=class_name, clip_dir=clip_dir))

    return entries


# ---------------------------------------------------------------------------
# LWH aspect-ratio check (detects yaw rotation: 45°, 90°, etc.)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class LwhCheckResult:
    """Result of comparing lwh.txt aspect ratio with gaussians.ply AABB."""
    available: bool = False   # whether both files exist
    expected_ratio: float = 0.0   # L/W from lwh.txt
    actual_ratio: float = 0.0     # ext_z/ext_x from gaussians.ply
    deviation: float = 0.0        # |actual - expected| / expected
    lwh: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ply_extents: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z)
    passed: bool = True  # default pass if data unavailable
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "passed": self.passed,
            "expected_ratio_LW": round(self.expected_ratio, 4),
            "actual_ratio_zx": round(self.actual_ratio, 4),
            "deviation": round(self.deviation, 4),
            "lwh": [round(v, 4) for v in self.lwh],
            "ply_extents_xyz": [round(v, 4) for v in self.ply_extents],
            "reason": self.reason,
        }


def _compute_lwh_check(sample_dir: Path, threshold: float) -> LwhCheckResult:
    """Compare aspect ratio from lwh.txt with gaussians.ply AABB.

    In the normalized PLY frame (before rescale_gaussians):
      - z-axis corresponds to object length (forward/backward)
      - x-axis corresponds to object width (left/right)
      - y-axis corresponds to object height (up/down)

    A correctly oriented reconstruction should have:
      extent_z / extent_x  ≈  L / W   (from lwh.txt)

    Rotations around the vertical axis change this ratio:
      - 90° rotation: ratio ≈ W/L (reciprocal)
      - 45° rotation: ratio → 1.0
    """
    from asset_harvester.tokengs.ply_io import read_ply

    result = LwhCheckResult()

    lwh_path = sample_dir / "multiview" / "lwh.txt"
    ply_path = sample_dir / "gaussians.ply"

    if not lwh_path.is_file():
        result.reason = "lwh.txt not found"
        return result
    if not ply_path.is_file():
        result.reason = "gaussians.ply not found"
        return result

    # Read lwh.txt
    try:
        lwh = np.loadtxt(str(lwh_path), dtype=np.float64)
        if lwh.size != 3:
            result.reason = f"lwh.txt has {lwh.size} values, expected 3"
            return result
        length, width, height = float(lwh[0]), float(lwh[1]), float(lwh[2])
    except Exception as e:
        result.reason = f"Failed to read lwh.txt: {e}"
        return result

    if width <= 0 or length <= 0:
        result.reason = f"Invalid lwh values: L={length}, W={width}"
        return result

    result.lwh = (length, width, height)

    # Read gaussians.ply and compute robust AABB (2nd-98th percentile)
    try:
        props = read_ply(str(ply_path))
        xyz = np.stack([props["x"], props["y"], props["z"]], axis=1).astype(np.float64)
    except Exception as e:
        result.reason = f"Failed to read gaussians.ply: {e}"
        return result

    if xyz.shape[0] < 10:
        result.reason = f"Too few points in PLY: {xyz.shape[0]}"
        return result

    # Use percentile-based extents for robustness against outliers
    lo = np.percentile(xyz, 2, axis=0)
    hi = np.percentile(xyz, 98, axis=0)
    extents = hi - lo  # [ext_x, ext_y, ext_z]

    ext_x, ext_y, ext_z = float(extents[0]), float(extents[1]), float(extents[2])
    result.ply_extents = (ext_x, ext_y, ext_z)

    if ext_x <= 0:
        result.reason = f"Zero x-extent in PLY: {ext_x}"
        return result

    result.available = True
    result.expected_ratio = length / width
    result.actual_ratio = ext_z / ext_x if ext_x > 0 else 0.0

    # Relative deviation
    if result.expected_ratio > 0:
        result.deviation = abs(result.actual_ratio - result.expected_ratio) / result.expected_ratio
    else:
        result.deviation = 0.0

    result.passed = result.deviation <= threshold
    if not result.passed:
        result.reason = f"aspect deviation {result.deviation:.1%} > threshold {threshold:.0%}"

    return result


# ---------------------------------------------------------------------------
# Per-sample orientation check (pairwise relative comparison + LWH)
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    sample_dir: Path
    class_name: str
    track_id: str
    # CLIP scores per group
    rear_group_rear_score: float = 0.0
    rear_group_front_score: float = 0.0
    front_group_rear_score: float = 0.0
    front_group_front_score: float = 0.0
    # Per-view raw scores for reporting
    rear_view_scores: dict[int, dict[str, float]] = field(default_factory=dict)
    front_view_scores: dict[int, dict[str, float]] = field(default_factory=dict)
    # CLIP check results
    rule1_pass: bool = False  # rear group more "rear" than front group
    rule2_pass: bool = False  # front group more "front" than rear group
    # LWH check result
    lwh_check: LwhCheckResult = field(default_factory=LwhCheckResult)

    @property
    def clip_passed(self) -> bool:
        return self.rule1_pass and self.rule2_pass

    @property
    def passed(self) -> bool:
        return self.clip_passed and self.lwh_check.passed

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "sample_dir": str(self.sample_dir),
            "class_name": self.class_name,
            "passed": self.passed,
            "clip_passed": self.clip_passed,
            "rule1_rear_group_more_rear": self.rule1_pass,
            "rule2_front_group_more_front": self.rule2_pass,
            "lwh_check": self.lwh_check.to_dict(),
            "rear_group_avg_rear_score": round(self.rear_group_rear_score, 4),
            "rear_group_avg_front_score": round(self.rear_group_front_score, 4),
            "front_group_avg_rear_score": round(self.front_group_rear_score, 4),
            "front_group_avg_front_score": round(self.front_group_front_score, 4),
            "rear_view_scores": {str(k): v for k, v in sorted(self.rear_view_scores.items())},
            "front_view_scores": {str(k): v for k, v in sorted(self.front_view_scores.items())},
        }


def _check_sample(
    entry: SampleEntry,
    scorer: OrientationScorer | None,
    rear_views: list[int],
    front_views: list[int],
    lwh_threshold: float,
) -> SampleResult:
    """Score a sample using CLIP relative comparison + LWH aspect ratio check."""
    result = SampleResult(
        sample_dir=entry.sample_dir,
        class_name=entry.class_name,
        track_id=entry.track_id,
    )

    # Person orientation is not used in downstream filtering. Skip checks.
    if entry.class_name in _PERSON_CLASSES:
        result.rule1_pass = True
        result.rule2_pass = True
        result.lwh_check.passed = True
        result.lwh_check.reason = "person class skipped"
        return result

    # --- LWH aspect ratio check (fast, no GPU needed) ---
    result.lwh_check = _compute_lwh_check(entry.sample_dir, threshold=lwh_threshold)

    # --- CLIP pairwise check ---
    if scorer is None:
        # Skip CLIP check (--skip-clip mode), mark as pass
        result.rule1_pass = True
        result.rule2_pass = True
        return result

    # Score rear group views
    rear_rear_scores: list[float] = []
    rear_front_scores: list[float] = []
    for view_idx in rear_views:
        image_path = entry.sample_dir / "multiview" / f"{view_idx}.png"
        if not image_path.is_file():
            continue
        scores = scorer.score_view(image_path, entry.class_name)
        result.rear_view_scores[view_idx] = scores
        rear_rear_scores.append(scores["rear"])
        rear_front_scores.append(scores["front"])

    # Score front group views
    front_rear_scores: list[float] = []
    front_front_scores: list[float] = []
    for view_idx in front_views:
        image_path = entry.sample_dir / "multiview" / f"{view_idx}.png"
        if not image_path.is_file():
            continue
        scores = scorer.score_view(image_path, entry.class_name)
        result.front_view_scores[view_idx] = scores
        front_rear_scores.append(scores["rear"])
        front_front_scores.append(scores["front"])

    # Need at least one view in each group
    if not rear_rear_scores or not front_front_scores:
        return result

    # Average scores per group
    result.rear_group_rear_score = sum(rear_rear_scores) / len(rear_rear_scores)
    result.rear_group_front_score = sum(rear_front_scores) / len(rear_front_scores)
    result.front_group_rear_score = sum(front_rear_scores) / len(front_rear_scores)
    result.front_group_front_score = sum(front_front_scores) / len(front_front_scores)

    # Pairwise rules:
    # Rule 1: rear group views should score higher on "rear" than front group views
    result.rule1_pass = result.rear_group_rear_score > result.front_group_rear_score
    # Rule 2: front group views should score higher on "front" than rear group views
    result.rule2_pass = result.front_group_front_score > result.rear_group_front_score

    return result


# ---------------------------------------------------------------------------
# Main filtering logic
# ---------------------------------------------------------------------------


def filter_orientation(
    input_dir: Path,
    dry_run: bool,
    delete: bool,
    yes: bool,
    output_json: Path | None,
    model_name: str,
    device: str | None,
    rear_views: list[int],
    front_views: list[int],
    lwh_threshold: float,
    skip_clip: bool,
    class_filter: list[str] | None,
) -> int:
    """Scan *input_dir*, check orientation via CLIP + LWH, optionally delete bad ones.

    Returns the number of samples that failed the orientation check.
    """
    entries = _find_samples(input_dir)
    if not entries:
        print(f"No multiview samples found under {input_dir}")
        return 0

    if class_filter:
        normalized_filter = {c.strip().lower() for c in class_filter}
        entries = [e for e in entries if e.class_name in normalized_filter]
        if not entries:
            print("No samples match the --class-filter, nothing to do.")
            return 0

    print(f"Found {len(entries)} sample(s) under {input_dir}")
    print(f"LWH aspect-ratio threshold: {lwh_threshold:.0%}")
    if skip_clip:
        print("CLIP check: DISABLED (--skip-clip)")
        scorer = None
    else:
        print(f"Rear  group views: {rear_views}")
        print(f"Front group views: {front_views}")
        print(f"Loading CLIP model {model_name!r}...")
        scorer = OrientationScorer(model_name=model_name, device=device)
        print("Model loaded.")
    print()

    passed: list[SampleResult] = []
    failed: list[SampleResult] = []

    for i, entry in enumerate(entries, start=1):
        result = _check_sample(
            entry, scorer,
            rear_views=rear_views, front_views=front_views,
            lwh_threshold=lwh_threshold,
        )
        status = "PASS" if result.passed else "FAIL"

        # Build detail string
        parts = []
        # LWH info
        lc = result.lwh_check
        if lc.available:
            lwh_flag = "ok" if lc.passed else "X"
            parts.append(f"LWH:{lwh_flag}(exp={lc.expected_ratio:.2f},act={lc.actual_ratio:.2f},dev={lc.deviation:.0%})")
        else:
            parts.append(f"LWH:skip({lc.reason})")
        # CLIP info
        if not skip_clip:
            r1 = "ok" if result.rule1_pass else "X"
            r2 = "ok" if result.rule2_pass else "X"
            parts.append(f"CLIP[R1:{r1} R2:{r2}]")
        detail = "  ".join(parts)
        print(f"[{i}/{len(entries)}] {status}  {entry.track_id}  {detail}")
        if result.passed:
            passed.append(result)
        else:
            failed.append(result)

    print(f"\nResults: {len(passed)} passed, {len(failed)} failed")

    if output_json is not None:
        report = {
            "input_dir": str(input_dir),
            "lwh_threshold": lwh_threshold,
            "skip_clip": skip_clip,
            "rear_views": rear_views,
            "front_views": front_views,
            "total": len(entries),
            "passed": len(passed),
            "failed": len(failed),
            "failed_samples": [r.to_dict() for r in failed],
            "passed_samples": [r.to_dict() for r in passed],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {output_json}")

    if not failed:
        return 0

    if not delete:
        if dry_run:
            print(
                "\nDry-run mode: no files deleted. "
                "Re-run with --delete to remove the failed samples."
            )
        return len(failed)

    # Confirm deletion
    if not yes:
        print(f"\nAbout to permanently delete {len(failed)} sample director(ies):")
        for r in failed[:10]:
            print(f"  {r.sample_dir}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        answer = input("\nProceed with deletion? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Deletion cancelled.")
            return len(failed)

    deleted = 0
    for r in failed:
        if r.sample_dir.is_dir():
            shutil.rmtree(r.sample_dir)
            print(f"  DELETED {r.sample_dir}")
            deleted += 1
        else:
            print(f"  SKIP (already gone) {r.sample_dir}")

    print(f"\nDeleted {deleted} sample director(ies).")
    return len(failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter multiview samples with incorrect orientation using CLIP.\n\n"
            "Uses relative comparison between two view groups:\n"
            "  Rear group  (views 0, 1, 15 — near azimuth 0°)\n"
            "  Front group (views 7, 8, 9  — near azimuth 180°)\n\n"
            "A sample passes when:\n"
            "  1. Rear group scores higher on 'rear' than front group\n"
            "  2. Front group scores higher on 'front' than rear group"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Dry-run on a single clip:
    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest/<clip_uuid>

  Dry-run over entire harvest root:
    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest

  Delete bad samples (with confirmation):
    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest --delete

  Delete without confirmation and write a report:
    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest \\
        --delete --yes \\
        --output-json ./outputs/orientation_report.json

  Check only automobile samples:
    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest \\
        --class-filter automobile,heavy_truck

  Custom view groups:
    python -m asset_harvester.utils.filter_orientation \\
        --input-dir ./outputs/ncore_harvest \\
        --rear-views 0,1,15 --front-views 7,8,9
""",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Harvest root or clip directory to scan for <clip>/<class>/<id>/multiview/.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face CLIP model ID (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--rear-views",
        default=",".join(str(v) for v in _DEFAULT_REAR_VIEWS),
        help=(
            "Comma-separated view indices for the rear group "
            f"(default: {','.join(str(v) for v in _DEFAULT_REAR_VIEWS)})."
        ),
    )
    parser.add_argument(
        "--front-views",
        default=",".join(str(v) for v in _DEFAULT_FRONT_VIEWS),
        help=(
            "Comma-separated view indices for the front group "
            f"(default: {','.join(str(v) for v in _DEFAULT_FRONT_VIEWS)})."
        ),
    )
    parser.add_argument(
        "--lwh-threshold",
        type=float,
        default=_DEFAULT_LWH_THRESHOLD,
        help=(
            "Max allowed relative deviation between expected (L/W from lwh.txt) "
            "and actual (extent_z/extent_x from gaussians.ply) aspect ratios. "
            f"Default: {_DEFAULT_LWH_THRESHOLD} ({_DEFAULT_LWH_THRESHOLD:.0%})."
        ),
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        default=False,
        help="Disable the CLIP-based check (only run LWH aspect-ratio check).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print results without deleting anything (default behaviour).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Delete sample directories that fail the orientation check.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Skip the deletion confirmation prompt (only effective with --delete).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write a JSON report of pass/fail results.",
    )
    parser.add_argument(
        "--class-filter",
        default=None,
        help="Optional comma-separated list of class names to check (default: all classes).",
    )
    return parser.parse_args()


def _parse_view_list(raw: str, name: str) -> list[int]:
    """Parse a comma-separated list of non-negative integers."""
    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            idx = int(item)
        except ValueError:
            raise ValueError(f"Invalid view index in {name}: {item!r}")
        if idx < 0:
            raise ValueError(f"View indices must be non-negative in {name}, got {idx}")
        result.append(idx)
    if not result:
        raise ValueError(f"{name} must contain at least one view index.")
    return result


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rear_views = _parse_view_list(args.rear_views, "--rear-views")
    front_views = _parse_view_list(args.front_views, "--front-views")

    overlap = set(rear_views) & set(front_views)
    if overlap:
        raise ValueError(f"--rear-views and --front-views must not overlap. Common indices: {sorted(overlap)}")

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None

    class_filter = None
    if args.class_filter:
        class_filter = [c.strip() for c in args.class_filter.split(",") if c.strip()]

    n_failed = filter_orientation(
        input_dir=input_dir,
        dry_run=args.dry_run,
        delete=args.delete,
        yes=args.yes,
        output_json=output_json,
        model_name=args.model_name,
        device=args.device,
        rear_views=rear_views,
        front_views=front_views,
        lwh_threshold=args.lwh_threshold,
        skip_clip=args.skip_clip,
        class_filter=class_filter,
    )

    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
