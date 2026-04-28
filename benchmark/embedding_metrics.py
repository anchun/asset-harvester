#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Patch embedding metrics for eval: keypoint detector, segmenter, DINOv3 embedder,
and compute_embedding_metrics_patch.

Used by eval.py to compute per-frame DINOv3 patch-feature distances (whole-
foreground and per-body-part for persons). Exposes SAM3DBodyKeypointDetector,
KeypointBodyPartSegmenter, load_dinov3_backbone, DINOv3PatchEmbedder,
compute_embedding_metrics_patch, and helpers (BODY_PART_NAMES).
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Path to this module's directory (benchmark/); used for sam-3d-body and DINOv3 checkpoints.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add sam-3d-body to Python path so the keypoint detector can import sam_3d_body.
_SAM3D_DIR = os.path.join(_SCRIPT_DIR, "sam-3d-body")
if os.path.isdir(_SAM3D_DIR) and _SAM3D_DIR not in sys.path:
    sys.path.insert(0, _SAM3D_DIR)


# =============================================================================
# Part 1: Keypoint detector (SAM 3D Body)
# =============================================================================
# SAM 3D Body outputs 70 MHR keypoints; we map a subset to COCO-17 for
# body-part segment definitions and compatibility with skeleton-based segmenter.

# Index mapping: MHR-70 keypoint indices -> COCO-17 order.
# COCO-17: 0 nose, 1 l-eye, 2 r-eye, 3 l-ear, 4 r-ear, 5 l-shoulder, 6 r-shoulder,
#           7 l-elbow, 8 r-elbow, 9 l-wrist, 10 r-wrist, 11 l-hip, 12 r-hip,
#           13 l-knee, 14 r-knee, 15 l-ankle, 16 r-ankle.
MHR70_TO_COCO17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


class SAM3DBodyKeypointDetector:
    """2D keypoint detector using SAM 3D Body mesh recovery.

    Runs the full SAM 3D Body pipeline on each image and extracts
    ``pred_keypoints_2d``, mapping from MHR-70 to COCO-17 format.

    Args:
        checkpoint_path: Path to the SAM 3D Body ``.ckpt`` checkpoint.
        mhr_path: Path to the MHR model asset (``.pt`` file).
        device: CUDA device string.
        output_format: ``"coco17"`` (default) maps to 17 COCO keypoints;
            ``"mhr70"`` returns all 70 MHR keypoints.
    """

    def __init__(
        self,
        checkpoint_path: str,
        mhr_path: str = "",
        device: str = "cuda:0",
        output_format: str = "coco17",
    ):
        self.device = device
        self.output_format = output_format
        self._estimator = None
        self._checkpoint_path = checkpoint_path
        self._mhr_path = mhr_path

    def _get_estimator(self):
        if self._estimator is not None:
            return self._estimator
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        device = torch.device(self.device)
        model, model_cfg = load_sam_3d_body(
            self._checkpoint_path,
            device=device,
            mhr_path=self._mhr_path,
        )
        self._estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
        )
        return self._estimator

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect 2D keypoints on a single RGB image.

        Args:
            image: [H, W, 3] uint8 RGB.

        Returns:
            keypoints: [K, 3] float32 — (x, y, confidence). K=17 for COCO-17, K=70 for MHR-70.
        """
        estimator = self._get_estimator()
        h, w = image.shape[:2]
        outputs = estimator.process_one_image(
            image,
            bboxes=np.array([[0, 0, w, h]], dtype=np.float32),
            bbox_thr=0.0,
            inference_type="body",
        )
        if not outputs:
            n_kps = 17 if self.output_format == "coco17" else 70
            return np.zeros((n_kps, 3), dtype=np.float32)
        kps_2d = outputs[0]["pred_keypoints_2d"]
        if self.output_format == "coco17":
            coco_kps = kps_2d[MHR70_TO_COCO17]
            conf = np.ones((17, 1), dtype=np.float32)
            return np.concatenate([coco_kps, conf], axis=-1).astype(np.float32)
        conf = np.ones((kps_2d.shape[0], 1), dtype=np.float32)
        return np.concatenate([kps_2d, conf], axis=-1).astype(np.float32)


# =============================================================================
# Part 2: Body-part and foreground segmentation
# =============================================================================
# KeypointBodyPartSegmenter assigns each foreground patch to the nearest body part
# (by distance to skeleton segments). COCO17_BODY_PART_SEGMENTS defines which
# keypoint pairs form the skeleton for each part.

BODY_PART_NAMES: list[str] = [
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]

COCO17_BODY_PART_SEGMENTS: dict[str, list[tuple[int, int]]] = {
    "head": [(0, 1), (0, 2), (1, 3), (2, 4)],
    "torso": [(5, 6), (5, 11), (6, 12), (11, 12)],
    "left_arm": [(5, 7), (7, 9)],
    "right_arm": [(6, 8), (8, 10)],
    "left_leg": [(11, 13), (13, 15)],
    "right_leg": [(12, 14), (14, 16)],
}


def _point_to_segment_distances(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> np.ndarray:
    """Min distance from each point to the line segment seg_a–seg_b.

    Used by KeypointBodyPartSegmenter to compute per-patch distances to each
    body part's skeleton segments; the part with smallest distance wins.
    """
    d = seg_b - seg_a
    len_sq = float(np.dot(d, d))
    if len_sq < 1e-10:
        return np.linalg.norm(points - seg_a, axis=-1)
    t = np.clip(np.dot(points - seg_a, d) / len_sq, 0.0, 1.0)
    proj = seg_a + t[:, np.newaxis] * d
    return np.linalg.norm(points - proj, axis=-1)


class KeypointBodyPartSegmenter:
    """Body-part segmentation via keypoint skeleton proximity.

    Builds a grid of patch centres; for each patch whose centre lies in the
    foreground, computes the minimum distance to each body part's skeleton
    (line segments between keypoint pairs). Assigns the patch to the nearest
    part, subject to max_patch_dist. Outputs binary masks per part (head,
    torso, left_arm, right_arm, left_leg, right_leg).
    """

    def __init__(
        self,
        keypoint_detector,
        patch_size: int = 16,
        confidence_thr: float = 0.3,
        max_patch_dist: int = 2,
    ):
        self.keypoint_detector = keypoint_detector
        self.patch_size = patch_size
        self.confidence_thr = confidence_thr
        self.max_patch_dist = max_patch_dist
        self.last_part_segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
        self.last_keypoints: np.ndarray | None = None

    def _select_segment_table(self, n_kps: int) -> dict[str, list[tuple[int, int]]]:
        return COCO17_BODY_PART_SEGMENTS

    def segment_body_parts(
        self,
        image: np.ndarray,
        fg_mask: np.ndarray,
        keypoints: np.ndarray | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Segment the image into body-part regions by assigning patches to the nearest skeleton segment.

        Algorithm overview
        ------------------
        The image is divided into a grid of patches (patch_size x patch_size pixels).
        Each patch is represented by its centre. Foreground patches (those whose
        centre lies inside the foreground mask) are assigned to one of six body
        parts: head, torso, left_arm, right_arm, left_leg, right_leg. Assignment
        is by minimum distance to that part's "skeleton": a set of line segments
        between keypoint pairs (see COCO17_BODY_PART_SEGMENTS). For each patch we
        compute the minimum distance to any segment of each part; the part with
        the smallest distance wins. A patch is only assigned if that distance is
        at most max_patch_dist * patch_size pixels; otherwise it is discarded.
        Output is one binary mask per part (uint8, 255 = belonging to that part).

        Parameters
        ----------
        image : np.ndarray
            Input image, shape (H, W) or (H, W, C). Used for keypoint detection
            when keypoints is None, and for image dimensions.
        fg_mask : np.ndarray
            Foreground mask, shape (H, W). Required. Binary: use bool, or uint8
            with 255 = foreground (only the binary FG vs BG is used). Only
            patches whose centre falls in the foreground are considered.
        keypoints : np.ndarray | None, optional
            Keypoints in COCO-17 order, shape (17, 3) with (x, y, confidence).
            When None, keypoints are detected from image via self.keypoint_detector.
            When provided (e.g. from a full-resolution image then transformed),
            detection is skipped and these keypoints define the skeletons.

        Returns
        -------
        dict[str, np.ndarray] | None
            Dict mapping part name (e.g. "head", "torso") to a binary mask
            (H, W), uint8, 255 where that part is assigned. Returns None if
            fewer than four parts have any assigned patches (i.e. more than two
            body parts "missing"), to avoid unreliable per-part metrics.

        Side effects
        ------------
        self.last_keypoints : set to the keypoints used (detected or passed in).
        self.last_part_segments : set to the skeleton line segments (in image
            coordinates) used per part, for debugging/visualization.

        Skeleton definition
        -------------------
        Each body part is defined by a list of keypoint index pairs (COCO17).
        Only segments whose both endpoints have confidence >= confidence_thr are
        used. Parts with no valid segments are skipped for that image.
        """
        h, w = image.shape[:2]
        assert fg_mask.shape[:2] == (h, w), "fg_mask must match image shape (H, W)"

        # Resolve keypoints: use provided or detect from image; store for later use.
        kps = keypoints if keypoints is not None else self.keypoint_detector.detect(image)
        self.last_keypoints = kps
        seg_table = self._select_segment_table(kps.shape[0])

        # Build patch grid: patch_size x patch_size, centres at (row+0.5)*ps, (col+0.5)*ps.
        ps = self.patch_size
        rows, cols = h // ps, w // ps
        if rows == 0 or cols == 0:
            return {}
        n_patches = rows * cols
        row_centers = (np.arange(rows) + 0.5) * ps
        col_centers = (np.arange(cols) + 0.5) * ps
        gx, gy = np.meshgrid(col_centers, row_centers)
        points = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (n_patches, 2) xy

        # Mark which patches have their centre in the foreground (from fg_mask; bool or 0/255).
        fg_bool = np.asarray(fg_mask, dtype=bool)
        centre_rows = np.clip((gy.ravel()).astype(int), 0, h - 1)
        centre_cols = np.clip((gx.ravel()).astype(int), 0, w - 1)
        fg_patches = fg_bool[centre_rows, centre_cols]

        # For each body part, keep only segments with both endpoints above confidence_thr.
        # Compute per-patch min distance to any segment of that part; store in part_dists.
        part_names: list[str] = []
        part_dists: list[np.ndarray] = []
        self.last_part_segments = {}
        for part_name, segments in seg_table.items():
            valid_segs: list[tuple[np.ndarray, np.ndarray]] = []
            for a, b in segments:
                if (
                    a < kps.shape[0]
                    and b < kps.shape[0]
                    and kps[a, 2] >= self.confidence_thr
                    and kps[b, 2] >= self.confidence_thr
                ):
                    valid_segs.append((kps[a, :2].astype(np.float64), kps[b, :2].astype(np.float64)))
            if not valid_segs:
                continue
            self.last_part_segments[part_name] = [
                ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))) for p1, p2 in valid_segs
            ]
            # Per-patch distance to this part = min over all segments of this part.
            min_dists = np.full(n_patches, np.inf)
            for p1, p2 in valid_segs:
                dists = _point_to_segment_distances(points, p1, p2)
                np.minimum(min_dists, dists, out=min_dists)
            part_names.append(part_name)
            part_dists.append(min_dists)

        if not part_names:
            return {}

        # Assign each patch to the nearest part; discard if distance > max_patch_dist * patch_size px.
        all_dists = np.stack(part_dists, axis=0)
        nearest_part = np.argmin(all_dists, axis=0)
        nearest_dist = np.min(all_dists, axis=0)
        max_px_dist = self.max_patch_dist * ps
        within_range = nearest_dist <= max_px_dist

        # Build one binary mask per part: only patches that are FG, within range, and nearest to that part.
        result: dict[str, np.ndarray] = {}
        for part_idx, part_name in enumerate(part_names):
            assigned = np.where((nearest_part == part_idx) & fg_patches & within_range)[0]
            if len(assigned) == 0:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            for idx in assigned:
                r, c = divmod(int(idx), cols)
                y0, x0 = r * ps, c * ps
                mask[y0 : min(y0 + ps, h), x0 : min(x0 + ps, w)] = 255
            result[part_name] = mask

        # Require at least 4 body parts (no more than 2 missing); else return None.
        if len(result) < 4:
            return None
        return result


# =============================================================================
# Part 3: DINOv3 backbone and patch embedder
# =============================================================================
# DINOv3 (HuggingFace, pretrained) is used for patch features; distinct from
# the DINOv3 inside SAM 3D Body. Dinov3Backbone wraps the model; DINOv3PatchEmbedder
# produces spatial feature maps and supports mask_pool for averaging over regions.

_DINOV3_DEFAULT_PATH = os.environ.get(
    "DINOV3_CKPT",
    os.path.join(_SCRIPT_DIR, "checkpoints", "dinov3-vith16plus-pretrain-lvd1689m"),
)
_DINOV3_HF_MODEL = "facebook/dinov3-vith16plus-pretrain-lvd1689m"


def load_dinov3_backbone(
    model_path: str | None = None,
    device: str = "cuda:0",
):
    """Load DINOv3 backbone from HuggingFace (local or remote).

    Precedence: model_path -> DINOV3_CKPT env -> default local dir -> HF download.
    """
    from transformers import AutoModel

    path = model_path or _DINOV3_DEFAULT_PATH
    if os.path.isdir(path):
        print(f"Loading DINOv3 pretrained backbone from {path}")
        model = AutoModel.from_pretrained(path)
    else:
        print(f"Local DINOv3 not found at {path}, downloading from HuggingFace...")
        model = AutoModel.from_pretrained(_DINOV3_HF_MODEL)
    model = model.to(device).eval()
    return Dinov3Backbone(model, device=device)


class Dinov3Backbone:
    """Wraps HuggingFace DINOv3 to provide get_intermediate_layers interface.

    Used by DINOv3PatchEmbedder to get patch tokens (and optionally CLS token)
    in the shape expected by the rest of the pipeline.
    """

    def __init__(self, model, device: str = "cuda:0"):
        self.model = model
        self.device = device
        config = model.config
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        self.num_register_tokens = getattr(config, "num_register_tokens", 0)
        self._prefix_len = 1 + self.num_register_tokens

    def parameters(self):
        return self.model.parameters()

    def get_intermediate_layers(
        self,
        x,
        n=1,
        reshape=False,
        return_class_token=False,
        return_extra_tokens=False,
        norm=True,
    ):
        with torch.no_grad():
            outputs = self.model(pixel_values=x, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0:1, :]
        patch_tokens = last_hidden[:, self._prefix_len :, :]
        if reshape:
            B, N, C = patch_tokens.shape
            Hp = Wp = int(math.sqrt(N))
            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, Hp, Wp)
        if return_class_token:
            return [(patch_tokens, cls_token)]
        return [patch_tokens]


_IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])


def _preprocess(image: np.ndarray, input_size: int, backbone, device: str) -> torch.Tensor:
    """Resize image to input_size, normalize with ImageNet stats, return [1,3,H,W] tensor."""
    pil = Image.fromarray(image).resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - _IMAGE_MEAN[:, None, None]) / _IMAGE_STD[:, None, None]
    dtype = next(backbone.parameters()).dtype
    return t.unsqueeze(0).to(device=device, dtype=dtype)


class DINOv3PatchEmbedder:
    """DINOv3 patch features with mask-based pooling for patch embedding metrics.

    Wraps a DINOv3 backbone (e.g. Dinov3Backbone) to produce spatial patch
    feature maps from an image and to pool those features over arbitrary
    binary masks. Used in compute_embedding_metrics_patch for whole-foreground
    and per-body-part cosine distances between GT and rendered crops.
    """

    def __init__(self, backbone, device: str = "cuda:0", input_size: int = 256):
        """Store backbone and config; input_size is the resize dimension for the image.

        Parameters
        ----------
        backbone
            Backbone with get_intermediate_layers(..., reshape=True), and
            attributes patch_size, embed_dim. Typically a Dinov3Backbone.
        device : str, optional
            Device to run the backbone and store tensors on (e.g. "cuda:0").
        input_size : int, optional
            Height and width to which the input image is resized before
            forwarding through the backbone (default 256).
        """
        self.backbone = backbone
        self.device = device
        self.input_size = input_size
        self.patch_size = backbone.patch_size
        self.embed_dim = backbone.embed_dim

    def embed_patches(self, image: np.ndarray) -> torch.Tensor:
        """Extract DINOv3 patch feature map for one image.

        The image is resized to (input_size, input_size), normalized, and
        passed through the backbone's intermediate layers. No gradients.

        Parameters
        ----------
        image : np.ndarray
            Input image, shape (H, W, 3) or (H, W), uint8 or float. Will be
            resized to (input_size, input_size) and normalized for the backbone.

        Returns
        -------
        torch.Tensor
            Patch features, shape (1, C, Hp, Wp), where Hp, Wp are the patch
            grid size and C is embed_dim. Dtype and device match the backbone.
        """
        img = _preprocess(image, self.input_size, self.backbone, self.device)
        with torch.no_grad():
            out = self.backbone.get_intermediate_layers(
                img,
                n=1,
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )
            features = out[0]
        return features

    def mask_pool(self, features: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """Average patch features over the foreground region of a binary mask.

        The mask is resized to the patch grid (Hp, Wp) with NEAREST. Any
        non-zero value is treated as foreground. Patch tokens inside the
        foreground are averaged to produce a single [C] vector. If the mask
        has no foreground after resize, returns a zero vector of shape (C,).

        Parameters
        ----------
        features : torch.Tensor
            Patch feature map from embed_patches, shape (1, C, Hp, Wp).
        mask : np.ndarray
            Binary mask, shape (H, W). Any non-zero is foreground (bool,
            uint8 0/255, or 0/1). Need not match feature resolution; will be
            resized to (Wp, Hp) with NEAREST.

        Returns
        -------
        torch.Tensor
            Pooled feature vector, shape (C,), on the same device and dtype
            as features.
        """
        _, C, Hp, Wp = features.shape
        # Resize mask to patch grid; treat any non-zero as foreground (accepts 0/255, 0/1, or bool).
        mask_resized = cv2.resize(np.asarray(mask, dtype=np.uint8), (Wp, Hp), interpolation=cv2.INTER_NEAREST)
        mask_bool = torch.from_numpy(mask_resized > 0).to(features.device, dtype=torch.bool)
        feat_flat = features[0].permute(1, 2, 0)
        masked = feat_flat[mask_bool]
        if masked.shape[0] == 0:
            return torch.zeros(C, device=features.device, dtype=features.dtype)
        return masked.mean(dim=0)


# =============================================================================
# Part 4: Patch embedding metrics
# =============================================================================
# Helpers for visualization (_draw_keypoints, _side_by_side, _draw_body_part_masks)
# and for cropping (_crop_to_foreground). The main entry is compute_embedding_metrics_patch.

_KP_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 128, 0),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (0, 128, 255),
    (128, 0, 255),
    (200, 100, 50),
    (50, 100, 200),
    (200, 50, 100),
    (100, 200, 50),
    (50, 200, 100),
    (100, 50, 200),
    (220, 220, 0),
    (0, 220, 220),
    (220, 0, 220),
    (180, 60, 60),
    (60, 180, 60),
    (60, 60, 180),
]


def _draw_keypoints(
    image: np.ndarray,
    kps: np.ndarray,
    valid_mask: np.ndarray,
    radius: int = 4,
) -> np.ndarray:
    """Draw keypoints on a copy of image; only indices where valid_mask is True are drawn."""
    vis = image.copy()
    for k in range(kps.shape[0]):
        if not valid_mask[k]:
            continue
        x, y = int(round(kps[k, 0])), int(round(kps[k, 1]))
        color = _KP_COLORS[k % len(_KP_COLORS)]
        cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return vis


def _side_by_side(*images: np.ndarray) -> np.ndarray:
    """Concatenate images horizontally, matching height with white separators between them."""
    target_h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            s = target_h / h
            img = np.array(Image.fromarray(img).resize((int(w * s), target_h), Image.BILINEAR))
        resized.append(img)
    sep = np.full((target_h, 2, 3), 255, dtype=np.uint8)
    parts = []
    for j, img in enumerate(resized):
        if j > 0:
            parts.append(sep)
        parts.append(img)
    return np.concatenate(parts, axis=1)


_BODY_PART_COLORS = {
    "head": (255, 100, 100),
    "torso": (255, 255, 100),
    "left_arm": (100, 255, 100),
    "right_arm": (100, 100, 255),
    "left_leg": (255, 100, 255),
    "right_leg": (100, 255, 255),
    "object": (100, 200, 255),
}


def _draw_body_part_masks(
    image: np.ndarray,
    part_masks: dict[str, np.ndarray],
    patch_grid: tuple[int, int] | None = None,
    part_segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] | None = None,
) -> np.ndarray:
    """Overlay part masks with distinct colors; optionally draw patch grid and skeleton segments."""
    alpha = 0.6
    vis = image.copy().astype(np.float32)
    for part_name, mask in part_masks.items():
        color = _BODY_PART_COLORS.get(part_name, (128, 128, 128))
        mask_f = (np.asarray(mask) > 0).astype(np.float32)[:, :, np.newaxis]
        vis = vis * (1 - alpha * mask_f) + np.array(color, dtype=np.float32) * alpha * mask_f
    vis = vis.clip(0, 255).astype(np.uint8)
    if patch_grid is not None:
        h, w = vis.shape[:2]
        rows, cols = patch_grid
        grid_color = (200, 200, 200)
        for r in range(1, rows):
            y = int(round(r * h / rows))
            cv2.line(vis, (0, y), (w - 1, y), grid_color, 1, cv2.LINE_AA)
        for c in range(1, cols):
            x = int(round(c * w / cols))
            cv2.line(vis, (x, 0), (x, h - 1), grid_color, 1, cv2.LINE_AA)
    if part_segments is not None:
        for part_name, segs in part_segments.items():
            color = _BODY_PART_COLORS.get(part_name, (128, 128, 128))
            for (x1, y1), (x2, y2) in segs:
                p1 = (int(round(x1)), int(round(y1)))
                p2 = (int(round(x2)), int(round(y2)))
                cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)
                cv2.circle(vis, p1, 2, color, -1, cv2.LINE_AA)
                cv2.circle(vis, p2, 2, color, -1, cv2.LINE_AA)
    return vis


def _crop_to_foreground(
    image: np.ndarray,
    pad_ratio: float = 0.05,
    fg_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to a square bounding box around foreground, with padding. Returns (crop, (y0,x0,y1,x1))."""
    h, w = image.shape[:2]
    if fg_mask is None:
        fg_mask = np.any(image < 250, axis=-1)
    else:
        fg_mask = fg_mask > 127 if fg_mask.dtype == np.uint8 else fg_mask.astype(bool)
    if not fg_mask.any():
        return image, (0, 0, h, w)
    rows = np.where(fg_mask.any(axis=1))[0]
    cols = np.where(fg_mask.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    bh, bw = y1 - y0, x1 - x0
    pad = int(max(bh, bw) * pad_ratio)
    y0, y1 = max(y0 - pad, 0), min(y1 + pad, h)
    x0, x1 = max(x0 - pad, 0), min(x1 + pad, w)
    bh, bw = y1 - y0, x1 - x0
    if bh > bw:
        diff = bh - bw
        x0 = max(x0 - diff // 2, 0)
        x1 = min(x0 + bh, w)
        x0 = max(x1 - bh, 0)
    elif bw > bh:
        diff = bw - bh
        y0 = max(y0 - diff // 2, 0)
        y1 = min(y0 + bw, h)
        y0 = max(y1 - bw, 0)
    return image[y0:y1, x0:x1].copy(), (y0, x0, y1, x1)


def compute_embedding_metrics_patch(
    rendered_images: torch.Tensor,
    rendered_alphas: torch.Tensor,
    gt_frames: list,
    masks: list,
    patch_embedder: DINOv3PatchEmbedder,
    body_part_segmenter=None,
    is_person: bool = True,
    output_dir: str | None = None,
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Compute per-frame DINOv3 patch embedding distances (no warping).

    For each frame pair (rendered, GT), the pipeline is:

    **Step 0 – Convert and validate**
        Convert rendered tensor to uint8 numpy; build rendered foreground mask
        from alpha. Load GT frame and GT mask for this frame. If no GT mask is
        provided for this frame, skip embedding metrics for this frame (continue).
        Otherwise assert GT mask and GT frame have the same resolution.

    **Step 1 – Prepare GT crop**
        Erode the GT mask slightly, fill GT background outside the eroded mask
        with ``bg_color`` (must match the render background used for
        ``rendered_images``), then crop to a tight square bounding box around
        the foreground (_crop_to_foreground with 5% padding). Result:
        gt_crop_raw, gt_bbox.

    **Step 2 – Prepare rendered crop**
        If the rendered foreground is too close to the image edge, pad image
        and alpha mask so _crop_to_foreground can apply full padding. Crop
        rendered to foreground bbox. Result: rendered_crop_raw, rend_bbox.

    **Step 3 – Resize to DINOv3 resolution**
        Resize both crops to patch_embedder.input_size (e.g. 512x512) for
        consistent patch grid. Yields gt_crop and rendered_crop.

    **Step 4 – Extract patch features**
        Run patch_embedder.embed_patches on both crops to get feat_gt and
        feat_rendered ([1, C, Hp, Wp]).

    **Step 5 – Whole-foreground distance (mean_emb_dist)**
        Build foreground masks at DINOv3 resolution: crop GT mask to gt_bbox
        and resize to dinov3_size; crop rendered alpha to rend_bbox and resize.
        mask_pool features over each foreground; compute 1 - cosine_similarity
        between the two pooled vectors. Append to frame_distances.

    **Step 6 – Per-body-part distances (mean_emb_dist_part, persons only)**
        Detect keypoints on the *uncropped* GT frame (better resolution), then
        map keypoints into the cropped+resized GT coordinate system. Run
        body_part_segmenter on gt_crop (with those keypoints) and on
        rendered_crop (keypoints detected on the crop). For each body part
        present in both, mask_pool features and compute cosine distance; store
        per-part and take the mean for this frame.

    **Step 7 – Debug output (optional)**
        If output_dir is set, save patch_input_XX.png (body-part or foreground
        overlays) and keypoints_XX.jpeg (keypoint overlay for persons).

    **Aggregation**
        Return dict with emb_dist, mean_emb_dist; for persons also
        emb_dist_part, mean_emb_dist_part, and per_part_<name>_emb_dist,
        mean_per_part_<name>_emb_dist.

    Args:
        rendered_images: [N, 3, H, W] float 0–1.
        rendered_alphas: [N, 1, H, W] float 0–1 (rendered foreground).
        gt_frames: List of N [H, W, 3] uint8 GT images.
        masks: List of N [H, W] uint8 (255=foreground); same resolution as gt_frames.
            Frames with no mask (None or index out of range) are skipped.
        patch_embedder: DINOv3PatchEmbedder instance.
        body_part_segmenter: For persons, KeypointBodyPartSegmenter (or None).
        is_person: If True, compute per-body-part metrics when segmenter is set.
        output_dir: If set, save debug overlays (patch_input_XX.png, keypoints_XX.jpeg).
        bg_color: RGB in [0, 1] for GT background fill and render padding. Must
            match the background used to render ``rendered_images`` (default white).

    Returns:
        Dict with emb_dist, mean_emb_dist; for persons also emb_dist_part,
        mean_emb_dist_part, per_part_<name>_emb_dist, mean_per_part_<name>_emb_dist.
    """
    num_frames = len(gt_frames)
    frame_distances: list[float] = []
    part_frame_distances: list[float] = []
    per_part_distances: dict[str, list[float]] = {name: [] for name in BODY_PART_NAMES} if is_person else {}
    bg_u8 = np.array(
        [int(round(c * 255.0)) for c in bg_color],
        dtype=np.uint8,
    )

    for i in range(num_frames):
        # -------------------------------------------------------------------------
        # Step 0: Convert and validate inputs for this frame
        # -------------------------------------------------------------------------
        # Convert rendered image from [3,H,W] float 0-1 to [H,W,3] uint8 for
        # cropping and OpenCV. Build rendered foreground mask from alpha channel
        # (same shape as rendered_np; 255 where alpha > 0.5). Load GT frame and
        # GT mask for frame i. If no mask is provided, skip this frame entirely.
        # Otherwise require mask and GT frame to have the same height/width.
        rendered_np = rendered_images[i].permute(1, 2, 0).cpu().numpy()
        rendered_np = (rendered_np * 255).clip(0, 255).astype(np.uint8)
        gt_frame = gt_frames[i]
        rend_alpha_i = rendered_alphas[i]
        if rend_alpha_i.dim() == 3:
            rend_alpha_i = rend_alpha_i[0]
        rend_fg = (rend_alpha_i.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        fg_mask = masks[i] if i < len(masks) else None
        if fg_mask is None:
            continue  # GT mask required; skip embedding metrics for this frame
        assert fg_mask.shape[:2] == gt_frame.shape[:2], (
            f"GT mask shape {fg_mask.shape[:2]} must match GT frame {gt_frame.shape[:2]}"
        )

        # -------------------------------------------------------------------------
        # Step 1: Prepare GT crop (erode mask, fill background, crop to bbox)
        # -------------------------------------------------------------------------
        # Binarize GT mask (255 where > 127). Erode once with 11x11 ellipse to
        # shrink the boundary slightly and avoid including mask edges in the
        # foreground. Paint GT background (outside eroded mask) with bg_u8 so
        # the crop matches the render background. _crop_to_foreground returns a
        # square crop with ~5% padding around the foreground bbox (gt_bbox in
        # full-frame coords: y0, x0, y1, x1).
        fg_binary = (fg_mask > 127).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        gt_fg = cv2.erode(fg_binary, kernel, iterations=1)
        gt_masked = gt_frame.copy()
        gt_masked[gt_fg == 0] = bg_u8
        gt_crop_raw, gt_bbox = _crop_to_foreground(gt_masked, fg_mask=gt_fg)

        # -------------------------------------------------------------------------
        # Step 2: Prepare rendered crop (pad if FG near edge, then crop to bbox)
        # -------------------------------------------------------------------------
        # _crop_to_foreground adds ~5% padding around the FG bbox. If the
        # rendered foreground is too close to the image edge, that padding would
        # extend outside the image. So we compute the required padding (need);
        # if any side has less than `need` pixels to the edge, we pad the
        # image with ``bg_u8`` and alpha with 0 so the crop can be taken with
        # full padding. Then crop to foreground bbox; rend_bbox is
        # (y0, x0, y1, x1) in the possibly-padded image.
        rh, rw = rendered_np.shape[:2]
        fg_r = rend_fg > 127
        if fg_r.any():
            rrows = np.where(fg_r.any(axis=1))[0]
            rcols = np.where(fg_r.any(axis=0))[0]
            ry0, ry1 = int(rrows[0]), int(rrows[-1]) + 1
            rx0, rx1 = int(rcols[0]), int(rcols[-1]) + 1
            need = int(max(ry1 - ry0, rx1 - rx0) * 0.05) + 1
            if ry0 < need or rx0 < need or (rh - ry1) < need or (rw - rx1) < need:
                border_rgb = (int(bg_u8[0]), int(bg_u8[1]), int(bg_u8[2]))
                rendered_np = cv2.copyMakeBorder(
                    rendered_np,
                    need,
                    need,
                    need,
                    need,
                    cv2.BORDER_CONSTANT,
                    value=border_rgb,
                )
                rend_fg = cv2.copyMakeBorder(
                    rend_fg,
                    need,
                    need,
                    need,
                    need,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        rendered_crop_raw, rend_bbox = _crop_to_foreground(rendered_np, fg_mask=rend_fg)

        # -------------------------------------------------------------------------
        # Step 3: Resize both crops to DINOv3 input size
        # -------------------------------------------------------------------------
        # The backbone expects a fixed input size (e.g. 512). Resize GT and
        # rendered crops to (dinov3_size, dinov3_size) with bilinear
        # interpolation so patch grid and feature dimensions are consistent.
        dinov3_size = patch_embedder.input_size
        gt_crop = np.array(Image.fromarray(gt_crop_raw).resize((dinov3_size, dinov3_size), Image.BILINEAR))
        rendered_crop = np.array(Image.fromarray(rendered_crop_raw).resize((dinov3_size, dinov3_size), Image.BILINEAR))

        # -------------------------------------------------------------------------
        # Step 4: Extract DINOv3 patch feature maps
        # -------------------------------------------------------------------------
        # Run the patch embedder on both crops. Output shape [1, C, Hp, Wp] where
        # Hp, Wp are the patch grid dimensions. No gradients needed for metrics.
        with torch.no_grad():
            feat_gt = patch_embedder.embed_patches(gt_crop)
            feat_rendered = patch_embedder.embed_patches(rendered_crop)

        # -------------------------------------------------------------------------
        # Step 5: Whole-foreground embedding and cosine distance (mean_emb_dist)
        # -------------------------------------------------------------------------
        # Crop the GT foreground mask to gt_bbox and resize to dinov3_size with
        # NEAREST to keep binary. Same for rendered: crop alpha to rend_bbox,
        # resize. mask_pool averages patch features over the foreground region
        # to get one vector per image. Distance = 1 - cosine_similarity (0 =
        # identical, 2 = opposite). If either mask is empty after crop/resize,
        # use 0.0 for this frame.
        gt_y0, gt_x0, gt_y1, gt_x1 = gt_bbox
        gt_fg_crop = gt_fg[gt_y0:gt_y1, gt_x0:gt_x1]
        gt_fg_crop = np.array(Image.fromarray(gt_fg_crop).resize((dinov3_size, dinov3_size), Image.NEAREST))
        gt_fg_mask = gt_fg_crop > 127  # binary (bool); only FG vs BG is used
        fg_masks_gt = {"object": gt_fg_mask} if gt_fg_mask.any() else {}
        ry0, rx0, ry1, rx1 = rend_bbox
        rend_fg_crop = rend_fg[ry0:ry1, rx0:rx1]
        rend_fg_crop = np.array(Image.fromarray(rend_fg_crop).resize((dinov3_size, dinov3_size), Image.NEAREST))
        rend_fg_mask = rend_fg_crop > 127  # binary (bool); only FG vs BG is used
        fg_masks_rendered = {"object": rend_fg_mask} if rend_fg_mask.any() else {}

        if fg_masks_gt and fg_masks_rendered:
            emb_gt_fg = patch_embedder.mask_pool(feat_gt, fg_masks_gt["object"])
            emb_rend_fg = patch_embedder.mask_pool(feat_rendered, fg_masks_rendered["object"])
            fg_dist = (1.0 - F.cosine_similarity(emb_gt_fg.unsqueeze(0), emb_rend_fg.unsqueeze(0))).item()
        else:
            fg_dist = 0.0
        frame_distances.append(fg_dist)

        # -------------------------------------------------------------------------
        # Step 6: Per-body-part segmentation and distances (persons only;
        #         mean_emb_dist_part and per_part_<name>_emb_dist)
        # -------------------------------------------------------------------------
        # Only when is_person, body_part_segmenter is set, and keypoint_detector
        # is available. Skip per-part metrics if keypoint_detector is None or if
        # segment_body_parts returns None (e.g. more than two body parts missing).
        # Keypoints are detected on full-res GT, then transformed to crop+resize
        # coords. Segment body parts on gt_crop and rendered_crop, passing
        # gt_fg_mask and rend_fg_mask so foreground is from the mask. For each
        # part present in both, mask_pool and compute cosine distance.
        masks_gt = {}
        masks_rendered = {}
        gt_skel = None
        rend_skel = None
        gt_kps = None
        rend_kps = None
        if is_person and body_part_segmenter is not None:
            kp_detector = getattr(body_part_segmenter, "keypoint_detector", None)
            if kp_detector is None:
                pass  # skip per-part metrics when keypoint detector is unavailable
            else:
                gt_kps_orig = kp_detector.detect(gt_frame)
                gt_y0, gt_x0 = gt_bbox[0], gt_bbox[1]
                crop_h, crop_w = gt_crop_raw.shape[:2]
                gt_kps_crop = gt_kps_orig.copy()
                gt_kps_crop[:, 0] = (gt_kps_crop[:, 0] - gt_x0) * dinov3_size / crop_w
                gt_kps_crop[:, 1] = (gt_kps_crop[:, 1] - gt_y0) * dinov3_size / crop_h
                masks_gt = body_part_segmenter.segment_body_parts(
                    gt_crop,
                    fg_mask=gt_fg_mask,
                    keypoints=gt_kps_crop,
                )
                masks_rendered = body_part_segmenter.segment_body_parts(
                    rendered_crop,
                    fg_mask=rend_fg_mask,
                )
                if masks_gt is None or masks_rendered is None:
                    pass  # skip per-part metrics when segmenter returned None
                else:
                    gt_skel = getattr(body_part_segmenter, "last_part_segments", None)
                    gt_kps = getattr(body_part_segmenter, "last_keypoints", None)
                    rend_skel = getattr(body_part_segmenter, "last_part_segments", None)
                    rend_kps = getattr(body_part_segmenter, "last_keypoints", None)
                    if not masks_gt:
                        print(f"    [warn] Frame {i}: segmenter returned no masks for GT")
                    if not masks_rendered:
                        print(f"    [warn] Frame {i}: segmenter returned no masks for rendered")
                    common_parts = set(masks_gt.keys()) & set(masks_rendered.keys())
                    part_dists = []
                    for part_name in common_parts:
                        emb_gt = patch_embedder.mask_pool(feat_gt, masks_gt[part_name])
                        emb_rendered = patch_embedder.mask_pool(feat_rendered, masks_rendered[part_name])
                        dist = (1.0 - F.cosine_similarity(emb_gt.unsqueeze(0), emb_rendered.unsqueeze(0))).item()
                        part_dists.append(dist)
                        per_part_distances[part_name].append(dist)
                    part_frame_dist = float(np.mean(part_dists)) if part_dists else 0.0
                    part_frame_distances.append(part_frame_dist)

        # -------------------------------------------------------------------------
        # Step 7: Debug visualizations (optional)
        # -------------------------------------------------------------------------
        # If output_dir is set: save patch_input_XX.png (GT | rendered with
        # body-part or whole-FG overlay) and, for persons with keypoints,
        # keypoints_XX.jpeg (keypoint overlay side-by-side).
        if output_dir is not None:
            _, _, Hp_gt, Wp_gt = feat_gt.shape
            _, _, Hp_r, Wp_r = feat_rendered.shape
            if is_person and body_part_segmenter is not None and masks_gt:
                gt_overlay = _draw_body_part_masks(
                    gt_crop,
                    masks_gt,
                    (Hp_gt, Wp_gt),
                    part_segments=gt_skel,
                )
                rend_overlay = _draw_body_part_masks(
                    rendered_crop,
                    masks_rendered,
                    (Hp_r, Wp_r),
                    part_segments=rend_skel,
                )
            else:
                gt_overlay = _draw_body_part_masks(gt_crop, fg_masks_gt, (Hp_gt, Wp_gt))
                rend_overlay = _draw_body_part_masks(
                    rendered_crop,
                    fg_masks_rendered,
                    (Hp_r, Wp_r),
                )
            overlay_path = os.path.join(output_dir, f"patch_input_{i:02d}.png")
            Image.fromarray(_side_by_side(gt_overlay, rend_overlay)).save(overlay_path)
            if gt_kps is not None and rend_kps is not None:
                conf_thr = getattr(body_part_segmenter, "confidence_thr", 0.3)
                gt_valid = gt_kps[:, 2] >= conf_thr
                rend_valid = rend_kps[:, 2] >= conf_thr
                gt_kp_vis = _draw_keypoints(gt_crop, gt_kps, gt_valid)
                rend_kp_vis = _draw_keypoints(rendered_crop, rend_kps, rend_valid)
                kp_path = os.path.join(output_dir, f"keypoints_{i:02d}.jpeg")
                Image.fromarray(_side_by_side(gt_kp_vis, rend_kp_vis)).save(kp_path, quality=95)

    # -------------------------------------------------------------------------
    # Aggregation: build result dict
    # -------------------------------------------------------------------------
    # emb_dist: list of per-frame whole-foreground distances; mean_emb_dist: mean.
    # For persons with part segmenter: emb_dist_part, mean_emb_dist_part, and
    # per_part_<name>_emb_dist / mean_per_part_<name>_emb_dist for each part.
    result = {
        "emb_dist": frame_distances,
        "mean_emb_dist": float(np.mean(frame_distances)) if frame_distances else 0.0,
    }
    if is_person and part_frame_distances:
        result["emb_dist_part"] = part_frame_distances
        result["mean_emb_dist_part"] = float(np.mean(part_frame_distances))
        for part_name, dists in per_part_distances.items():
            if dists:
                result[f"per_part_{part_name}_emb_dist"] = dists
                result[f"mean_per_part_{part_name}_emb_dist"] = float(np.mean(dists))
    return result
