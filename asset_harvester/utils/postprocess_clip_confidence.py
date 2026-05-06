#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score multiview outputs against expected class labels with CLIP.

This script scans harvest outputs with the layout:

    <input-dir>/<clip_uuid>/<class_name>/<sample_id>/multiview/*.png

For each PNG under ``multiview/``, it computes a CLIP text-image match over a
candidate class vocabulary and counts how often the top-1 prediction matches the
sample's expected ``class_name``. A ``confidence.json`` report is written into
each clip directory.

Typical usage
-------------
Single clip:

    python -m asset_harvester.utils.postprocess_clip_confidence \
        --input-dir ./outputs/ncore_harvest/<clip_uuid>

Batch over a harvest root:

    python -m asset_harvester.utils.postprocess_clip_confidence \
        --input-dir ./outputs/ncore_harvest

`run.sh` automatically invokes this script for `--data-root` runs unless
`--skip-clip-postprocess` is passed.

Report contents
---------------
Each clip-level ``confidence.json`` contains:

- clip summary statistics
- sample-level ``mean_target_confidence``
- sample-level ``mean_predicted_confidence``
- ``majority_predicted_class`` for each sample
- ``prediction_histogram`` for each sample
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT_NAME = "confidence.json"
DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"
DEFAULT_CLASS_VOCAB = [
    "automobile",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "rider",
    "person",
    "trailer",
    "traffic_cone",
    "barrier",
]
LABEL_ALIASES = {
    "automobile": ["automobile", "car", "passenger car", "sedan"],
    "truck": ["truck", "pickup truck", "cargo truck"],
    "bus": ["bus", "coach bus"],
    "bicycle": ["bicycle", "bike"],
    "motorcycle": ["motorcycle", "motorbike"],
    "rider": ["rider", "cyclist", "motorcyclist"],
    "person": ["person", "pedestrian", "human"],
    "trailer": ["trailer", "cargo trailer"],
    "traffic_cone": ["traffic cone", "road cone", "orange cone"],
    "barrier": ["road barrier", "traffic barrier", "guard rail"],
}


@dataclass(frozen=True)
class SampleSpec:
    clip_dir: Path
    class_name: str
    sample_id: str
    multiview_dir: Path
    image_paths: tuple[Path, ...]

    @property
    def sample_dir(self) -> Path:
        return self.multiview_dir.parent

    @property
    def track_id(self) -> str:
        return f"{self.class_name}/{self.sample_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                description="Postprocess multiview PNG outputs with CLIP and write per-clip confidence reports.",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Examples:

    python -m asset_harvester.utils.postprocess_clip_confidence \
            --input-dir ./outputs/ncore_harvest/<clip_uuid>

    python -m asset_harvester.utils.postprocess_clip_confidence \
            --input-dir ./outputs/ncore_harvest \
            --class-names automobile,truck,bus,bicycle,motorcycle,rider,person

Output:
    Writes one confidence.json per clip directory.

Notes:
    - Expected layout: <clip_uuid>/<class_name>/<sample_id>/multiview/*.png
    - run.sh invokes this automatically for --data-root workflows.
""",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Harvest root or clip directory to scan for <clip>/<class>/<id>/multiview/*.png.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face CLIP model ID (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run CLIP on (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of multiview PNGs to score per forward pass (default: 16).",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optional comma-separated candidate class names. If omitted, use discovered classes plus defaults.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help=f"Prompt template used for each class alias (default: {DEFAULT_PROMPT_TEMPLATE!r}).",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum number of PNG files required to score a sample (default: 1).",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Report filename written inside each clip directory (default: {DEFAULT_OUTPUT_NAME}).",
    )
    return parser.parse_args()


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _display_label(label: str) -> str:
    return label.replace("_", " ").strip()


def _label_aliases(label: str) -> list[str]:
    normalized = _normalize_label(label)
    aliases = LABEL_ALIASES.get(normalized)
    if aliases is not None:
        return aliases
    pretty = _display_label(normalized)
    return [pretty]


def _parse_class_names(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    names = []
    seen = set()
    for item in raw_value.split(","):
        name = _normalize_label(item)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _discover_samples(input_dir: Path) -> tuple[dict[Path, list[SampleSpec]], list[dict[str, str]], set[str]]:
    clips: dict[Path, list[SampleSpec]] = defaultdict(list)
    skipped: list[dict[str, str]] = []
    discovered_classes: set[str] = set()

    for multiview_dir in sorted(path for path in input_dir.rglob("multiview") if path.is_dir()):
        image_paths = tuple(sorted(path for path in multiview_dir.glob("*.png") if path.is_file()))
        sample_dir = multiview_dir.parent
        class_dir = sample_dir.parent
        clip_dir = class_dir.parent

        if sample_dir == input_dir or class_dir == input_dir:
            skipped.append(
                {
                    "path": str(multiview_dir),
                    "reason": "Directory layout does not match <clip>/<class>/<sample>/multiview.",
                }
            )
            continue

        class_name = _normalize_label(class_dir.name)
        if not class_name:
            skipped.append({"path": str(sample_dir), "reason": "Empty class directory name."})
            continue

        discovered_classes.add(class_name)
        clips[clip_dir].append(
            SampleSpec(
                clip_dir=clip_dir,
                class_name=class_name,
                sample_id=sample_dir.name,
                multiview_dir=multiview_dir,
                image_paths=image_paths,
            )
        )

    for sample_specs in clips.values():
        sample_specs.sort(key=lambda spec: (spec.class_name, spec.sample_id))

    return dict(sorted(clips.items(), key=lambda item: str(item[0]))), skipped, discovered_classes


def _build_candidate_classes(discovered_classes: set[str], override: list[str]) -> list[str]:
    seen = set()
    result = []
    for label in override or sorted(discovered_classes):
        normalized = _normalize_label(label)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    if len(result) < 2:
        for label in DEFAULT_CLASS_VOCAB:
            normalized = _normalize_label(label)
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)

    return result


class ClipClassifier:
    def __init__(self, model_name: str, class_names: list[str], prompt_template: str, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "This script requires torch and transformers in the active environment. "
                "Install the repo with the multiview_diffusion or camera-estimator extras."
            ) from exc

        if not class_names:
            raise ValueError("At least one candidate class is required.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.class_names = class_names
        self.prompt_template = prompt_template
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._label_prompt_indices: list[list[int]] = []
        self._text_features = self._encode_texts()

    def _encode_texts(self) -> "torch.Tensor":
        prompt_texts = []
        label_prompt_indices: list[list[int]] = []

        for class_name in self.class_names:
            prompt_indices = []
            for alias in _label_aliases(class_name):
                prompt_indices.append(len(prompt_texts))
                prompt_texts.append(self.prompt_template.format(alias))
            label_prompt_indices.append(prompt_indices)

        self._label_prompt_indices = label_prompt_indices
        inputs = self.processor(text=prompt_texts, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def score_images(self, image_paths: list[Path], batch_size: int) -> list[dict[str, object]]:
        from PIL import Image

        results: list[dict[str, object]] = []
        scale = self.model.logit_scale.exp()

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                prompt_logits = scale * image_features @ self._text_features.T
                label_logits = self._aggregate_prompt_logits(prompt_logits)
                probabilities = label_logits.softmax(dim=-1)
                top_probabilities, top_indices = probabilities.max(dim=-1)

            for batch_index, path in enumerate(batch_paths):
                pred_index = int(top_indices[batch_index].item())
                predicted_class = self.class_names[pred_index]
                results.append(
                    {
                        "image": path.name,
                        "predicted_class": predicted_class,
                        "predicted_confidence": float(top_probabilities[batch_index].item()),
                        "label_probabilities": {
                            label: float(probabilities[batch_index, label_index].item())
                            for label_index, label in enumerate(self.class_names)
                        },
                    }
                )

            for image in images:
                image.close()

        return results

    def _aggregate_prompt_logits(self, prompt_logits: "torch.Tensor") -> "torch.Tensor":
        batch_size = prompt_logits.shape[0]
        label_logits = self.torch.full(
            (batch_size, len(self.class_names)),
            fill_value=-math.inf,
            device=prompt_logits.device,
            dtype=prompt_logits.dtype,
        )
        for label_index, prompt_indices in enumerate(self._label_prompt_indices):
            label_logits[:, label_index] = prompt_logits[:, prompt_indices].amax(dim=-1)
        return label_logits


def _score_sample(
    classifier: ClipClassifier,
    sample: SampleSpec,
    batch_size: int,
    min_images: int,
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    if len(sample.image_paths) < min_images:
        return None, {
            "track_id": sample.track_id,
            "sample_dir": str(sample.sample_dir),
            "reason": f"Only {len(sample.image_paths)} PNG files found; requires at least {min_images}.",
        }
    if sample.class_name not in classifier.class_names:
        return None, {
            "track_id": sample.track_id,
            "sample_dir": str(sample.sample_dir),
            "reason": f"Expected class '{sample.class_name}' is not present in the CLIP candidate vocabulary.",
        }

    image_results = classifier.score_images(list(sample.image_paths), batch_size=batch_size)
    matching_count = 0
    target_confidence_sum = 0.0
    predicted_confidence_sum = 0.0
    histogram: Counter[str] = Counter()

    for item in image_results:
        predicted_class = str(item["predicted_class"])
        histogram[predicted_class] += 1
        if predicted_class == sample.class_name:
            matching_count += 1
        label_probabilities = item["label_probabilities"]
        target_confidence_sum += float(label_probabilities[sample.class_name])
        predicted_confidence_sum += float(item["predicted_confidence"])

    image_count = len(image_results)
    majority_predicted_class = histogram.most_common(1)[0][0] if histogram else None
    return {
        "track_id": sample.track_id,
        "class_name": sample.class_name,
        "sample_id": sample.sample_id,
        "mean_target_confidence": target_confidence_sum / image_count if image_count else 0.0,
        "mean_predicted_confidence": predicted_confidence_sum / image_count if image_count else 0.0,
        "majority_predicted_class": majority_predicted_class,
        "prediction_histogram": dict(sorted(histogram.items())),
    }, None


def _summarize_clip(clip_id: str, clip_dir: Path, sample_reports: list[dict[str, object]], skipped_samples: list[dict[str, str]]) -> dict:
    sample_count = len(sample_reports)
    total_target_confidence = sum(float(item["mean_target_confidence"]) for item in sample_reports)
    class_stats: dict[str, dict[str, float | int]] = {}

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in sample_reports:
        grouped[str(sample["class_name"])].append(sample)

    for class_name, items in sorted(grouped.items()):
        class_sample_count = len(items)
        class_target_confidence = sum(float(item["mean_target_confidence"]) for item in items)
        class_stats[class_name] = {
            "sample_count": class_sample_count,
            "mean_target_confidence": class_target_confidence / class_sample_count if class_sample_count else 0.0,
        }

    return {
        "clip_id": clip_id,
        "clip_dir": str(clip_dir),
        "summary": {
            "sample_count": sample_count,
            "skipped_sample_count": len(skipped_samples),
            "mean_target_confidence": total_target_confidence / sample_count if sample_count else 0.0,
            "class_stats": class_stats,
        },
        "samples": sample_reports,
        "skipped_samples": skipped_samples,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.min_images <= 0:
        raise ValueError("--min-images must be > 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    clips, discovery_skips, discovered_classes = _discover_samples(input_dir)
    if not clips:
        raise RuntimeError(f"No multiview sample directories found under {input_dir}")

    override_classes = _parse_class_names(args.class_names)
    candidate_classes = _build_candidate_classes(discovered_classes, override_classes)
    classifier = ClipClassifier(
        model_name=args.model_name,
        class_names=candidate_classes,
        prompt_template=args.prompt_template,
        device=args.device,
    )

    print(f"Found {sum(len(samples) for samples in clips.values())} sample(s) across {len(clips)} clip(s)")
    print(f"Using CLIP model: {args.model_name}")
    print(f"Candidate classes: {', '.join(candidate_classes)}")

    discovery_skips_by_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in discovery_skips:
        path = Path(item["path"])
        candidate = path.parent.parent if path.name == "multiview" else path.parent
        discovery_skips_by_clip[str(candidate)].append(item)

    for clip_dir, samples in clips.items():
        clip_id = clip_dir.name
        print(f"\nScoring clip {clip_id} ({len(samples)} sample(s))")
        sample_reports = []
        skipped_samples = list(discovery_skips_by_clip.get(str(clip_dir), []))

        for sample in samples:
            report, skipped = _score_sample(
                classifier=classifier,
                sample=sample,
                batch_size=args.batch_size,
                min_images=args.min_images,
            )
            if skipped is not None:
                skipped_samples.append(skipped)
                print(f"  SKIP {sample.track_id}: {skipped['reason']}")
                continue
            assert report is not None
            sample_reports.append(report)
            print(
                "  "
                f"{sample.track_id}: mean_target_conf={report['mean_target_confidence']:.3f} "
                f"mean_pred_conf={report['mean_predicted_confidence']:.3f} "
                f"majority={report['majority_predicted_class']}"
            )

        report = _summarize_clip(clip_id=clip_id, clip_dir=clip_dir, sample_reports=sample_reports, skipped_samples=skipped_samples)
        report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["model_name"] = args.model_name
        report["prompt_template"] = args.prompt_template
        report["candidate_classes"] = candidate_classes
        report["input_dir"] = str(input_dir)

        output_path = clip_dir / args.output_name
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"  Wrote {output_path} "
            f"(overall mean_target_conf={report['summary']['mean_target_confidence']:.3f}, "
            f"samples={report['summary']['sample_count']})"
        )


if __name__ == "__main__":
    main()