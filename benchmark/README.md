# AH Benchmark

Evaluate `run_inference.py` output by rendering Gaussian splat assets from reserved-view cameras and comparing results to ground-truth frames.

**Pipeline:** For each sample (`output_dir/<class_name>/<sample_id>/`), the script loads `gaussians.ply` and the path in `reserved_views.json`, renders at the reserved views, aligns renders to GT masks, then computes **PSNR**, **LPIPS**, **SSIM**, and optional **DINOv3 patch embedding** distances (whole-foreground; for persons, also per-body-part).

---

## Metrics

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak signal-to-noise ratio (rendered vs GT). |
| **LPIPS** | Learned perceptual image patch similarity. |
| **SSIM** | Structural similarity. |
| **mean_emb_dist** | DINOv3 patch-feature cosine distance over the **whole foreground** (1 − cosine similarity). Computed only for frames that have a GT mask. |
| **mean_emb_dist_part** | For **pedestrians only**: per-body-part (head, torso, arms, legs) cosine distances, then averaged. Requires keypoint detector and body-part segmenter; skipped if keypoint detector is missing or segmenter returns too few parts. |

Embedding metrics require the **SAM 3D Body** checkpoint and **DINOv3** backbone (see `install.sh` under `./benchmark`). If they are missing, only PSNR/LPIPS/SSIM are reported.

---

## Inputs

`eval.py` expects the **run_inference output directory** with this layout:

```
output_dir/
├── <class_name>/              # e.g. VRU_pedestrians,consumer_vehicles
│   └── <sample_id>/           # e.g. 0d7b602f2da8c364
│       ├── gaussians.ply
│       └── reserved_views.json
└── ...
```

`reserved_views.json` must point to a folder that contains:

```
<reserved_views_path>/
├── camera.json                # frame_filenames, mask_filenames, normalized_cam_positions, cam_dists, cam_fovs, object_lwh
├── frame_00.jpeg              # GT frames (names from camera.json)
├── frame_01.jpeg
├── mask_00.png                # Foreground masks (same resolution as frames; required for embedding metrics)
├── mask_01.png
└── ...
```

- GT frames and masks must have the **same resolution**.
- Rendering uses `--output_size` (default 512). GT masks (and frames) **must** match this size — the alignment step asserts `mask.shape == (output_size, output_size)`.

---

## Outputs

Results are written to `--eval_output_dir` (default `<output_dir>/eval`), mirroring the input tree:

```
eval_output_dir/
├── <class_name>/
│   └── <sample_id>/
│       ├── render_comparison_00.jpeg   # GT | rendered (unless --no_comparisons)
│       ├── render_comparison_01.jpeg
│       ├── patch_input_00.png         # Body-part / foreground overlay (if --vis_debug)
│       ├── keypoints_00.jpeg         # Keypoint overlay, persons only (if --vis_debug)
│       └── ...
└── rendering_metrics_summary.txt      # Per-sample and overall PSNR/LPIPS/SSIM/embedding
```

---

## Installation


### Setup

Install the benchmark dependencies after the main environment has been set up:

```bash
conda create --name av-object-benchmark --clone asset-harvester

conda activate av-object-benchmark 
bash benchmark/install.sh
```


This will:
1. Install benchmark-specific packages (`pytorch-lightning`, `yacs`, `ninja`, `termcolor`, `transformers>=4.56.0`).
2. Download DINOv3 pretrained weights to `benchmark/checkpoints/dinov3-vith16plus-pretrain-lvd1689m/`.
3. Clone `sam-3d-body/` and download the SAM 3D Body checkpoint to `benchmark/checkpoints/sam-3d-body-dinov3/`.

### Optional: custom checkpoint paths

```bash
export SAM3D_CKPT=/path/to/model.ckpt
export SAM3D_MHR_PATH=/path/to/mhr_model.pt
export DINOV3_CKPT=/path/to/dinov3-vith16plus-pretrain-lvd1689m
```

If the SAM 3D Body checkpoint is not found, eval still runs but skips embedding metrics.

---

## Usage

```bash
conda activate av-object-benchmark 
```

**Basic run (all samples, default 512 resolution):**

```bash
python eval.py --output_dir /path/to/run_inference_output --eval_output_dir ./eval
```

**Save debug images (keypoints + patch overlays):**

```bash
python eval.py --output_dir /path/to/run_inference_output --eval_output_dir ./eval --vis_debug
```

**Other options:**

```bash
# Single class only
python eval.py --output_dir /path --eval_output_dir ./eval --class_name VRU_pedestrians

# Limit number of samples (with optional seed)
python eval.py --output_dir /path --eval_output_dir ./eval --max_samples 10 --sample_seed 42

# Only samples whose path contains a substring
python eval.py --output_dir /path --eval_output_dir ./eval --sample 0d7b602f

# No comparison images
python eval.py --output_dir /path --eval_output_dir ./eval --no_comparisons

# Render at 1024
python eval.py --output_dir /path --eval_output_dir ./eval --output_size 1024
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | *(required)* | Root directory from `run_inference.py` (`class_name/sample_id/gaussians.ply`, `reserved_views.json`). |
| `--eval_output_dir` | `<output_dir>/eval` | Where to write metrics and images. |
| `--class_name` | all | Restrict to samples under this class folder. |
| `--max_samples` | all | Cap number of samples; use with `--sample_seed` for subsampling. |
| `--sample_seed` | `0` | Random seed when `--max_samples` is set. |
| `--sample` | all | Only run on sample paths containing this substring. |
| `--output_size` | `512` | Render resolution (and typical GT/mask size). |
| `--no_comparisons` | false | Do not write `render_comparison_XX.jpeg`. |
| `--vis_debug` | false | Write `patch_input_XX.png` and `keypoints_XX.jpeg`. |

---

## Scripts and code layout

| File | Role |
|------|------|
| **eval.py** | Main entry. Loads samples from run_inference output, renders via `utils`, aligns to GT masks, computes PSNR/LPIPS/SSIM (`utils.compute_metrics`) and patch embedding metrics (`embedding_metrics.compute_embedding_metrics_patch`). |
| **embedding_metrics.py** | Keypoint detector (`SAM3DBodyKeypointDetector`), body-part segmenter (`KeypointBodyPartSegmenter`), DINOv3 backbone and patch embedder (`Dinov3Backbone`, `DINOv3PatchEmbedder`), and `compute_embedding_metrics_patch` (whole-foreground and per-body-part distances). GT mask required per frame; frames without a mask are skipped. |
| **utils.py** | PLY loading, camera setup, `render_from_cameras`, `align_rendered_to_gt_masks`, `compute_metrics`, and comparison image creation. |
| **install.sh** | Installs benchmark deps, clones SAM 3D Body, and downloads DINOv3 + SAM 3D Body checkpoints. |