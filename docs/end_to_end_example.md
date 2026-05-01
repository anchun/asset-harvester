# Full End-to-End Example

This guide walks through the complete Asset Harvester pipeline: from raw NCore V4 driving logs to simulation-ready 3D Gaussian splat assets.

## Prerequisites

Make sure you have completed the [setup and checkpoint download](../README.md#user-guide) steps described in the main README.

## Download Sample Data (Optional)

Download a sample NCore V4 clip to try the pipeline:

```bash
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore \
    --repo-type dataset \
    --local-dir ./outputs/ncore-clips \
    --include 'clips/2a6f78b4-f20c-439b-9d26-250532cb63c0/*'
```
or manually from [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore).

## Step 0: NCore V4 Data Format

Asset Harvester consumes [NCore V4](https://nvidia.github.io/ncore/index.html) driving logs, which bundle multi-camera images, lidar point clouds, and 3D cuboid track annotations into a single clip format.

To inspect a clip before running the pipeline, use [ncore_vis](https://nvidia.github.io/ncore/tools/ncore_vis.html) — a browser-based visualizer for NCore V4 data that renders camera feeds, lidar, and cuboid overlays interactively.

## Step 1: NCore Parsing

Parse NCore V4 clip data (cameras, lidar, cuboid tracks) into multi-view object crops.

```bash
bash scripts/run_ncore_parser.sh --component-store "path/to/clip.json"
```

Using the sample data:

```bash
bash scripts/run_ncore_parser.sh \
    --component-store "./outputs/ncore-clips/clips/2a6f78b4-f20c-439b-9d26-250532cb63c0/pai_2a6f78b4-f20c-439b-9d26-250532cb63c0.json"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--component-store` | *(required)* | Clip `.json` manifest, comma-separated NCore V4 component-store paths, or `.zarr.itar` globs |
| `--output-path` | `outputs/ncore_parser/<clip_uuid>` | Output directory (auto-derived from component-store path) |
| `--segmentation-ckpt` | `checkpoints/AH_object_seg_jit.pt` | Mask2Former JIT checkpoint |
| `--camera-ids` | all 5 default cameras | Comma-separated camera sensor IDs |
| `--track-ids` | all tracks | Comma-separated track IDs to process |

### Batch Processing

To process all clips under a directory in batch (results saved per clip UUID):

```bash
bash scripts/batch_ncore_parser.sh --input-dir ./outputs/ncore-clips/clips
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `outputs/ncore-clips/clips` | Directory containing clip subdirectories |
| `--output-dir` | `outputs/ncore_parser` | Base output directory (each clip saved to `<output-dir>/<clip_uuid>/`) |
| `--segmentation-ckpt` | `checkpoints/AH_object_seg_jit.pt` | Mask2Former JIT checkpoint |
| `--camera-ids` | all 5 default cameras | Comma-separated camera sensor IDs |
| `--track-ids` | all tracks | Comma-separated track IDs to process |
| `--parallel N` | 1 (sequential) | Number of clips to process in parallel |

## Step 2: Multiview Diffusion + Gaussian Lifting

Generate consistent multi-view images and lift them to 3D Gaussian splats via TokenGS.

```bash
bash run.sh --data-root ./outputs/ncore_parser/<clip_uuid> --output-dir ./outputs/ncore_harvest/<clip_uuid>
```

Using the sample data:

```bash
bash run.sh \
    --data-root ./outputs/ncore_parser/2a6f78b4-f20c-439b-9d26-250532cb63c0 \
    --output-dir ./outputs/ncore_harvest/2a6f78b4-f20c-439b-9d26-250532cb63c0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `outputs/ncore_parser/` | Input directory with `sample_paths.json` (typically `outputs/ncore_parser/<clip_uuid>`) |
| `--diffusion-ckpt` | `checkpoints/AH_multiview_diffusion.safetensors` | Diffusion model checkpoint |
| `--lifting-ckpt` | `checkpoints/AH_tokengs_lifting.safetensors` | TokenGS checkpoint |
| `--output-dir` | `outputs/` | Output directory |
| `--num-steps` | 30 | Number of diffusion inference steps |
| `--cfg-scale` | 2.0 | Classifier-free guidance scale |
| `--max-samples` | 0 (all) | Max samples to process |
| `--skip-lifting` | off | Disable TokenGS Gaussian lifting (multiview only) |
| `--offload` | off | Offload diffusion models to CPU during lifting |

Outputs per sample: `multiview/` (generated views), `3d_lifted/` (TokenGS-rendered views), `gaussians.ply`, `multiview.mp4`, `3d_lifted.mp4`.

### Batch Processing

To run multiview diffusion + lifting for all parsed clips in batch:

```bash
bash scripts/batch_run.sh --input-dir ./outputs/ncore_parser --output-dir ./outputs/ncore_harvest
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `outputs/ncore_parser` | Base directory containing per-clip ncore_parser outputs (each must have `sample_paths.json`) |
| `--output-dir` | `outputs/ncore_harvest` | Base output directory (each clip saved to `<output-dir>/<clip_uuid>/`) |
| `--diffusion-ckpt` | `checkpoints/AH_multiview_diffusion.safetensors` | Diffusion model checkpoint |
| `--lifting-ckpt` | `checkpoints/AH_tokengs_lifting.safetensors` | TokenGS checkpoint |
| `--num-steps` | 30 | Number of diffusion inference steps |
| `--cfg-scale` | 2.0 | Classifier-free guidance scale |
| `--skip-lifting` | off | Disable TokenGS Gaussian lifting (multiview only) |
| `--offload` | off | Offload diffusion models to CPU during lifting |

## Step 3: Generate External Assets Metadata to use with NVIDIA Omniverse NuRec (Optional)

To use Asset Harvester with a NuRec reconstruction, generate a `metadata.yaml` file using the script below. The Step 2 output directory can then be used as input to the NuRec workflow for asset replacement and insertion, described [here](https://sw-docs.gitlab-master-pages.nvidia.com/av-sim/early-access/nurec/use-ah-assets.html).

```bash
python asset_harvester/utils/generate_external_assets_metadata.py --input-dir ./outputs/ncore_harvest/<clip_uuid>
```

**Note:**

- **Please disable PPISP when reconstructing the 3D scene with NuRec. PPISP can introduce color-space mismatches, which lead to object over-saturation after insertion.**

- **Asset Harvester does not predict object scale. Asset insertion uses the object scales stored in metadata extracted from the original data clip. If object scale looks incorrect after insertion, please debug the source clip cuboid dimensions and the insertion code.**


If you need to regenerate masks for direct image inputs, prefer the module entry point:

```bash
python -m asset_harvester.utils.image_segment --help
```

Direct file execution also works after the editable install:

```bash
python asset_harvester/utils/image_segment.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | *(required)* | Root of the input directory (lifting output) |

## Step 4: Rescale 3DGS Assets to Real-World Dimensions (Optional)

The `gaussians.ply` files produced in Step 2 are normalized. To rescale them to real-world dimensions (using `multiview/lwh.txt`) and convert coordinates to `(x-forward, y-left, z-up)`, run:

```bash
python asset_harvester/utils/rescale_gaussians.py --input-dir ./outputs/ncore_harvest/<clip_uuid>
```

This writes a `gaussians_sim.ply` alongside each `gaussians.ply`.

### Batch Processing

To rescale all clips under a directory in batch:

```bash
bash scripts/batch_rescale_gaussians.sh --input-dir ./outputs/ncore_harvest
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `outputs/ncore_harvest` | Base directory containing per-clip harvest outputs |
| `--mode` | `forward` | `forward`: gaussians.ply → gaussians_sim.ply; `reverse`: gaussians_sim.ply → gaussians_nurec.ply |

## Benchmark Evaluation

After running step 2 with lifting, evaluate Gaussian reconstructions against ground truth views. Requires sample data that includes held-out views (e.g. `data_samples/rectified_AV_objects/`):

```bash
conda activate av-object-benchmark
python benchmark/eval.py --output_dir outputs
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | *(required)* | Root output directory from run_inference |
| `--eval_output_dir` | `<output_dir>/eval` | Where to write eval results |
| `--output_size` | 512 | Render resolution |
| `--no_comparisons` | off | Skip saving comparison images |

Results are summarized in `rendering_metrics_summary.txt` under the eval output directory.
