#!/usr/bin/env bash

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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AH_PUBLIC_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    cat <<EOF
Usage: bash scripts/batch_run.sh [options]

Batch run multiview diffusion + Gaussian lifting for all parsed clips.
Each clip's results are saved in a subdirectory named after the clip UUID.

Optional:
  --input-dir             Base directory containing per-clip ncore_parser outputs
                          (default: outputs/ncore_parser)
  --output-dir            Base output directory
                          (default: outputs/ncore_harvest)
  --diffusion-ckpt        Path to multiview diffusion checkpoint
                          (default: checkpoints/AH_multiview_diffusion.safetensors)
  --lifting-ckpt          Path to TokenGS checkpoint
                          (default: checkpoints/AH_tokengs_lifting.safetensors)
  --num-steps             Number of diffusion inference steps (default: 50)
  --cfg-scale             Classifier-free guidance scale (default: 2.0)
  --skip-lifting          Disable TokenGS Gaussian lifting (multiview only)
  --offload               Offload diffusion models to CPU during lifting
  --help                  Show this help message

Extra arguments are passed through to run.sh.
EOF
    exit "${1:-0}"
}

INPUT_DIR="${AH_PUBLIC_DIR}/outputs/ncore_parser"
OUTPUT_DIR="${AH_PUBLIC_DIR}/outputs/ncore_harvest"
DIFFUSION_CKPT=""
LIFTING_CKPT=""
NUM_STEPS=""
CFG_SCALE=""
SKIP_LIFTING=false
OFFLOAD=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)        INPUT_DIR="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --diffusion-ckpt)   DIFFUSION_CKPT="$2"; shift 2 ;;
        --lifting-ckpt)     LIFTING_CKPT="$2"; shift 2 ;;
        --num-steps)        NUM_STEPS="$2"; shift 2 ;;
        --cfg-scale)        CFG_SCALE="$2"; shift 2 ;;
        --skip-lifting)     SKIP_LIFTING=true; shift ;;
        --offload)          OFFLOAD=true; shift ;;
        --help|-h)          usage 0 ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: input directory not found: ${INPUT_DIR}"
    exit 1
fi

# Find clip directories that contain sample_paths.json
CLIPS=()
for dir in "${INPUT_DIR}"/*/; do
    [ -f "${dir}sample_paths.json" ] && CLIPS+=("${dir}")
done

NUM_CLIPS=${#CLIPS[@]}

if [ "${NUM_CLIPS}" -eq 0 ]; then
    echo "ERROR: no clips with sample_paths.json found under ${INPUT_DIR}"
    exit 1
fi

echo "Input directory:    ${INPUT_DIR}"
echo "Output directory:   ${OUTPUT_DIR}"
echo "Clips found:        ${NUM_CLIPS}"
echo ""

# Build pass-through arguments
PASS_ARGS=()
[ -n "${DIFFUSION_CKPT}" ] && PASS_ARGS+=(--diffusion-ckpt "${DIFFUSION_CKPT}")
[ -n "${LIFTING_CKPT}" ] && PASS_ARGS+=(--lifting-ckpt "${LIFTING_CKPT}")
[ -n "${NUM_STEPS}" ] && PASS_ARGS+=(--num-steps "${NUM_STEPS}")
[ -n "${CFG_SCALE}" ] && PASS_ARGS+=(--cfg-scale "${CFG_SCALE}")
[ "${SKIP_LIFTING}" = true ] && PASS_ARGS+=(--skip-lifting)
[ "${OFFLOAD}" = true ] && PASS_ARGS+=(--offload)

FAILED=0
PROCESSED=0

for clip_dir in "${CLIPS[@]}"; do
    clip_name=$(basename "${clip_dir}")
    PROCESSED=$(( PROCESSED + 1 ))

    echo "[${PROCESSED}/${NUM_CLIPS}] Processing: ${clip_name}"

    if bash "${AH_PUBLIC_DIR}/run.sh" \
        --data-root "${clip_dir}" \
        --output-dir "${OUTPUT_DIR}/${clip_name}" \
        ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
        echo "DONE: ${clip_name}"
    else
        echo "FAILED: ${clip_name}"
        FAILED=$(( FAILED + 1 ))
    fi
    echo ""
done

echo "Batch complete. Processed: ${PROCESSED}, Failed: ${FAILED}"
