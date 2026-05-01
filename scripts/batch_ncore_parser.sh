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
Usage: bash scripts/batch_ncore_parser.sh [options]

Batch convert all ncore clips under a directory. Each clip is processed
individually and results are saved in a subdirectory named after the clip.

Optional:
  --input-dir             Directory containing clip subdirectories
                          (default: outputs/ncore-clips/clips)
  --output-dir            Base output directory
                          (default: outputs/ncore_parser)
  --segmentation-ckpt     Mask2Former JIT checkpoint
                          (default: checkpoints/AH_object_seg_jit.pt)
  --camera-ids            Comma-separated camera sensor IDs (passed through)
  --track-ids             Comma-separated track IDs (passed through)
  --parallel N            Run N clips in parallel (default: 1, sequential)
  --help                  Show this help message

Extra arguments are passed through to run_ncore_parser.sh.
EOF
    exit "${1:-0}"
}

INPUT_DIR="${AH_PUBLIC_DIR}/outputs/ncore-clips/clips"
OUTPUT_DIR="${AH_PUBLIC_DIR}/outputs/ncore_parser"
SEG_CKPT="${AH_PUBLIC_DIR}/checkpoints/AH_object_seg_jit.pt"
CAMERA_IDS=""
TRACK_IDS=""
PARALLEL=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)    INPUT_DIR="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --segmentation-ckpt) SEG_CKPT="$2"; shift 2 ;;
        --camera-ids)   CAMERA_IDS="$2"; shift 2 ;;
        --track-ids)    TRACK_IDS="$2"; shift 2 ;;
        --parallel)     PARALLEL="$2"; shift 2 ;;
        --help|-h)      usage 0 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: input directory not found: ${INPUT_DIR}"
    exit 1
fi

CLIPS=("${INPUT_DIR}"/*)
NUM_CLIPS=${#CLIPS[@]}

echo "Input directory:    ${INPUT_DIR}"
echo "Output directory:   ${OUTPUT_DIR}"
echo "Segmentation ckpt:  ${SEG_CKPT}"
echo "Clips found:        ${NUM_CLIPS}"
echo "Parallel jobs:      ${PARALLEL}"
echo ""

PASS_ARGS=()
[ -n "${CAMERA_IDS}" ] && PASS_ARGS+=(--camera-ids "${CAMERA_IDS}")
[ -n "${TRACK_IDS}" ] && PASS_ARGS+=(--track-ids "${TRACK_IDS}")

FAILED=0
PROCESSED=0
SKIPPED=0
RUNNING=0

for clip_dir in "${CLIPS[@]}"; do
    [ -d "${clip_dir}" ] || continue
    clip_name=$(basename "${clip_dir}")
    json_manifest="${clip_dir}/pai_${clip_name}.json"

    if [ ! -f "${json_manifest}" ]; then
        echo "SKIP: no manifest found for ${clip_name}"
        continue
    fi

    # Incremental: skip if output already has sample_paths.json
    if [ -f "${OUTPUT_DIR}/${clip_name}/sample_paths.json" ]; then
        echo "SKIP (already parsed): ${clip_name}"
        SKIPPED=$(( SKIPPED + 1 ))
        continue
    fi

    echo "[$(( PROCESSED + 1 ))/${NUM_CLIPS}] Processing: ${clip_name}"

    (
        bash "${SCRIPT_DIR}/run_ncore_parser.sh" \
            --component-store "${json_manifest}" \
            --output-path "${OUTPUT_DIR}/${clip_name}" \
            --segmentation-ckpt "${SEG_CKPT}" \
            ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        && echo "DONE: ${clip_name}" \
        || echo "FAILED: ${clip_name}"
    ) &

    RUNNING=$(( RUNNING + 1 ))
    PROCESSED=$(( PROCESSED + 1 ))

    if [ "${RUNNING}" -ge "${PARALLEL}" ]; then
        wait -n 2>/dev/null || FAILED=$(( FAILED + 1 ))
        RUNNING=$(( RUNNING - 1 ))
    fi
done

# Wait for remaining jobs
while [ "${RUNNING}" -gt 0 ]; do
    wait -n 2>/dev/null || FAILED=$(( FAILED + 1 ))
    RUNNING=$(( RUNNING - 1 ))
done

echo ""
echo "Batch complete. Processed: ${PROCESSED}, Skipped: ${SKIPPED}, Failed: ${FAILED}"
