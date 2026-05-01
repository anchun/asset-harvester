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
Usage: bash scripts/batch_rescale_gaussians.sh [options]

Batch rescale 3DGS assets for all clips under a directory.
Each clip subdirectory is processed independently.

Optional:
  --input-dir     Base directory containing per-clip harvest outputs
                  (default: outputs/ncore_harvest)
  --mode          forward (gaussians.ply -> gaussians_sim.ply) or
                  reverse (gaussians_sim.ply -> gaussians_nurec.ply)
                  (default: forward)
  --help          Show this help message
EOF
    exit "${1:-0}"
}

INPUT_DIR="${AH_PUBLIC_DIR}/outputs/ncore_harvest"
MODE="forward"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)  INPUT_DIR="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --help|-h)    usage 0 ;;
        *)            echo "Unknown argument: $1"; usage 1 ;;
    esac
done

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: input directory not found: ${INPUT_DIR}"
    exit 1
fi

# Find clip directories (those containing at least one gaussians.ply somewhere)
CLIPS=()
for dir in "${INPUT_DIR}"/*/; do
    [ -d "${dir}" ] && CLIPS+=("${dir%/}")
done

NUM_CLIPS=${#CLIPS[@]}

if [ "${NUM_CLIPS}" -eq 0 ]; then
    echo "ERROR: no clip subdirectories found under ${INPUT_DIR}"
    exit 1
fi

echo "Input directory:  ${INPUT_DIR}"
echo "Mode:             ${MODE}"
echo "Clips found:      ${NUM_CLIPS}"
echo ""

FAILED=0
PROCESSED=0

for clip_dir in "${CLIPS[@]}"; do
    clip_name=$(basename "${clip_dir}")
    PROCESSED=$(( PROCESSED + 1 ))

    echo "[${PROCESSED}/${NUM_CLIPS}] Rescaling: ${clip_name}"

    if python3 "${AH_PUBLIC_DIR}/asset_harvester/utils/rescale_gaussians.py" \
        --input-dir "${clip_dir}" \
        --mode "${MODE}"; then
        echo "DONE: ${clip_name}"
    else
        echo "FAILED: ${clip_name}"
        FAILED=$(( FAILED + 1 ))
    fi
done

echo ""
echo "Batch complete. Processed: ${PROCESSED}, Failed: ${FAILED}"
