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
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

info()  { echo -e "\n\033[1;34m==>\033[0m \033[1m$*\033[0m"; }
ok()    { echo -e "\033[1;32m  ✓\033[0m $*"; }

info "Installing benchmark evaluation dependencies"
pip install pytorch-lightning yacs ninja termcolor
pip install "transformers>=4.56.0"
ok "Benchmark pip deps"

info "Downloading DINOv3 pretrained weights from HuggingFace"
hf download facebook/dinov3-vith16plus-pretrain-lvd1689m \
    --local-dir "$REPO_DIR/benchmark/checkpoints/dinov3-vith16plus-pretrain-lvd1689m"
ok "DINOv3 weights"

info "Cloning SAM 3D Body repo"
if [ ! -d "$REPO_DIR/benchmark/sam-3d-body" ]; then
    git clone https://github.com/facebookresearch/sam-3d-body "$REPO_DIR/benchmark/sam-3d-body"
else
    ok "sam-3d-body already present — skipping clone"
fi
ok "SAM 3D Body repo"

info "Downloading SAM 3D Body checkpoint (gated repo — may require HF login)"
if hf download facebook/sam-3d-body-dinov3 \
    --local-dir "$REPO_DIR/benchmark/checkpoints/sam-3d-body-dinov3" 2>&1; then
    ok "SAM 3D Body checkpoint"
else
    echo -e "\033[1;33m  ⚠\033[0m SAM 3D Body checkpoint download failed (gated repo)."
    echo "    Visit https://huggingface.co/facebook/sam-3d-body-dinov3 to request access."
    echo "    Benchmark will still run but skip embedding metrics."
fi
