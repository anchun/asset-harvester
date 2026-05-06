#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AH_PUBLIC_DIR="$(dirname "$SCRIPT_DIR")"

python3 "${AH_PUBLIC_DIR}/asset_harvester/utils/collect_sim_assets.py" "$@"