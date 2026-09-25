#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SoftMimicGen installation script
# Usage: bash softmimicgen.sh

set -e

ISAACLAB_REPO="https://github.com/masoudmoghani/IsaacLab.git"
ISAACLAB_BRANCH="softmimicgen"
ISAACLAB_DIR="third_party/IsaacLab"
RAPPRENTICE_REPO="https://github.com/masoudmoghani/rapprentice.git"
RAPPRENTICE_BRANCH="main"
RAPPRENTICE_DIR="third_party/rapprentice"
HF_DATASET_REPO="masoudmoghani/SoftMimicGen"
ANNOTATED_DATASET_DIR="datasets/annotated_dataset"
ASSET_DATA_DIR="source/softmimicgen_assets/data"
CONDA_ENV_NAME="softmimicgen"

echo "[SoftMimicGen] Starting installation..."

# Step 1: Create conda environment
echo "[SoftMimicGen] Creating conda environment '${CONDA_ENV_NAME}'..."
conda create -n ${CONDA_ENV_NAME} python=3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

# Step 2: Install Isaac Sim via pip
echo "[SoftMimicGen] Installing Isaac Sim 5.1.0..."
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Step 3: Install PyTorch with CUDA support
echo "[SoftMimicGen] Installing PyTorch..."
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Step 4: Clone Isaac Lab if not present
if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "[SoftMimicGen] Cloning Isaac Lab..."
    git clone -b ${ISAACLAB_BRANCH} ${ISAACLAB_REPO} ${ISAACLAB_DIR}
else
    echo "[SoftMimicGen] Isaac Lab already exists at ${ISAACLAB_DIR}, skipping clone."
fi

# Step 5: Install Isaac Lab
echo "[SoftMimicGen] Installing Isaac Lab..."
export ACCEPT_EULA=yes
export OMNI_KIT_ACCEPT_EULA=YES
cd ${ISAACLAB_DIR}
./isaaclab.sh -i
cd ../..

# Step 6: Install rapprentice
echo "[SoftMimicGen] Installing rapprentice..."
if [ ! -d "$RAPPRENTICE_DIR" ]; then
    git clone -b ${RAPPRENTICE_BRANCH} ${RAPPRENTICE_REPO} ${RAPPRENTICE_DIR}
else
    echo "[SoftMimicGen] rapprentice already exists at ${RAPPRENTICE_DIR}, skipping clone."
fi
pip install -e ${RAPPRENTICE_DIR}

# Step 7: Fix compatible dependency versions
pip install qpsolvers==4.8.1

# Step 8: Install SoftMimicGen packages
echo "[SoftMimicGen] Installing SoftMimicGen packages..."
pip install -e source/softmimicgen
pip install -e source/softmimicgen_assets
pip install -e source/softmimicgen_tasks

# Step 9: Download datasets from Hugging Face (if not already present)
if [ ! -d "$ANNOTATED_DATASET_DIR" ]; then
    echo "[SoftMimicGen] Downloading annotated datasets..."
    hf download ${HF_DATASET_REPO} annotated_dataset/ --repo-type dataset --local-dir datasets
else
    echo "[SoftMimicGen] Annotated datasets already exist at ${ANNOTATED_DATASET_DIR}, skipping download."
fi

if [ ! -d "$ASSET_DATA_DIR" ]; then
    echo "[SoftMimicGen] Downloading asset data..."
    hf download ${HF_DATASET_REPO} data/ --repo-type dataset --local-dir source/softmimicgen_assets
else
    echo "[SoftMimicGen] Asset data already exists at ${ASSET_DATA_DIR}, skipping download."
fi

echo ""
echo "[SoftMimicGen] Installation complete!"
echo ""
echo "  To activate the environment:"
echo "    conda activate ${CONDA_ENV_NAME}"
echo ""
echo "  To verify installation:"
echo "    python scripts/environments/list_envs.py"
echo ""
