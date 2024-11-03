#!/bin/bash -e

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

usage="$(basename "$0") [-h] [-e ENV_NAME] [-f INSTALL_FAIROTAG] --
Install the neuralfeels environment
where:
    -h  show this help text
    -e  name of the environment, default=_neuralfeels
"

options=':he:'
while getopts $options option; do
    case "$option" in
    h)
        echo "$usage"
        exit
        ;;
    e) ENV_NAME=$OPTARG ;;
    :)
        printf "missing argument for -%s\n" "$OPTARG" >&2
        echo "$usage" >&2
        exit 1
        ;;
    \?)
        printf "illegal option: -%s\n" "$OPTARG" >&2
        echo "$usage" >&2
        exit 1
        ;;
    esac
done

# if ENV_NAME is not set, then set it to _neuralfeels
if [ -z "$ENV_NAME" ]; then
    ENV_NAME=_neuralfeels
fi

echo "Environment Name: $ENV_NAME"

unset PYTHONPATH LD_LIBRARY_PATH

# # remove any exisiting env
micromamba remove -y -n $ENV_NAME --all
micromamba env create -y --name $ENV_NAME --file environment.yml
micromamba activate $ENV_NAME

# Following the instructions from https://docs.nerf.studio/quickstart/installation.html for the right combination of cuda / torch / tinycudann
python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann -y
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 -y
micromamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Check if the install is successful
python -c "import torch; assert torch.cuda.is_available()"
if nvcc --version &>/dev/null; then
    echo "nvcc is installed and working."
else
    echo "nvcc is not installed or not in PATH."
    exit 1
fi

# Install tinycudann for instant-ngp backbone. Common issues:
# - Setup with gcc/g++ 9 if it throws errors (see issue: https://github.com/NVlabs/tiny-cuda-nn/issues/284)
# - Differing compute capabilities: https://github.com/NVlabs/tiny-cuda-nn/issues/341#issuecomment-1651814335
pip install ninja \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
    git+https://github.com/facebookresearch/segment-anything.git \ 
git+https://github.com/suddhu/tacto.git@master

# Install github.com/facebookresearch/theseus
micromamba install -y suitesparse # required for theseus
pip install theseus-ai

# Install neuralfeels package
pip install -e .

# Make entrypoint executable
chmod +x scripts/run
