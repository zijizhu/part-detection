#!/bin/bash

set -x

python -m pip install -U torch torchvision torchaudio jupyterlab matplotlib pandas scipy scikit-learn scikit-image pycocotools tensorboard tqdm

mkdir datasets/cub
cd datasets/cub
wget -q --show-progress "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -xzf CUB_200_2011.tgz --no-same-owner
