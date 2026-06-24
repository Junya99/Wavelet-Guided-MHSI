# Wavelet-Guided Spectral-Spatial Learning for Medical Hyperspectral Image Segmentation

Official implementation of:

**Wavelet-Guided Spectral-Spatial Learning for Medical Hyperspectral Image Segmentation**

by Junya Ji, Nick Stone, Yanda Meng, and Xujiong Ye*

## Introduction

MICCAI 2026: The camera-ready paper will be available through the MICCAI official proceedings. The official paper link will be added here once it becomes available.

<!-- TODO: Insert Fig. 2 here after the paper is officially published. -->

This repository provides training and evaluation code for a wavelet-guided spectral-spatial segmentation framework for medical hyperspectral images. The method performs one-dimensional wavelet decomposition along the spectral dimension, incorporates low-frequency spectral priors, and improves spectral-spatial interaction using Band-Aware Aggregation and Spectral-Enhanced Skip Connections.

## Repository Structure

```text
code/
  train_ours.py              # training script for the proposed model
  eval_ours.py               # evaluation script
  train_sk.py                # pre-training script for best_sk_weight.pth
  basemodel_ours.py          # proposed model
  basemodel_sk.py            # pre-training model
  Data_Generate.py           # dataset loader
  argument.py                # data augmentation utilities
  convertbn2gn.py            # BatchNorm-to-GroupNorm conversion
  local_utils/               # losses, metrics, and logging utilities
  hamburger/                 # spectral Hamburger modules
dataset/
  train_val_test.json        # train/val/test split
checkpoints/
  best_sk_weight.pth         # pre-trained spectral prior checkpoint
  best_dice0.7646.pth        # trained segmentation checkpoint
```

## Environment

The experiments were run in a conda environment named `vmunet`.

Tested environment:

- Python 3.8.20
- PyTorch 1.13.0+cu117
- torchvision 0.14.0+cu117
- CUDA 11.7

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate vmunet
```

## Dataset

### MDC

The official MDC dataset can be found on Kaggle: [MHSI Choledoch Dataset Preprocessed Dataset](https://www.kaggle.com/datasets/hfutybx/mhsi-choledoch-dataset-preprocessed-dataset).

Due to the dataset size, we use the preprocessed data from [Dual-Stream-MHSI](https://github.com/DeepMed-Lab-ECNU/Dual-Stream-MHSI) for the experiments in our paper.

Expected dataset structure:

```text
DATA_ROOT/
  MHSI/
    xxx.hdr
    xxx.img
  Mask/
    xxx.png
  train_val_test.json
```

### MOD

The official MOD dataset can be found on Dryad: [MODID](https://datadryad.org/dataset/doi:10.5061/dryad.nvx0k6dxw#citations).

Due to the dataset size, the preprocessed MOD data can be found at [preprocessed MOD data](https://drive.google.com/drive/folders/124NCpp3-8DKHOai_4nV0ytsUm78sfKzj?usp=drive_link). We use this resized version for our experiments.

Expected dataset structure:

```text
DATA_ROOT/
  data_cropped/
    xxx.hdr
    xxx.img
  masks_cropped/
    xxx.png
  train_val_test.json
```


## Training

### 1. Pre-train the spectral prior checkpoint

To obtain `best_sk_weight.pth`, run:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 \
  code/train_sk.py \
  -r /path/to/DATA_ROOT \
  -name Wavelet-SK
```

Place the resulting checkpoint at:

```text
checkpoints/best_sk_weight.pth
```

### 2. Train the proposed model

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 \
  code/train_ours.py \
  -r /path/to/DATA_ROOT \
  -e 100 \
  -b_group 15 \
  -link_p 0 0 1 0 1 0 \
  -lf boundary \
  -bw 0.3 \
  -hw 256 320 \
  -n dual \
  -backbone resnet34 \
  -name Wavelet_Guided_MHSI
```


## Evaluation

The trained checkpoint can be found at [checkpoint](https://drive.google.com/drive/folders/19CdMipK5M8hK8uZpfaP2ZUwEOdXw-SQO?usp=drive_link). Download it and place it under `checkpoints/`.

Then run:

```bash
python code/eval_ours.py \
  -r /path/to/DATA_ROOT \
  -pm checkpoints/best_dice0.7646.pth
```

The evaluation script writes `log_eval.csv` to the directory containing the `.pth` checkpoint passed through `--pretrained_model`.

For example:

- If `--pretrained_model checkpoints/best_dice0.7646.pth` is used, the output will be `checkpoints/log_eval.csv`.


## Citation

If you find this repository useful, please cite our paper. The BibTeX entry will be added after the MICCAI official proceedings are available.

```bibtex
@inproceedings{ji2026wavelet,
  title     = {Wavelet-Guided Spectral-Spatial Learning for Medical Hyperspectral Image Segmentation},
  author    = {Ji, Junya and Stone, Nick and Meng, Yanda and Ye, Xujiong},
  booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention},
  year      = {2026}
}
```

## Acknowledgements

We thank the authors of [Dual-Stream-MHSI](https://github.com/DeepMed-Lab-ECNU/Dual-Stream-MHSI) for releasing the preprocessed MDC data and related resources.
