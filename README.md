<div align="center">

# Embryo classification based on microscopic images

## The 4th Annual International Data Science & AI Competition 2023 (World Championship 2023 - Computer Vision track)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<a href="https://www.kaggle.com/competitions/world-championship-2023-embryo-classification"><img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white"></a><br>

</div>

## Description

Classification of embryos as `good` or `not good` reflecting the embryo's potential for successful implantation.

In this repository you can train *ResNet* or *EfficientNet* based models.

Since the dataset is very small, you can train validate your models using *k-fold cross validation* using configs with `kfold`.

You can also used pretrain models from https://github.com/nasa/pretrained-microscopy-models and fine-tune them on given dataset.

## Download the data

Download the data from https://www.kaggle.com/competitions/world-championship-2023-embryo-classification/data, extract and place it in `data`.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/bartooo/embryo-classification
cd embryo-classification

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/bartooo/embryo-classification
cd embryo-classification

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## How to make a submission

Evaluate and write model's predictions to csv file

```bash
python src/eval.py experiment=experiment_name.yaml
```
