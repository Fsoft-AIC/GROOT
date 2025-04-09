<div align="center">

# GROOT: Effective Design of Biological Sequences with Limited Experimental Data
</div>

## Table of Contents:

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
    - [Setup benchmarks](#setup-benchmarks)
    - [Download pretrained weights](#download-pretrained-weights)
- [Usage](#usage)
    - [Training](#training)
    - [Optimization](#optimization)
- [Citation](#citation)

## Introduction
This is the official implementation of the paper "GROOT: Effective Design of Biological Sequences with Limited Experimental Data".

![GROOT](./static/framework.png)

## Dependencies
The [`environment.yaml`](environment.yml) file contains the necessary dependencies to run GROOT. It requires Python 3.10 and CUDA version 11.8 to run the main pipeline.

## Installation

Follow these steps to install GROOT:

```shell
conda env create -f environment.yml
conda activate groot
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -e .
```

Please setup [wandb](https://wandb.ai/) to train VAE model with logging utilities. Otherwise, you can disable it by prepending `WANDB_DISABLED=true` before running python script.

### Setup benchmarks
The checkpoints of oracles provided by [GGS](https://github.com/kirjner/GGS) are already included in the repository, under the directory [`ckpts`](./ckpts/).

The benchmark datasets are provided in [`data`](./data/) directory. Otherwise, you can generate your own sub-dataset by spliting the ground-truth with script [`split_data.py`](./scripts/split_data.py).

### Download pretrained weights
We provide the pretrained VAE model for AAV and GFP dataset [here](https://drive.google.com/drive/folders/1NdrmB4NgG-V5mIv_JMJyw_pln2M9N7VD?usp=sharing).

## Usage

### Training

To train VAE model for each benchmark dataset, run script [`train_vae.py`](./scripts/train_vae.py) as follows:

```shell
python scripts/train_vae.py [CONFIG_FILE] --csv_file [CSV_FILE] --devices [DEVICES] --output_dir [OUTPUT_DIR] --dataset [DATASET]
```

| Parameter | Type | Description | Options | Required | Default |
|--|--|--|--|--|--|
|`config_file`|str|Path to config module||✔️||
|`output_dir`|str|Path to output directory||✔️||
|`csv_file`|str|Path to training data||✔️||
|`dataset`|str|Training dataset|AAV, GFP|✔️||
|`expected_kl`|float|Expected KL-Divergence value|||40|
|`batch_size`|int|Batch size|||64|
|`devices`|str|Training devices separated by comma|||-1|
|`ckpt_path`|str|Path to checkpoint to resume training|||None|
|`wandb_id`|str|WandB experimental id to resume|||None|
|`prefix`|str|Prefix to add to checkpoint file|||""|

You can use the configuration templates in [`vae`](./scripts/configs/vae/) directory. Checkpoints will be saved in `[OUTPUT_DIR]/vae_ckpts/` folder.

### Optimization

To perform optimization, run script [`optimize.py`](./scripts/optimize.py) as follows:

```shell
python scripts/optimize.py [CONFIG_FILE] --devices [DEVICES] --dataset [DATASET] --model_ckpt_path [VAE_CKPT_PATH] --optim_config_path [OPTIM_CONFIG_PATH] --level [LEVEL] --output_dir [OUTPUT_DIR]
```

| Parameter | Type | Description | Options | Required | Default |
|--|--|--|--|--|--|
|`config_file`|str|Path to config module||✔️||
|`model_ckpt_path`|str|Path to VAE checkpoint||✔️||
|`dataset`|str|Training dataset|AAV, GFP|✔️||
|`level`|str|Benchmark difficulty|easy, medium, hard, harder1, harder2, harder3|✔️||
|`optim_config_path`|str|Path to optimization configuration file||✔️||
|`output_dir`|str|Path to output directory||✔️||
|`batch_size`|int|Batch size|||128|
|`devices`|str|Training devices separated by comma|||-1|
|`changes`|list[str]|List of modifications made to replace argument in `optim_config_file`|||[]|

For more details about these arguments, refer to the [`optimize.sh`](./scripts/optimize.sh) file.

## Citation
If our paper or codebase aids your research, please consider citing us:
```bibtex
@inproceedings{10.1145/3690624.3709291,
author = {Tran, Thanh V. T. and Ngo, Nhat Khang and Nguyen, Viet Anh and Hy, Truong Son},
title = {GROOT: Effective Design of Biological Sequences with Limited Experimental Data},
year = {2025},
isbn = {9798400712456},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3690624.3709291},
doi = {10.1145/3690624.3709291},
abstract = {Latent space optimization (LSO) is a powerful method for designing discrete, high-dimensional biological sequences that maximize expensive black-box functions, such as wet lab experiments. This is accomplished by learning a latent space from available data and using a surrogate model fΦ to guide optimization algorithms toward optimal outputs. However, existing methods struggle when labeled data is limited, as training fΦ with few labeled data points can lead to subpar outputs, offering no advantage over the training data itself. We address this challenge by introducing GROOT, a GRaph-based Latent SmOOThing for Biological Sequence Optimization. In particular, GROOT generates pseudo-labels for neighbors sampled around the training latent embeddings. These pseudo-labels are then refined and smoothed by Label Propagation. Additionally, we theoretically and empirically justify our approach, demonstrate GROOT's ability to extrapolate to regions beyond the training set while maintaining reliability within an upper bound of their expected distances from the training regions. We evaluate GROOT on various biological sequence design tasks, including protein optimization (GFP and AAV) and three tasks with exact oracles from Design-Bench. The results demonstrate that GROOT equalizes and surpasses existing methods without requiring access to black-box oracles or vast amounts of labeled data, highlighting its practicality and effectiveness. We release our code at https://github.com/Fsoft-AIC/GROOT.},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
pages = {1385–1396},
numpages = {12},
keywords = {label propagation, landscape smoothing, latent space optimization, protein optimization},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```
