![header](imgs/header.jpeg)

# BindingGym: A Large-Scale Mutational Dataset Toward Deciphering Protein-Protein Interactions

## Overview
BindingGym is a comprehensive dataset designed to advance the study of protein-protein interactions (PPIs). Containing millions of data points derived from binding-related deep mutational scanning (DMS) experiments, BindingGym serves as a robust resource for developing and benchmarking deep learning models. These models can predict the effects of mutations on protein binding affinity and enhance our understanding of protein-protein interactions.


## Features
- **Binding-Focused**: BindingGym exclusively collects DMS results pertinent to binding interactions, omitting data related to catalytic activity, fluorescence, and other phenotypes.
- **Comprehensive Protein Partner Modeling**: The dataset not only includes the mutated protein but also all its interacting partners, accurately reflecting the experimental setups.
- **Structure-Based Method Support**: Each target assay in BindingGym is meticulously paired with its corresponding protein complex structure through manual curation, enabling diverse computational approaches in the study of PPIs.
- **Diverse Data Splits**: The dataset is pre-split into various configurations to facilitate thorough comparisons between current and future models.
- **Inter-Assay Learning Support**: Assays are grouped and split by similarity in the inter-assay split, enhancing benchmark fairness and supporting the models' generalization capabilities in predicting PPIs.


## Getting Started
To get started with BindingGym, clone this repository and install the required dependencies. Detailed usage instructions and examples are provided below.


### Setup Environment

Create a new environment for running zero-shot prediction and training

`bash install.sh`

### Download Data

Download `input.zip`, then decompress it in `BindingGym`

```
wget https://zenodo.org/records/12514160/files/input.zip?download=1 -O input.zip
unzip input.zip
```

The MSA file is provided, if you need to run `EVE, Tranception, TranceptEVE` on your own data, you'll need to download `Uniref100.fasta` to generate the MSA file

`Uniref100.fasta`: https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz

As for how to generate MSA files, see `modelzoo/msa_BindingGym.py`

### Setup Configuration

setup your `BindingGym` environment and `uniref100` path in `modelzoo/config.sh`

### Run zero-shot prediction 

If you want to run `esm-1v`, you can

```
cd modelzoo/esm-1v
bash run.sh $gpu_id
```

If you have multi-gpus, you can run in parallel like this `bash run.sh 0,1`

For other models, you just need to go into the respective folder and run the command

Note: if you want to run `PPIformer`, you need to install an additional environment, see: https://github.com/anton-bushuiev/PPIformer. Then setup your `PPIformer` environment path in `modelzoo/config.sh`

### Training

```
cd training
conda activate BindingGym
bash run.sh $gpu_id
```

### Calculate metrics

see `calc_metric.ipynb`

### Reproduce the paper's results 

see `BindingGym_final_results.ipynb` and `BindingGym_SI.ipynb`

## Resources
The dataset is deposited at https://zenodo.org/records/12514160

## License
BindingGym is released under the MIT license.

