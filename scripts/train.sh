#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
python train.py --config configs/experiment_config.yaml
