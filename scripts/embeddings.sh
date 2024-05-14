#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
python generate_embeddings.py --config configs/embed_config.yaml
