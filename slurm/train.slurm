#!/bin/bash
#SBATCH -J vit_train
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Log job info
echo $SLURM_JOB_NODELIST
nvidia-smi -L

# Initialize micromamba
source ~/.init_mamba

# Activate environment
micromamba activate stm-classification

# Run training
python train.py \
    --epochs 2 \
    --batch-size 32 \
    --samples-per-image 2 \
    --num-workers 4 \
    --output-dir outputs \
    --model-dir outputs/models
