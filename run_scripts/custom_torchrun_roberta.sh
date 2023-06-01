#!/bin/bash -x
#SBATCH --nodelist=ben[13]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=lora
#SBATCH --output=slurm_output/lora_glue_%j.out
#SBATCH --mem=120G

eval "$(/home/users/giovannipuccetti/miniconda3/bin/conda shell.bash hook)" # init conda
conda activate transformers

export WANDB_MODE=offline
export MASTER_PORT=12804
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export WANDB_MODE=offline

cd /home/users/giovannipuccetti/Repos/transformers
torchrun --nproc_per_node 4 examples/pytorch/text-classification/run_glue.py ./run_scripts/configs/torchrun_mnli_cfg.json
