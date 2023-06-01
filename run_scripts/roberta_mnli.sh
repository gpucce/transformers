#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lora
#SBATCH --output=slurm_output/lora_%j.out
#SBATCH --mem=60G

eval "$(/home/users/giovannipuccetti/miniconda3/bin/conda shell.bash hook)" # init conda
conda activate transformers

export WANDB_MODE=offline
export MASTER_PORT=12804
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1
date="roberta_regular_finetune/${date}"

export WANDB_MODE=offline

export task="mnli"
printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1
export date="${date}_${task}"

torchrun --nproc_per_node 4 examples/pytorch/text-classification/run_glue.py \
    --task_name $task \
    --model_name "roberta-base" \
    --do_lora true \
    --lora_r 8 \
    --lora_alpha 8 \
    --num_train_epochs 60 \
    --learning_rate 1.e-5 \
    --run_name "roberta_lora_${date}" \
    --per_device_train_batch_size 32 \
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps 1000 \
    --output_dir "test/roberta_lora_${date}" \
    --warmup_ratio 0.06 \
    --save_total_limit 2 \
    --save_strategy "epoch"
