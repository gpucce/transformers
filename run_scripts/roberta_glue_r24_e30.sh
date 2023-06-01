#!/bin/bash -x
#SBATCH --nodelist=ben[13-19]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=lora
#SBATCH --output=slurm_output/lora_glue_%j.out
#SBATCH --mem=60G

eval "$(/home/users/giovannipuccetti/miniconda3/bin/conda shell.bash hook)" # init conda
conda activate transformers

export WANDB_MODE=offline
export MASTER_PORT=12804
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export WANDB_MODE=offline

export task="mnli"
printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1

cd /home/users/giovannipuccetti/Repos/transformers
srun --cpu_bind=v --accel-bind=gn python examples/pytorch/text-classification/run_glue.py \
    --task_name ${task} \
    --model_name "roberta-base" \
    --do_lora true \
    --lora_r 24 \
    --lora_alpha 24 \
    --num_train_epochs 30 \
    --learning_rate 4e-5 \
    --run_name "roberta_lora_${date}" \
    --per_device_train_batch_size 32 \
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps 1000 \
    --output_dir "test_multi_run_r24_e30/${date}" \
    --warmup_ratio 0.06 \
    --save_total_limit 2 \
    --save_strategy "epoch" \
    --disable_tqdm true \
    --weight_decay 0.1 \
    --task_per_node true