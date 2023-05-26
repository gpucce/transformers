
printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1

python transformers/examples/pytorch/text-classification/run_glue.py \
    --task_name "sst2" \
    --model_name "roberta-base" \
    --do_lora true \
    --lora_r 8 \
    --lora_alpha 8 \
    --num_train_epochs 60 \
    --learning_rate 1.e-5 \
    --run_name "roberta_relora_${date}" \
    --per_device_train_batch_size 32 \
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps 1000 \
    --output_dir "test/roberta_relora_${date}" \
    --warmup_ratio 0.06 \
    --save_total_limit 2 \
    --save_strategy "epoch" \
    --relora_epoch 4
