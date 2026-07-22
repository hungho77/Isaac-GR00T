set -x -euo pipefail

export NUM_GPUS=1
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-3000}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"


WANDB_FLAG=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAG+=(--use_wandb)
fi

# torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base_model_path weights/nvidia/GR00T-N1.7-3B \
    --dataset_path  /mnt/data/sftp/data/hunght23/DATA_ZOO/aloha_left_arm_pick_carrot_put_cup_added_recovery \
    --modality_config_path examples/ALOHA/mobile_aloha_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir tmp/mobile_aloha_finetune_left_arm_pick_carrot_put_cup_21072026 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --max_steps "$MAX_STEPS" \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    "${WANDB_FLAG[@]}" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --shard_size "$SHARD_SIZE" \
    --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH" \
    --episode_sampling_rate "$EPISODE_SAMPLING_RATE"
