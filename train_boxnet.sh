#!/bin/bash
# Usage: 
# sh train_boxnet.sh $NODE_NUM $CURRENT_NODE_RANK $GPUS_PER_NODE
# For example: to train in one machine with 8 GPUs, use:
# sh train_boxnet.sh 1 0 8

ROOT_DIR=../results

MODEL_NAME=stablediffusion_bbox

MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

NNODES=$1
GPUS_PER_NODE=$3

MICRO_BATCH_SIZE=6

CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1

cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON

DATA_ARGS="\
        --webdataset_base_urls \
        ***/{*****..*****}.tar \
        --num_workers 2 \
        --batch_size $MICRO_BATCH_SIZE \
        --shard_width 5 \
        --hr_size 512 \
        --train_split 1.0 \
        --val_split 0.0 \
        --test_split 0.0 \
        --resample_train \
        --shuffle_cat \
        --test_repeat 1 \
        --no_class \
        --set_cost_class 100 \
        "

MODEL_ARGS="\
        --model_path ***/stable-diffusion-v1-5 \
        --learning_rate 1e-4 \
        --weight_decay 1e-4 \
        --warmup_steps 5000 \
        --loss_proportion 0.0 \
        --min_learning_rate 1e-7 \
        --lr_decay_steps 50000 \
        --timestep_range 0 1000 \
        --scheduler_type cosine_with_restarts \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        --save_steps 3000 \
        "

## --strategy deepspeed_stage_${ZERO_STAGE} \
TRAINER_ARGS="\
        --max_epoch 10 \
        --accelerator gpu \
        --devices $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --precision 32 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

python -m torch.distributed.run \
    --nnodes $NNODES \
    --master_addr *** \
    --master_port *** \
    --node_rank $2 \
    --nproc_per_node $GPUS_PER_NODE \
    train_boxnet.py $options
