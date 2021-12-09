#!/usr/bin/env bash

#if [[ $# -lt 1 ]]; then
#    echo "Must provide at least two arguments";
#    echo "bash train.sh <gpuids> <num_negative_ctx> [<model_type>, default=codebert-base]";
#    exit;
#fi

GPU=$1;
PRETRAINED_MODEL_NAME=${2:-"bert"}

CODE_BASE_DIR=`realpath ../`;
DPR_PATH="${CODE_BASE_DIR}/dpr";
LR=5e-5
# pick model from https://huggingface.co/models?search=google/bert_uncase
if [[ $PRETRAINED_MODEL_NAME == "bert" ]]; then
    pretrained_model="google/bert_uncased_L-6_H-512_A-8";
    encoder_model_type="hf_bert"
    PER_GPU_TRAIN_BATCH_SIZE=4;
elif [[ $PRETRAINED_MODEL_NAME == "codebert-small" ]]; then
    pretrained_model="huggingface/CodeBERTa-small-v1";
    encoder_model_type="hf_roberta"
    PER_GPU_TRAIN_BATCH_SIZE=4;
elif [[ $PRETRAINED_MODEL_NAME == "codebert-base" ]]; then
    pretrained_model="microsoft/codebert-base";
    encoder_model_type="hf_roberta"
    PER_GPU_TRAIN_BATCH_SIZE=2;
elif [[ $PRETRAINED_MODEL_NAME == "lstm" ]]; then
    pretrained_model="huggingface/CodeBERTa-small-v1";
    encoder_model_type="hf_lstm"
    PER_GPU_TRAIN_BATCH_SIZE=80;
    LR=1e-3;
else
    echo "Model type must be one of 'bert', 'codebert-small', 'codebert-base', 'lstm'";
    exit;
fi

USER_NAME=`whoami`
MODEL_BASE_DIR="${CODE_BASE_DIR}/models";
MODEL_DIR="${MODEL_BASE_DIR}/${PRETRAINED_MODEL_NAME}";
DATA_BASE_DIR="${CODE_BASE_DIR}/data";
script="${CODE_BASE_DIR}/retriever/source/train.py";
export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=$GPU;

NUMBER_OF_DEVICES=`echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c`;
NUMBER_OF_DEVICES=`expr ${NUMBER_OF_DEVICES} + 1`;

EFFECTIVE_BATCH_SIZE=96;
UPDATE_FREQ=1;

REQUIRED_NUM_GPU=$(($EFFECTIVE_BATCH_SIZE / $PER_GPU_TRAIN_BATCH_SIZE));
BATCH_SIZE=$(($PER_GPU_TRAIN_BATCH_SIZE * $NUMBER_OF_DEVICES))
if [[ "$BATCH_SIZE" -gt "$EFFECTIVE_BATCH_SIZE" ]]; then
    echo "Warning: $REQUIRED_NUM_GPU GPUs are enough for fine-tuning.";
else
    UPDATE_FREQ=$(($EFFECTIVE_BATCH_SIZE / $BATCH_SIZE));
fi

function train() {
    NUM_NEG_CTX=$1;
    CHECKPOINT_DIR_PATH="${MODEL_DIR}/pandas_${NUM_NEG_CTX}";
    mkdir -p $CHECKPOINT_DIR_PATH;
    CKPT_FILENAME="dpr_biencoder";
    train_files=()
    dev_files=()
    for dataset in "pandas_${NUM_NEG_CTX}"; do
        train_files+=(${DATA_BASE_DIR}/${dataset}/train.json);
        dev_files+=(${DATA_BASE_DIR}/${dataset}/valid.json);
    done
    LOG_FILE="${CHECKPOINT_DIR_PATH}/training.log";
    python ${script} \
        --fp16 \
        --dataset csnet \
        --output_dir ${CHECKPOINT_DIR_PATH} \
        --checkpoint_file_name ${CKPT_FILENAME} \
        --batch_size $BATCH_SIZE \
        --dev_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $UPDATE_FREQ \
        --train_file "${train_files[@]}" \
        --dev_file "${dev_files[@]}" \
        --sequence_length 512 \
        --num_train_epochs 20 \
        --eval_per_epoch 1 \
        --learning_rate $LR \
        --max_grad_norm 2.0 \
        --encoder_model_type ${encoder_model_type} \
        --pretrained_model_cfg ${pretrained_model} \
        --val_av_rank_start_epoch 0 \
        --warmup_steps 500 \
        --val_av_rank_max_qs 50000 \
        --seed 1234 2>&1 | tee $LOG_FILE;
}

for num_ctx in 2; do
  echo "Training for ${num_ctx} number of negative contexts!"
  train ${num_ctx};
done
