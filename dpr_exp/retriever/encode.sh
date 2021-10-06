#!/usr/bin/env bash

if [[ $# -lt 3 ]]; then
    echo "Must provide at least two arguments";
    echo "bash train.sh <gpuids> <dataset> <file_id> [<model_type>, default=codebert-base]";
    exit;
fi

GPU=$1;
DATASET_NAME=$2;
FILE_ID=$3;
PRETRAINED_MODEL_NAME=${4:-"bert"};

INPUT_FILENAME="api_list.json";

USER_NAME=`whoami`

# pick model from https://huggingface.co/models?search=google/bert_uncase
if [[ $PRETRAINED_MODEL_NAME == "bert" ]]; then
    pretrained_model="google/bert_uncased_L-6_H-512_A-8";
    encoder_model_type="hf_bert";
elif [[ $PRETRAINED_MODEL_NAME == "codebert-small" ]]; then
    pretrained_model="huggingface/CodeBERTa-small-v1";
    encoder_model_type="hf_roberta";
elif [[ $PRETRAINED_MODEL_NAME == "codebert-base" ]]; then
    pretrained_model="microsoft/codebert-base";
    encoder_model_type="hf_roberta";
elif [[ $PRETRAINED_MODEL_NAME == "lstm" ]]; then
    pretrained_model="huggingface/CodeBERTa-small-v1";
    encoder_model_type="hf_lstm"
else
    echo "Model type must be one of 'bert', 'codebert-small', 'codebert-base', 'lstm'";
    exit;
fi

if [[ $DATASET_NAME != "java" ]] && [[ $DATASET_NAME != "python" ]]; then
    echo "Dataset name must be either java or python.";
    echo "bash train.sh <dataset> <gpuids>";
    exit;
fi

CODE_BASE_DIR=`realpath ../`;
DPR_PATH="${CODE_BASE_DIR}/dpr";
MODEL_BASE_DIR="${CODE_BASE_DIR}/models";
MODEL_DIR="${MODEL_BASE_DIR}/${PRETRAINED_MODEL_NAME}";
CHECKPOINT_DIR_PATH="${MODEL_DIR}/pandas";
CKPT_FILENAME="checkpoint_best.pt";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=${GPU};
BATCH_SIZE=1024;

DATA_BASE_DIR="${CODE_BASE_DIR}/data/";
OUT_DIR="${CODE_BASE_DIR}/data/encoded";
mkdir -p $OUT_DIR

LOG_FILE="${OUT_DIR}/encoding_${INPUT_FILENAME%.*}.log";
script="${CODE_BASE_DIR}/retriever/source/encode.py";

python ${script} \
    --fp16 \
    --encoder_model_type $encoder_model_type \
    --pretrained_model_cfg $pretrained_model \
    --model_file ${CHECKPOINT_DIR_PATH}/${CKPT_FILENAME} \
    --batch_size $BATCH_SIZE \
    --ctx_file $DATA_BASE_DIR/$INPUT_FILENAME \
    --shard_size 5000000 \
    --out_dir ${OUT_DIR} 2>&1 | tee $LOG_FILE;
