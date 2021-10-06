#!/usr/bin/env bash

if [[ $# -lt 3 ]]; then
    echo "Must provide at least two arguments";
    echo "bash train.sh <gpuids> <dataset> <index_lang> [<model_type>, default=codebert-base]";
    exit;
fi

GPU=$1;
DATASET_NAME=$2;
INDEX_LANG=$3;
PRETRAINED_MODEL_NAME=${4:-"codebert-base"}
USER_NAME=`whoami`

if [[ $DATASET_NAME != "java" ]] && [[ $DATASET_NAME != "python" ]]; then
    echo "Dataset name must be either java or python.";
    echo "bash retrieve.sh <gpuids> <dataset> <index_lang>";
    exit;
fi
if [[ $INDEX_LANG != "java" ]] && [[ $INDEX_LANG != "python" ]]; then
    echo "Index language must be either java or python.";
    echo "bash retrieve.sh <gpuids> <dataset> <index_lang>";
    exit;
fi

CODE_BASE_DIR=`realpath ../`;
DPR_PATH="${CODE_BASE_DIR}/dpr";

MODEL_BASE_DIR="/local/wasiahmad/workspace/projects/RetrievalBasedCodeTranslation";
MODEL_DIR="${MODEL_BASE_DIR}/models/retrieval/${PRETRAINED_MODEL_NAME}";
MODEL_FILE="${MODEL_DIR}/csnet/checkpoint_best.pt";

INDEX_DIR="/local/wasiahmad/workspace/projects/RetrievalBasedCodeTranslation/indices/${INDEX_LANG}";
DATA_BASE_DIR="/home/saikatc/workspace/projects/RetrievalBasedCodeTranslation";
DATA_DIR="${DATA_BASE_DIR}/data/retrieval/${DATASET_NAME}";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=${GPU};

script="${CODE_BASE_DIR}/retriever/source/retrieve.py";
BATCH_SIZE=512;

OUT_DIR="${MODEL_BASE_DIR}/retrieved_docs/${DATASET_NAME}";
mkdir -p $OUT_DIR

SPLIT=valid; TOP_K=50;
FN_TYPE='standalone'
INPUT_FILE="${DATA_DIR}/csnet.${SPLIT}.json";

CTX_DIR="/local/wasiahmad/github_data/${INDEX_LANG}";

INDEX_PATH=$INDEX_DIR/train.functions_${FN_TYPE};
OUT_FILE="${OUT_DIR}/${SPLIT}_top${TOP_K}_${INDEX_LANG}.json";
LOG_FILE="${OUT_DIR}/${SPLIT}_top${TOP_K}_${INDEX_LANG}.log";
if [[ ! -f $OUT_FILE ]]; then
    python ${script} \
        --fp16 \
        --no_eval \
        --ctx_file $CTX_DIR/train.*.functions_${FN_TYPE}.tok \
        --model_file $MODEL_FILE \
        --index_path $INDEX_PATH \
        --qa_file $INPUT_FILE \
        --out_file $OUT_FILE \
        --n_docs $TOP_K \
        --batch_size $BATCH_SIZE \
        --match exact \
        --sequence_length 256 2>&1 | tee $LOG_FILE;
fi
