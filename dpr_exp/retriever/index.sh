#!/usr/bin/env bash

if [[ $# -lt 1 ]]; then
    echo "Must provide at least two arguments";
    echo "bash train.sh <dataset> [<file_id>, default=0]";
    exit;
fi

DATASET_NAME=$1;
FILE_ID=${2:-0};
USER_NAME=`whoami`

if [[ $DATASET_NAME != "java" ]] && [[ $DATASET_NAME != "python" ]]; then
    echo "Dataset name must be either java or python.";
    echo "bash train.sh <gpuids> <dataset>";
    exit;
fi

ENCODED_DIR="/local/${USER_NAME}/workspace/projects/RetrievalBasedCodeTranslation/encoded/${DATASET_NAME}";
INDEX_DIR="/local/${USER_NAME}/workspace/projects/RetrievalBasedCodeTranslation/indices/${DATASET_NAME}";
mkdir -p $INDEX_DIR

CODE_BASE_DIR=`realpath ../`;
script="${CODE_BASE_DIR}/retriever/source/index.py";
export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;

FN_TYPE='standalone'

if [[ $FN_TYPE = "standalone" ]]; then
    OUTPUT_ENCODED_FILE=${ENCODED_DIR}/train.*.functions_${FN_TYPE}.*.pkl;
    INDEX_PATH=$INDEX_DIR/train.functions_${FN_TYPE};
    LOG_FILE="${INDEX_DIR}/indexing.functions_${FN_TYPE}.log";
else
    OUTPUT_ENCODED_FILE=${ENCODED_DIR}/train.${FILE_ID}.functions_${FN_TYPE}.*.pkl;
    INDEX_PATH=$INDEX_DIR/train.${FILE_ID}.functions_${FN_TYPE};
    LOG_FILE="${INDEX_DIR}/indexing.${FILE_ID}.functions_${FN_TYPE}.log";
fi

if [[ ! -f ${INDEX_PATH}.index.dpr ]]; then
    python ${script} \
        --encoded_ctx_file $OUTPUT_ENCODED_FILE \
        --index_path $INDEX_PATH \
        --vector_size 768 \
        --index_buffer 50000 \
        --sequence_length 256 2>&1 | tee $LOG_FILE;
fi
