#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import logging
import pickle
import time
from typing import Tuple, Iterator

import numpy as np

from dpr.options import add_encoder_params, setup_args_gpu, print_args, add_tokenizer_params, add_cuda_params
from dpr.indexer.faiss_indexers import DenseHNSWFlatIndexer, DenseFlatIndexer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def main(args):
    logger.info('Encoder vector_size=%d', args.vector_size)
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(args.vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(args.vector_size, args.index_buffer)

    input_paths = args.encoded_ctx_file
    logger.info('Reading all passages data from files: %s', input_paths)
    index.index_data(input_paths)
    index.serialize(args.index_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--encoded_ctx_file', nargs='+', default='[-]',
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--index_path', type=str, required=True, help='Path where to save the index')
    parser.add_argument('--vector_size', type=int, default=768, help='Hidden size of the encoders')
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')

    args = parser.parse_args()
    setup_args_gpu(args)
    print_args(args)
    main(args)
