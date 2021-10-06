#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import re
import pathlib

import argparse
import time
import logging
import json
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint, move_to_device

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def gen_ctx_vectors(
        ctx_rows: List[Tuple[object, str, str]],
        model: nn.Module,
        tensorizer: Tensorizer,
        insert_title: bool = True,
        pad_to_local_max: bool = False
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []

    last_wall_time = start_time = time.time()
    pad_id = tensorizer.tokenizer.pad_token_id
    if pad_to_local_max:
        tensorizer.pad_to_max = False
    for j, batch_start in enumerate(range(0, n, bsz)):
        if pad_to_local_max:
            batch_token_tensors, lengths = [], []
            for ctx in ctx_rows[batch_start:batch_start + bsz]:
                t = tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None)
                batch_token_tensors.append(t)
                lengths.append(t.size(0))

            max_len = max(lengths)
            for i in range(len(batch_token_tensors)):
                l = batch_token_tensors[i].size(0)
                if l < max_len:
                    batch_token_tensors[i] = torch.cat([
                        batch_token_tensors[i], torch.tensor([pad_id] * (max_len - l))
                    ], dim=0)

        else:
            batch_token_tensors = [
                tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None)
                for ctx in ctx_rows[batch_start:batch_start + bsz]
            ]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), args.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend([
            (ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))
        ])

        if total % 10 == 0:
            current_time = time.time()
            logger.info(
                'Encoding: #passages: %d, wall: %f, total_wall: %f.',
                total, current_time - last_wall_time, current_time - start_time
            )
            last_wall_time = current_time

    return results


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    # the following doesn't work for fp16
    # encoder, _ = setup_for_distributed_mode(
    #     encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16, args.fp16_opt_level
    # )
    encoder.to(args.device)
    if args.fp16:
        encoder = encoder.half()
        if args.n_gpu > 1:
            encoder = torch.nn.DataParallel(encoder)

    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    prefix_len = len('ctx_model.')
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items()
        if key.startswith('ctx_model.')
    }
    model_to_load.load_state_dict(ctx_state)

    logger.info('reading data from file=%s', args.ctx_file)
    filepref = pathlib.Path(args.ctx_file).stem

    def process_data():
        indices = np.argsort(lengths).tolist()
        sorted_data = [data[i] for i in indices]
        logger.info(
            'Producing encodings for passages range: %d to %d',
            shard_id * args.shard_size, (shard_id + 1) * args.shard_size
        )
        vectors = gen_ctx_vectors(
            sorted_data,
            encoder,
            tensorizer,
            insert_title=False,
            pad_to_local_max=True
        )
        file = os.path.join(args.out_dir, filepref + '.{}'.format(shard_id) + '.pkl')
        pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
        logger.info('Writing results to %s' % file)
        with open(file, mode='wb') as f:
            pickle.dump(vectors, f)

    total_data, shard_id = 0, 0
    data, lengths = [], []
    start_time = time.time()
    with open(args.ctx_file) as f:
        descriptions = json.load(f)
        for api in descriptions.keys():
            description = descriptions[api]
            doc_id = api
            data.append((doc_id, description, None))
            lengths.append(len(description.split()))

            if len(data) == args.shard_size:
                total_data += len(data)
                process_data()
                data, lengths = [], []
                shard_id += 1

    if len(data) > 0:
        total_data += len(data)
        process_data()

    logger.info(
        'Total passages processed %d. time elapsed %f sec.', total_data, time.time() - start_time
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', required=True, type=str, help='Input file to encode')
    parser.add_argument('--out_dir', required=True, type=str, help='directory path where results will be saved')
    parser.add_argument('--shard_size', type=int, default=50000, help="Total amount of data in 1 shard")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--dataset', type=str, default=None, help=' to build correct dataset parser ')

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)

    main(args)
