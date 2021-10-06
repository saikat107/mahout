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
import os
import csv
import pathlib
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches_by_id
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
            self,
            question_encoder: nn.Module,
            batch_size: int,
            tensorizer: Tensorizer,
            index: DenseIndexer
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
            self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_qa_jsonl_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location, encoding='utf-8') as reader:
        data = json.load(reader)
        for ex in data:
            yield ex['question'], ex['positive_ctxs'][0]['answers']


def validate(
        answers: List[List[str]],
        result_ctx_ids: List[Tuple[List[object], List[float]]],
        workers_num: int, match_type: str
) -> List[List[bool]]:
    match_stats = calculate_matches_by_id(answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
        questions: List[str],
        answers: List[List[str]],
        top_passages_and_scores: List[Tuple[List[object], List[float]]],
        per_question_hits: List[List[bool]],
        out_file: str
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'id': i,
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)


def save_retrieved_docs(
        questions: List[str],
        answers: List[List[str]],
        top_passages_and_scores: List[Tuple[List[object], List[float]]],
        out_file: str,
        documents: Dict[object, str],
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(results_and_scores[0])

        if documents is not None:
            doc = {
                'id': i,
                'question': q,
                'answers': q_answers,
                'ctxs': [
                    {
                        'id': results_and_scores[0][c],
                        "text": documents.get(results_and_scores[0][c], None),
                        'score': scores[c],
                    } for c in range(ctxs_num)
                ]
            }
        else:
            doc = {
                'id': i,
                'question': q,
                'answers': q_answers,
                'ctxs': [
                    {
                        'id': results_and_scores[0][c],
                        'score': scores[c],
                    } for c in range(ctxs_num)
                ]
            }
        merged_data.append(doc)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)


def load_passages(ctx_files: List[str]) -> Dict[object, str]:
    docs = dict()
    for ctx_file in ctx_files:
        filepref = pathlib.Path(ctx_file).stem
        with open(ctx_file) as f:
            for idx, line in enumerate(f):
                function = line.strip()
                doc_id = filepref + '.' + str(idx)
                docs[doc_id] = function

    return docs


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)
    encoder = encoder.question_model

    # the following doesn't work for fp16
    # encoder, _ = setup_for_distributed_mode(
    #     encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
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

    prefix_len = len('ctx_model.')
    question_encoder_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
        key.startswith('ctx_model.')
    }
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    index = DenseHNSWFlatIndexer(vector_size) if args.hnsw_index \
        else DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)
    retriever.index.deserialize_from(args.index_path)

    # get questions & answers
    questions = []
    question_answers = []

    for ds_item in parse_qa_jsonl_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    if args.shard_size == -1:
        shard_size = len(questions)
    else:
        shard_size = args.shard_size

    all_passages = None
    if args.ctx_file:
        all_passages = load_passages(args.ctx_file)

    current_shard = 1
    start_index = 0

    while start_index < len(questions):
        logger.info('Processing Shard %d with data from index %d to %d' % (
            current_shard, start_index, start_index + shard_size))
        current_shard_questions = questions[start_index:(start_index + shard_size)]
        # This time, let's query with the code.
        current_shard_answers = question_answers[start_index:(start_index + shard_size)]
        current_shard_queries = [x[0] for x in current_shard_answers]
        start_index += shard_size
        out_file = None
        if args.out_file:
            out_file = args.out_file + '.shard' + str(current_shard)
        current_shard += 1
        import numpy as np
        if np.random.uniform() > 0.5:
            print("\n\n")
            print("=" * 100)
            print(current_shard_queries[0])
            print('=' * 100)
            print("\n\n")

        questions_tensor = retriever.generate_question_vectors(current_shard_queries)
        questions_tensor = questions_tensor.numpy()
        if questions_tensor.dtype != 'float32':
            questions_tensor = np.float32(questions_tensor)
        # get top k results
        top_ids_and_scores = retriever.get_top_docs(questions_tensor, args.n_docs)
        if args.no_eval:
            # only perform retrieval, no validation
            if args.out_file:
                save_retrieved_docs(
                    current_shard_questions, current_shard_answers, top_ids_and_scores, out_file, all_passages
                )
        else:
            questions_doc_hits = validate(
                current_shard_answers, top_ids_and_scores, args.validation_workers, args.match
            )
            if args.out_file:
                save_results(
                    current_shard_questions, current_shard_answers, top_ids_and_scores, questions_doc_hits, out_file
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default=None, help="Question and answers in JSON format")
    parser.add_argument('--ctx_file', type=str, nargs='+', default=None, help="File containing function content")
    parser.add_argument('--out_file', type=str, default=None, help='output .json file path to write results to')
    parser.add_argument('--index_path', type=str, required=True, help='Path from where to load index')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string', 'exact'],
                        help="Answer matching logic type")
    parser.add_argument('--n_docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--no_eval", action='store_true', help='If enabled, only perform retrieval')
    parser.add_argument('--shard_size', type=int,
                        help='Number of examples to be considered for retrieval and writing at a time, '
                             '-1 for processing complete data.',
                        default=-1)

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)
