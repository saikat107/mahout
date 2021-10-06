import argparse
import copy
import json
import math

import nltk
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params
)
from dpr.utils.model_utils import (
    load_states_from_checkpoint,
    move_to_device
)


def get_test_data(api_lists):
    examples = []
    fail_count = 0
    for i in range(1, 4):
        example_file = f"../25_K_Examples/part-{i}-output/taken_answers_with_all_details.json"
        data = json.load(open(example_file))
        for e in tqdm(data):
            try:
                ques_id = e['question_id']
                qtitle = e['formatted_input']['question']['title']
                qdesc = e['formatted_input']['question']['ques_desc']
                codes = e['formatted_input']['answer']['code']
                apis = set()
                for c in codes:
                    tokens = nltk.wordpunct_tokenize(c)
                    for token in tokens:
                        token = token.strip()
                        if token in api_lists:
                            apis.add(token)
                api_seq = list(sorted(apis))
                if len(api_seq) <= 0:
                    continue
                e['formatted_input']['answer']["api_seq"] = api_seq
                examples.append({
                    'id': ques_id,
                    'query': qtitle.strip() + " " + qdesc.strip(),
                    "apis": api_seq,
                    'link': e['link'],
                    "example": e['formatted_input']
                })
            except Exception as ex:
                print(ex)
                fail_count += 1
    return examples


class RetrieverModel:
    def __init__(self, model_path, batch_size=64, quiet=False):
        parser = argparse.ArgumentParser()
        add_encoder_params(parser)
        add_tokenizer_params(parser)
        add_cuda_params(parser)

        self.args = parser.parse_args({})
        self.quiet = quiet
        self.args.model_file = model_path
        setup_args_gpu(self.args)
        saved_state = load_states_from_checkpoint(self.args.model_file)
        set_encoder_params_from_state(
            saved_state.encoder_params,
            self.args,
            quiet=self.quiet
        )
        self.batch_size = batch_size
        self.tensorizer, self.encoder, _ = init_biencoder_components(
            self.args.encoder_model_type,
            self.args,
            inference_only=True
        )
        self.encoder.load_state_dict(saved_state.model_dict)
        self.query_model = self.encoder.question_model
        self.document_model = self.encoder.ctx_model
        self.api_lists = json.load(open("data/api_list.json"))
        self.apis = list(sorted(self.api_lists.keys()))
        self.api_docs = [self.api_lists[a] for a in self.apis]

        self.doc_vectors = self.generate_query_vectors()

    def generate_query_vectors(self):
        return self.generate_vectors(
            model=self.document_model,
            sentences=self.api_docs,
            batch_size=self.batch_size,
            task='"API_VECTORS"'
        )

    def generate_vectors(self, model, sentences, batch_size, task):
        if not self.quiet:
            print(
                "Generating vectors for %d sentences using %s task model" % (
                    len(sentences),
                    task
                )
            )
        tensors = []
        for ex in sentences:
            tensor = self.tensorizer.text_to_tensor(ex)
            tensors.append(tensor)

        ids = torch.stack(tensors, dim=0)
        seg_batch = torch.zeros_like(ids)
        attn_mask = self.tensorizer.get_attn_mask(ids)

        model.to(self.args.device)
        l = ids.size(0)
        print(batch_size)
        start_idx = 0
        vectors = []
        num_batches = math.ceil(l / batch_size)
        print(num_batches)
        with torch.no_grad():
            batches = range(num_batches) if self.quiet else tqdm(range(num_batches))
            for _ in batches:
                end_idx = start_idx + batch_size
                if end_idx > l:
                    end_idx = l
                _ids = move_to_device(ids[start_idx:end_idx, :], self.args.device)
                _seg_batch = move_to_device(seg_batch[start_idx:end_idx, :], self.args.device)
                _attn_mask = move_to_device(attn_mask[start_idx:end_idx, :], self.args.device)
                _, _vectors, _ = model(_ids, _seg_batch, _attn_mask)
                vectors.append(_vectors.cpu())
                start_idx = end_idx
        vectors = torch.cat(vectors, dim=0)
        return vectors

    def retrieve_apis(self, examples, top_k=10):
        query_sentences = [ex["query"] for ex in examples]
        query_vectors = self.generate_vectors(
            model=self.query_model,
            sentences=query_sentences,
            batch_size=self.batch_size,
            task='"QUESTION_VECTORS"'
        )
        print(self.doc_vectors.shape, query_vectors.shape)
        similarity_results = cosine_similarity(
            query_vectors.cpu().numpy(),
            self.doc_vectors.cpu().numpy()
        )
        singled_out = []
        return_examples = []

        for exid, ex in enumerate(examples):
            example = copy.deepcopy(ex)
            pred_similaroty = [(a, s) for a, s in zip(self.apis, similarity_results[exid, :].tolist())]
            sorted_apis = sorted(pred_similaroty, key=lambda x: x[1])[::-1]
            example["expected"] = example["apis"]
            example.pop("apis", None)
            example["predicted"] = sorted_apis[:top_k]
            predictions = set([e[0] for e in sorted_apis[:top_k]])
            if len(set(example["expected"]).difference(predictions)) == 0:
                singled_out.append(example)
                pass
            return_examples.append(example)
            pass
        return return_examples, singled_out


if __name__ == '__main__':
    apis = list(json.load(open("data/api_list.json")).keys())
    test_examples = get_test_data(apis)
    model_path = "models/bert/pandas_1/dpr_biencoder.3.2387"
    retriever = RetrieverModel(
        model_path=model_path,
        batch_size=64,
        quiet=False
    )
    _, singles = retriever.retrieve_apis(test_examples)
    print(len(singles))
    print(json.dumps(singles[4:10], indent=4))
    correct_example_file = open("Correct_solutions.json", "w")
    json.dump(singles, correct_example_file, indent=4)
    correct_example_file.close()
