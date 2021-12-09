import copy
import json
import numpy as np


class Vocab:
    def __init(self, vocab_file):
        self.token_to_idx = json.load(open(vocab_file))
        self.idx_to_tokens = {self.token_to_idx[token]: token for token in self.token_to_idx.keys()}
        pass


class BiGramModel:
    def __init__(self, train_data):
        self.bigram_freq = {}
        self.conditional_freq = {}
        with open(train_data) as fp:
            for line in fp:
                line = line.strip()
                words = ["<s>"] + line.split() + ["</s>"]
                l = len(words)
                for i in range(l - 1):
                    t0 = words[i]
                    t1 = words[i + 1]
                    if t0 not in self.conditional_freq.keys():
                        self.conditional_freq[t0] = {}
                    if t1 not in self.conditional_freq[t0].keys():
                        self.conditional_freq[t0][t1] = 0
                    self.conditional_freq[t0][t1] += 1
                    bigram_tuple = (words[i], words[i + 1])
                    if bigram_tuple not in self.bigram_freq:
                        self.bigram_freq[bigram_tuple] = 0
                    self.bigram_freq[bigram_tuple] += 1
        self.conditional_prob = {}
        for t0 in self.conditional_freq:
            frequencies = self.conditional_freq[t0]
            total = sum([self.conditional_freq[t0][t1] for t1 in frequencies])
            if total == 0:
                total = 100000000
            self.conditional_prob[t0] = {
                t1: self.conditional_freq[t0][t1] / total for t1 in self.conditional_freq[t0].keys()
            }

    def calculate_probs(self, tokens):
        if tokens[0] != "<s>":
            tokens = ["<s>"] + tokens
        if tokens[-1] != "</s>":
            tokens = tokens + ["</s>"]
        l = len(tokens)
        prob = 0.
        for i in range(l - 1):
            t0 = tokens[i]
            t1 = tokens[i + 1]
            if t0 not in self.conditional_prob.keys():
                p = 1e-9
            elif t1 not in self.conditional_prob[t0].keys():
                p = 1e-9
            else:
                p = self.conditional_prob[t0][t1]
                if p == 0:
                    p = 1e-9
            prob += np.log(p)
        prob = prob / l
        return prob

    def get_top_tokens(self, token, mask):
        if token not in self.conditional_prob:
            return ["</s>"]
        probabilities = copy.copy(self.conditional_prob[token])
        mask_token_probs = []
        for t, prior in mask:
            if t not in probabilities:
                p = np.log(1e-9) * (prior)
            else:
                p = np.log(probabilities[t]) * (prior)
            mask_token_probs.append((t, p))
        mask_token_probs = sorted(mask_token_probs, key=lambda x: x[1], reverse=True)
        return mask_token_probs

    def beam_search(self, mask, beam_size=20, min_len=1, max_len=5):
        if isinstance(mask, str):
            mask = [(m, 1.0) for m in mask]
        complete_beams = []
        beam = [
            (["<s>"], 0)
        ]
        while len(complete_beams) < beam_size and len(beam) > 0:
            new_beam = []
            for cand_sent, score in beam:
                last_token = cand_sent[-1]
                current_length = len(cand_sent) - 1
                if current_length >= max_len:
                    current_mask = [("</s>", 0.)]
                elif current_length >= min_len:
                    current_mask = mask + [("</s>", 0.)]
                else:
                    current_mask = mask
                top_toks_with_score = self.get_top_tokens(token=last_token, mask=current_mask)
                # print(cand_sent, top_toks_with_score)
                for t, s in top_toks_with_score:
                    new_score = (score * len(cand_sent) + s) / (len(cand_sent) + 1)
                    new_beam.append(
                        (cand_sent + [t], new_score)
                    )
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)
            beam = []
            for cand_sent, score in new_beam:
                if cand_sent[-1] == "</s>":
                    complete_beams.append((cand_sent, score))
                else:
                    beam.append((cand_sent, score))
                if len(beam) == beam_size:
                    break
        complete_beams = sorted(complete_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        final_sequences = [(cand_sent[1:-1], score) for cand_sent, score in complete_beams]
        return final_sequences
        pass


if __name__ == '__main__':
    model = BiGramModel(
        train_data="/home/saikatc/HDD_4TB/from_server/StackOverFlow-Pandas/dpr_exp/data/ngram/train.txt")
    # print(json.dumps(model.conditional_prob, indent=4))
    beams = model.beam_search(
        mask=[("groupby", 0.85),
              ("cummax", 0.75)],
        beam_size=5,
        min_len=2,
        max_len=5
    )
    print(beams)
    pass
