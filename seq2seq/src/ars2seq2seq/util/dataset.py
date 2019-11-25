"""
Updated version of vocab.py, and is intended to replace it.
"""
import os
import re
import unicodedata
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from ars2seq2seq.util.vocab import process_sentence, Lang, unicodeToAscii, normalize_string, length_filter_pair
from ars2seq2seq.util.vocab import load_vocab, save_vocab

from ars2seq2seq.util.entities import normalize_sal_entities, reorder_numbered_placeholders
from ars2seq2seq.util.vocab import UNK_token, SOS_token, EOS_token, PAD_token

def _vocab_fname(lang_name, root_dir):
    return os.path.join(root_dir, "{}.vocab".format(lang_name))


def load_vocab(lang_name, root_dir):
    return Lang(lang_name, load_path=_vocab_fname(lang_name, root_dir))

def read_pairfile(pair_fname,
                  lang1_name,
                  lang2_name,
                  root_dir,
                  max_length=None,
                  reverse=False,
                  filter_fn=None,
                  lang1 = None,
                  lang2 = None,
                  reorder_numplaceholders=False,
                  normalize_sal_entities=False,
                  force_saveout_vocab=False,
                  match_parens=False):
    print("Reading paired datafile={}".format(pair_fname))
    print("Reordering options (if any): ")
    if reorder_numplaceholders:
        print("\tNumbered placeholders")
    print("Normalization (if any):")
    if normalize_sal_entities:
        print("\tEntities")
    # Read the file and split into lines
    lines = open(pair_fname, encoding='utf-8').\
        read().strip().split('\n')

    print("Normalizing, total lines={}".format(len(lines)))
    with ThreadPool(12) as p:
        def inner_fn(l):
            normed_pairs = [normalize_string(s) for s in l.split('\t')]
            pair_str1, pair_str2 = normed_pairs[0], normed_pairs[1]
            if normalize_sal_entities:
                pair_str1, pair_str2, _ = normalize_sal_entities(pair_str1, pair_str2)
            if reorder_numplaceholders:
                pair_str1, pair_str2, _ = reorder_numbered_placeholders(pair_str1, pair_str2)
            return [pair_str1, pair_str2]
        pairs = list(tqdm(p.imap_unordered(inner_fn, lines)))

    if filter_fn:
        # Inefficient, but need to guarantee that inner_fn won't return
        # None deliberately.
        pairs = [pair for pair in pairs if filter_fn(pair)]

    if max_length is not None:
        pairs = [pair for pair in pairs if len(pair[0].split()) <= max_length and len(pair[1].split()) <= max_length]

    if lang1 is None:
        lang1 = Lang(lang1_name, load_path=_vocab_fname(lang1_name, root_dir))
    if lang2 is None:
        lang2 = Lang(lang2_name, load_path=_vocab_fname(lang2_name, root_dir))

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = lang2
        output_lang = lang1
    else:
        input_lang = lang1
        output_lang = lang2

    input_lang_empty = len(input_lang) == input_lang.BASELINE_N
    output_lang_empty = len(output_lang) == output_lang.BASELINE_N

    print("Generating vocab for languages")
    for pair in pairs:
        input_lang.fit_sentence(pair[0])
        output_lang.fit_sentence(pair[1])
    if force_saveout_vocab:
        print("Saving vocab out")
        input_lang.save()
        output_lang.save()
    else:
        if input_lang_empty:
            print("Input lang originally empty, saving out")
            input_lang.save()
        if output_lang_empty:
            print("Output lang originally empty, saving out")
            output_lang.save()

    # Save out sample
    with open(pair_fname+"sample.txt", "w") as f:
        for i in range(0, min(len(pairs), 100)):
            pair = pairs[i]
            f.write(pair[0])
            f.write("\t")
            f.write(pair[1])
            f.write("\n")

    return input_lang, output_lang, pairs



def indexes_from_sentence(lang, sentence):
    indices = [lang.word2index.get(word, UNK_token) for word in process_sentence(sentence)] 
    if match_parens:
        return indices
    else:
        return indices + [EOS_token]
    

def indices_from_pair(pair, input_lang, output_lang):
    p1 = indexes_from_sentence(input_lang, pair[0])
    p2 = indexes_from_sentence(output_lang, pair[1])
    return (p1, p2)


def pad_seq(seq, max_length, pad_idx=0):
    seq += [pad_idx for i in range(max_length - len(seq))]
    return seq

# TODO: Need to figure out how to set DataLoader to generate a batch in length
# descending order

def custom_collate_fn(data_pairs):
    return data_pairs


class PairDataset(Dataset):
    def __init__(self, input_lang, output_lang, pairs, device="cpu"):
        self.input_lang, self.output_lang = input_lang, output_lang
        self.pairs = pairs
        self.device = device
        self.indices_lookup = {}

    def _pair2key(self, pair):
        return "---".join(pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        pair_key = self._pair2key(self.pairs[item])
        if pair_key not in self.indices_lookup:
            self.indices_lookup[pair_key] = indices_from_pair(self.pairs[item], self.input_lang, self.output_lang)
        vec_pair = self.indices_lookup[pair_key]
        return vec_pair

    def preload_pairs(self):
        print("Vectorizing, total pairs={}".format(len(self.pairs)))
        with ThreadPool(12) as p:
            def inner_fn(pair):
                # Have to generate these in CPU, use getitem to load to device
                input_idxes, target_idxes = indices_from_pair(pair, self.input_lang, self.output_lang)
                return [self._pair2key(pair), input_idxes, target_idxes]
            idx_pair_tuples = list(tqdm(p.imap_unordered(inner_fn, self.pairs)))
            print("Caching")
            for t in idx_pair_tuples:
                self.indices_lookup[t[0]] = t[1:]


def form_batch_from_pairs(paired_idxes, device="cpu"):
    # Sort pairs first by input length.  Packed RNN representation requires sequences be given
    # in descending order (striding policy probably fails otherwise).
    # This is probably part of the single array packing strategy that permits fast RNN inference over
    # batches of sequences.
    #
    # TODO: Issue, padding is variable amount, dependent upon the batch size.  May have to vectorize to
    # numeric sequences firs,t then do the conversion to Tensor here
    sorted_pairs = sorted(paired_idxes, key=lambda x: len(x[0]), reverse=True)  # Length decreasing input sentences
    input_lengths = [len(pair[0]) for pair in sorted_pairs]
    target_lengths = [len(pair[1]) for pair in sorted_pairs]
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)
    padded_pairs = [(pad_seq(pair[0], max_input_len), pad_seq(pair[1], max_target_len)) for pair in sorted_pairs]
    input_X = Variable(torch.LongTensor([p[0] for p in padded_pairs])).transpose(0, 1).to(device)
    target_X = Variable(torch.LongTensor([p[1] for p in padded_pairs])).transpose(0, 1).to(device)
    return input_X, input_lengths, target_X, target_lengths

