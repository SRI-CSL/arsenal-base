"""
Runs and scores the validation paired file
"""

import os
import argparse
import json
import torch
from ars2seq2seq.util.vocab import load_lang
from ars2seq2seq.models.multi_attn_seq2seq import MultiHeadSeq2Seq
from ars2seq2seq.models.batched_seq2seq import BatchedSeq2Seq
from ars2seq2seq.util.dataset import read_pairfile
from ars2seq2seq.util.vocab import Lang

parser = argparse.ArgumentParser(description='')
parser.add_argument('-model_dir', required=True, help='Path to cached model to validate')
parser.add_argument('-val_file', required=True, help='Path to paired file to validate')
parser.add_argument('-results_file', default='validation-results.txt', help='Path to results file to write out to.')

cmdline_args = parser.parse_args()
device="cpu"

model_root = cmdline_args.model_dir

with open(os.path.join(model_root, "setup.json"), "r") as f:
    args = json.load(f)
    num_hidden=args['hidden_size']
    input_lang_name = args['input_lang']
    output_lang_name = args['output_lang']
    max_length=args['max_length']
    num_layers=args['num_layers']
    num_attn=args['num_attn']
    match_parens=args['match_parens'],
    input_vocab_fname = "{}.vocab".format(input_lang_name)
    output_vocab_fname = "{}.vocab".format(output_lang_name)
    vocab_dir = model_root
    input_lang = load_lang(input_lang_name, os.path.join(vocab_dir, input_vocab_fname))
    output_lang = load_lang(output_lang_name, os.path.join(vocab_dir, output_vocab_fname))
    seq2seq = MultiHeadSeq2Seq(num_hidden, input_lang, output_lang, device=device,
                               max_length=max_length, num_layers=num_layers, num_attn=num_attn, match_parens=match_parens)
    seq2seq.load_checkpoint(override_checkpoint_dir=model_root)

    _, _, val_pairs = read_pairfile(cmdline_args.val_file,
                                                  input_lang_name, output_lang_name,
                                                  None,
                                                  lang1=input_lang, lang2=output_lang,
                                                  reorder_numplaceholders=True)

    acc, res_str = seq2seq.evaluate(val_pairs, n=None, randomize=False)
    with open(cmdline_args.results_file, "w") as res_f:
        res_f.write(res_str)
    print("Acc={}".format(acc))
