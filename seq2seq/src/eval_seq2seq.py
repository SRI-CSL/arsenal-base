import os
import torch
from ars2seq2seq.models.multi_attn_seq2seq import MultiHeadSeq2Seq
from ars2seq2seq.http.cst_generator import load_from_setup
from nvidia_utils import get_freer_gpu

"""
Reads in a text file (one per line) and prints out the translation.
If the line is tabs delimited, presumes the first portion is the language, 2nd is target, and will
attempt to judge accuracy.

If pretty is activated, both the original, target (if any), and generated are printed out.  If not, just the
target decodings are generated.
"""

device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('model_root', type=str, help='Path to checkpoint directory produced by seq2seq training')
parser.add_argument('input_file', type=str, help='Input sentences to convert to target, one per line')
parser.add_argument('-normalize_entities', type=str, default="false", help="Set to True to use Microwave/Isolette heuristics to normalize entities to ID__$DIGIT form")
parser.add_argument("-pretty", type=str, default="True", help="Pretty print results")
parser.add_argument("-convert_to_json", type=str, default="False", help="Converts results to JSON (if they aren't already)")
args = parser.parse_args()

convert_to_json = args.convert_to_json.lower() == "true"
cst_generator = load_from_setup(args.model_root, device=device, verbose=True, reorder_numbered_placeholders=True,
                                convert_to_json=convert_to_json)
pretty_print = args.pretty.lower().strip() == "true"

with open(args.input_file, 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    golds = None
    if len(lines[0].split('\t')) == 2:
        print("Golds present (tab delimited).  Will check accuracies")
        sentences = [sentence.split('\t')[0] for sentence in lines if len(sentence) > 0]
        golds = [sentence.split('\t')[1] for sentence in lines if len(sentence) > 0]
    else:
        sentences = [line for line in lines if len(line) > 0]
    generated_tgts = cst_generator.process_txt_sentences(sentences)
    idx = 0
    num_correct = 0
    for tgt_txt in generated_tgts:
        if pretty_print:
            if golds is None:
                print("\n--------------------------")
                print("# {}".format(idx))
                print("Src: {}".format(sentences[idx]))
                print("Out: {}".format(tgt_txt))
                idx += 1
            else:
                print("\n--------------------------")
                print("# {}".format(idx))
                print("Src: {}".format(sentences[idx]))
                print("Tgt: {}".format(golds[idx]))
                matched = golds[idx].lower() == tgt_txt.lower()
                matched_marker = ""
                if matched:
                    num_correct += 1
                else:
                    matched_marker = "* "
                print("{}Out: {}".format(matched_marker, tgt_txt))
                idx += 1
        else:
            print(tgt_txt)
    if not(golds is None):
        print("Acc = {:.5f}".format(num_correct / idx))

