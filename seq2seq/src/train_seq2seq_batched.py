import os
import torch
from ars2seq2seq.models.multi_attn_seq2seq import MultiHeadSeq2Seq
from ars2seq2seq.models.batched_seq2seq import BatchedSeq2Seq
from ars2seq2seq.models.batched_seq2seq_v2 import BatchedSeq2Seq_v2
from ars2seq2seq.util.dataset import read_pairfile, PairDataset
from ars2seq2seq.util.vocab import Lang

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data_root', required=True)
parser.add_argument('-output_name', required=True)
parser.add_argument('-input_lang', default='eng')
parser.add_argument('-output_lang', default='cst')
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-debug', default=None)
parser.add_argument('-max_length', default=1000, type=int)
parser.add_argument('-batch_size', default=10, type=int)

parser.add_argument('-eval_every', default=1000, type=int)
parser.add_argument('-save_every', default=10000, type=int)
parser.add_argument('-sample_every', default=100, type=int)

args = parser.parse_args()

print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_HIDDEN=128
NUM_LAYERS=1
INITIAL_LEARNING_RATE = args.learning_rate
DEBUG=args.debug

BATCH_SIZE=args.batch_size

reverse=False
root_dir = "data/"
OUTPUT_ROOT = os.path.join("output/batched", args.output_name)

if DEBUG:
    OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "debug")
else:
    OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "main")

NORM_ENTITIES=False

model_eng_vocab_fpath = os.path.join(OUTPUT_ROOT, "{}.vocab".format(args.input_lang))
model_cst_vocab_fpath = os.path.join(OUTPUT_ROOT, "{}.vocab".format(args.output_lang))

if os.path.isfile(model_eng_vocab_fpath):
    print("Loading existing vocab from {}".format(model_eng_vocab_fpath))
    eng_lang = Lang(args.input_lang, load_path=model_eng_vocab_fpath)
else:
    eng_lang = None

if os.path.isfile(model_cst_vocab_fpath):
    print("Loading existing vocab from {}".format(model_cst_vocab_fpath))
    cst_lang = Lang(args.output_lang, load_path=model_cst_vocab_fpath)
else:
    cst_lang = None

TRAIN_FILE = "{}-{}.train.txt".format(args.input_lang, args.output_lang)
VAL_FILE = "{}-{}.val.txt".format(args.input_lang, args.output_lang)
if DEBUG:
    TRAIN_FILE = "{}-{}.train.debug.txt".format(args.input_lang, args.output_lang)
    VAL_FILE = "{}-{}.val.debug.txt".format(args.input_lang, args.output_lang)

eng_lang, cst_lang, train_pairs = read_pairfile(os.path.join(root_dir, TRAIN_FILE), "eng", "cst",
                                                root_dir,
                                                max_length=200,
                                                lang1=eng_lang, lang2=cst_lang,
                                                normalize_sal_entities=NORM_ENTITIES,
                                                reverse=reverse)
eng_lang, cst_lang, val_pairs = read_pairfile(os.path.join(root_dir, VAL_FILE), "eng", "cst",
                                              root_dir,
                                              lang1=eng_lang, lang2=cst_lang,
                                              normalize_sal_entities=NORM_ENTITIES,
                                              reverse=reverse)
# Remove validation pairs alreadyin training pairs
#print("Filtering")
#val_pairs = [pair for pair in val_pairs if pair not in train_pairs]
#print("# filtered val={}".format(len(val_pairs)))

val_pairs = val_pairs[0:args.batch_size] # Filter down

train_pair_dataset = PairDataset(eng_lang, cst_lang, train_pairs, device=device)
val_pair_dataset = PairDataset(eng_lang, cst_lang, val_pairs, device=device)
print("Train dataset, prevectorizing")
train_pair_dataset.preload_pairs()
print("Val dataset, prevectorizing")
val_pair_dataset.preload_pairs()

print("Selected device={}".format(device))
# seq2seq = MultiHeadSeq2Seq(NUM_HIDDEN, eng_lang, cst_lang,
#                   device=device,
#                   initial_learning_rate=INITIAL_LEARNING_RATE,
#                   output_root=OUTPUT_ROOT,
#                   num_layers=NUM_LAYERS)
seq2seq = BatchedSeq2Seq_v2(NUM_HIDDEN, eng_lang, cst_lang,
                         batch_size=BATCH_SIZE,
                  device=device,
                  initial_learning_rate=INITIAL_LEARNING_RATE,
                  output_root=OUTPUT_ROOT,
                  num_layers=NUM_LAYERS)
print("\nModel instantiated, details=\n{}\n".format(seq2seq))

PROFILING = False

if PROFILING:
    import cProfile
    cProfile.run('seq2seq.train(1000000, train_pair_dataset, val_pair_dataset, profile=True)')
else:
    seq2seq.train(1000000, train_pair_dataset, val_pair_dataset,
                  eval_every=args.eval_every, sample_every=args.sample_every,
                  save_every=args.save_every)
