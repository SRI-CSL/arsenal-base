import os
import torch
from ars2seq2seq.models.multi_attn_seq2seq import MultiHeadSeq2Seq
from ars2seq2seq.models.batched_seq2seq import BatchedSeq2Seq
from ars2seq2seq.util.dataset import read_pairfile
from ars2seq2seq.util.vocab import Lang
from nvidia_utils import get_freer_gpu

"""
Trains up a seq2seq model, using a single instance run.  

The source data directory is specified by -data_root.  The files should be formatted as,

   $SRC-$TGT.train.txt
   $SRC-$TGT.val.txt
   
where $SRC and $TGT are the source and target languages.  By default these are 'eng' and 'cst'.

The script also offers a debug option, which is set by adding -debug.  This allows the learning
to target an optional set of (presumably smaller) training files.  These should be under the
-data_root, and are specified as,

   $SRC-$TGT.train.debug.txt
   $SRC-$TGT.val.debug.txt
   
Debug runs also use a limited number of iterations.

"""

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-data_root', required=True, help='Source directory for training data (see README or comments for format)')
parser.add_argument('-output_name', required=True, help='Name for this experiment (used as prefix for adjunct files)')
parser.add_argument('-input_lang', default='eng', help='String name for input language')
parser.add_argument('-output_lang', default='cst', help='String name for target language')
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-debug', default=False, action='store_true', help='Activate debug mode (default off)')
parser.add_argument('-iters', default=1000000, type=int, help='Number of training iterations')
parser.add_argument('-debug_iters', default=200, type=int, help='Number of training iterations, when debug is active')
parser.add_argument('-max_length', default=1000, type=int, help='Maximum number of tokens to consider (input and output)')
parser.add_argument('-eval_every', default=1000, type=int, help='Evaluate train and val pairs at N iterations')
parser.add_argument('-save_every', default=10000, type=int, help='Save model, vocab, and setup to checkpoint every N iterations')
parser.add_argument('-sample_every', default=100, type=int, help='Sample every N iterations to the Tensorboard compatible log')
parser.add_argument('-num_layers', default=2, type=int, help='Number of layers')
parser.add_argument('-num_hidden', default=128, type=int, help='Number of hidden nodes per layer')
parser.add_argument('-reorder_numbered_placeholders', default="True", help="Reorder numbered placeholders ('_\d+$') to orthographic order")
parser.add_argument('-match_parens', default=False, action='store_true', help="Match parentheses")
parser.add_argument('-init_type', default='', help="Ensure typability, producing CST of a given type")
parser.add_argument('-dropout', default=0.1, type=float, help="Dropout (0 to 1)")
args = parser.parse_args()

print("Training with arguments:")
print(args)

NUM_HIDDEN=args.num_hidden
NUM_LAYERS=args.num_layers
DEBUG=args.debug
INITIAL_LEARNING_RATE = args.learning_rate

reverse=False
root_dir = args.data_root
OUTPUT_ROOT = os.path.join("output/single", args.output_name)
if DEBUG:
    OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "debug")
else:
    OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "main")

print("Saving model output to {}".format(OUTPUT_ROOT))


REORDER_NUMBERED_PLACEHOLDERS = args.reorder_numbered_placeholders.lower().strip() == "true"
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

eng_lang, cst_lang, train_pairs = read_pairfile(os.path.join(root_dir, TRAIN_FILE),
                                                args.input_lang, args.output_lang,
                                                root_dir,
                                                lang1=eng_lang, lang2=cst_lang,
                                                normalize_sal_entities=NORM_ENTITIES,
                                                reorder_numplaceholders=REORDER_NUMBERED_PLACEHOLDERS,
                                                reverse=reverse,
                                                match_parens=args.match_parens)

eng_lang, cst_lang, val_pairs = read_pairfile(os.path.join(root_dir, VAL_FILE),
                                              args.input_lang, args.output_lang,
                                              root_dir,
                                              lang1=eng_lang, lang2=cst_lang,
                                              normalize_sal_entities=NORM_ENTITIES,
                                              reorder_numplaceholders=REORDER_NUMBERED_PLACEHOLDERS,
                                              reverse=reverse,
                                              match_parens=args.match_parens)

device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")
print("Selected device={}".format(device))

seq2seq = MultiHeadSeq2Seq(NUM_HIDDEN, eng_lang, cst_lang,
                   device=device,
                   initial_learning_rate=INITIAL_LEARNING_RATE,
                   output_root=OUTPUT_ROOT,
                   num_layers=NUM_LAYERS,
                   max_length=args.max_length,
                   match_parens=args.match_parens,
                   init_type=args.init_type,
                   dropout=args.dropout)
print("\nModel instantiated, details=\n{}\n".format(seq2seq))
if DEBUG:
    seq2seq.train(args.debug_iters, train_pairs, val_pairs,
                  eval_every=args.eval_every,
                  sample_every=args.sample_every,
                  save_every=args.save_every)
else:
    seq2seq.train(args.iters, train_pairs, val_pairs,
                  eval_every=args.eval_every,
                  sample_every=args.sample_every,
                  save_every=args.save_every)
