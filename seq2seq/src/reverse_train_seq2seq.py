import os
import torch
from ars2seq2seq.models.multi_attn_seq2seq import MultiHeadSeq2Seq
from ars2seq2seq.util.dataset import read_pairfile

NUM_HIDDEN=128
NUM_LAYERS=2
INITIAL_LEARNING_RATE = 0.001


reverse=True
root_dir = "combined_data"
OUTPUT_ROOT = "output/reversed"
NORM_ENTITIES=False

cst_lang, eng_lang, train_pairs = read_pairfile(os.path.join(root_dir, "eng-cst.train.txt"), "eng", "cst",
                                                root_dir,
                                                normalize_sal_entities=NORM_ENTITIES,
                                                reverse=reverse)
cst_lang, eng_lang, val_pairs = read_pairfile(os.path.join(root_dir, "eng-cst.val.txt"), "eng", "cst",
                                              root_dir,
                                              lang1=eng_lang, lang2=cst_lang,
                                              normalize_sal_entities=NORM_ENTITIES,
                                              reverse=reverse)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Selected device={}".format(device))
seq2seq = MultiHeadSeq2Seq(NUM_HIDDEN, cst_lang, eng_lang,
                  device=device,
                  initial_learning_rate=INITIAL_LEARNING_RATE,
                  output_root=OUTPUT_ROOT,
                  num_layers=NUM_LAYERS)
print("Model instantiated, name={}".format(seq2seq.exp_name()))
seq2seq.train(100000, train_pairs, val_pairs)
