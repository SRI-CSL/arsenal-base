#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime

import datasets
import torch
from tabulate import tabulate
from tqdm import tqdm
from transformers import EncoderDecoderModel, BertTokenizerFast

from arsenal_tokenizer import PreTrainedArsenalTokenizer
from utils import get_latest_trainingrun, get_latest_checkpoint

torch.set_printoptions(threshold=5000)

# Not sure where this comes from (the exact same code worked before), but without below line the code now produces
# the following error:
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
#   That is dangerous, since it can degrade performance or cause incorrect results. The best thing
#   to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding
#   static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented
#   workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to
#   continue to execute, but that may cause crashes or silently produce incorrect results. For more
#   information, please see http://www.intel.com/software/products/support/.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Relation Extraction")
parser.add_argument("-data_dir",       type=str,       default="../arsenal/large_files/datasets/2021-04-14T1922",
                                                                                        help="location of the data directory")
parser.add_argument("-epochs",         type=int,       default=30,                      help="number of training epochs")
parser.add_argument("-fp16",           type=str2bool,  default=True,                    help="Mixed precision training, can only be used on CUDA devices")
parser.add_argument("-results_dir",    type=str,       default="../arsenal/large_files/models/transformers/",
                                                                                        help="root location of all training results")
parser.add_argument("-run_subdir",     type=str,       default=None,                    help="subdirectory in results for specific training run (if none is provided, latest run from results dir is used)")
parser.add_argument("-model_name",     type=str,       default="translation_model",     help="subdirectory in run dir for translation model")
parser.add_argument("-num_beams",      type=int,       default=5,                       help="number of beams to use in beam search when generating predictions")
parser.add_argument("-num_outputs",    type=int,       default=5,                       help="number of beam search outputs to generate for each instance")

args = parser.parse_args()
data_dir = args.data_dir
results_dir = args.results_dir
run_id = args.run_subdir
num_beams = args.num_beams
num_outputs = args.num_outputs

if run_id is None:
    run_id = get_latest_trainingrun(os.path.join(results_dir))

# run_id = "04-29-2021"

print(f"training run: {run_id}")
checkpoint_id = get_latest_checkpoint(os.path.join(results_dir, run_id, args.model_name))
print(f"checkpoint: {checkpoint_id}")
model_dir = os.path.join(results_dir, run_id, args.model_name, checkpoint_id)
print(f"using model from {model_dir} to generate predictions")

dataset_properties = json.load(open(os.path.join(data_dir, "dataset_properties.json")))
special_tokens = dataset_properties["special_tokens"]
target_vocab = dataset_properties["target_vocab"]
max_length = dataset_properties["encoder_max_len"]

source_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

bert2arsenal = EncoderDecoderModel.from_pretrained(model_dir)

val_data = datasets.load_from_disk(os.path.join(data_dir, "arsenal_val"))

outfile = open(os.path.join(results_dir, run_id, f"typeforce_eval_{run_id}_{checkpoint_id}.txt"), "w")

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert2arsenal.to(torch_device)

batch_size = 10
num_batches = int(val_data.num_rows / batch_size)

num_beams=1
num_outputs=1

# # pp vocab for manual inspection
# ids = list(target_tokenizer.id2vocab.keys())
# ids.sort()
# for id in ids:
#     word = target_tokenizer.id2vocab[id]
#     info = word.split("#")
#     info = "\t".join(info)
#     print(f"{id}\t{info}")


inst = 0
for i in tqdm(range(num_batches)):


    batch_range = range(i*batch_size, (i+1)*batch_size)
    batch = val_data.select(list(batch_range))


    batch_ids = torch.tensor(batch["input_ids"], device=torch_device)
    batch_masks = torch.tensor(batch["attention_mask"], device=torch_device)

    tmp_out1 = bert2arsenal.generate(batch_ids, attention_mask=batch_masks,
                                    decoder_start_token_id=target_tokenizer.cls_token_id, num_beams=num_beams,
                                    num_return_sequences=num_outputs,
                                    type_forcing_vocab=target_tokenizer.id2vocab, no_repeat_ngram_size=0)

    tmp_out2 = bert2arsenal.generate(batch_ids, attention_mask=batch_masks,
                                                 decoder_start_token_id=target_tokenizer.cls_token_id, num_beams=num_beams,
                                                 num_return_sequences=num_outputs, no_repeat_ngram_size=0)

    # expand with padding zeros s.t. both results have the same dimension and can be compared
    max_dim_1 = max(tmp_out1.shape[1], tmp_out2.shape[1])
    out1 = torch.zeros(batch_size, max_dim_1, device=torch_device)
    out2 = torch.zeros(batch_size, max_dim_1, device=torch_device)
    out1[:,:tmp_out1.shape[1]] = tmp_out1
    out2[:, :tmp_out2.shape[1]] = tmp_out2


    for j in range(batch_size):


        i1 = [int(t) for t in out1[j].tolist() if t != 0]
        i2 = [int(t) for t in out2[j].tolist() if t != 0]

        true_seq = [t for t in batch['labels'][j] if t != -100]

        if not i1 == true_seq and i1 == i2:

            diff_pos_to_truth = 0
            for p in range(min(len(i1), len(true_seq))):
                if i1[p] != true_seq[p]:
                    diff_pos_to_truth = p
                    break

            outfile.write(f"\n\n################# instance {inst}: wrong, but type-correct #################\n")
            outfile.write(f"correct sequence:       {true_seq}\n")
            outfile.write(f"original prediction:    {i1}\n")
            outfile.write(f"- first difference to true sequence at    {diff_pos_to_truth}: changed from {true_seq[diff_pos_to_truth]} to {i1[diff_pos_to_truth]} \n")

            rel_ids = [true_seq[diff_pos_to_truth], i1[diff_pos_to_truth]]
            rel_tokens = target_tokenizer.decode(rel_ids)
            rel_tokens = rel_tokens.split()

            rows = ["true token", "predicted token"]
            table = []
            for i in range(2):
                parts = rel_tokens[i].split("#")
                table.append([rows[i], rel_ids[i]] + parts[:2])

            outfile.write(tabulate(table, headers=["relevant tokens", "id", "token", "type"]))

        elif i1 != i2:

            diff_pos_to_truth = 0
            for p in range(min(len(i1), len(true_seq))):
                if i1[p] != true_seq[p]:
                    diff_pos_to_truth = p
                    break

            diff_pos_between_preds = 0
            for p in range(len(i1)):
                if i1[p] != i2[p]:
                    diff_pos_between_preds = p
                    break

            outfile.write(f"\n\n################# instance  {inst}: wrong type #################\n")
            outfile.write(f"correct sequence:       {true_seq}\n")
            outfile.write(f"original prediction:    {i1}\n")
            outfile.write(f"type-forced prediction: {i2}\n")
            outfile.write(f"- first difference to true sequence at    {diff_pos_to_truth}: changed from {true_seq[diff_pos_to_truth]} to {i1[diff_pos_to_truth]} \n")
            outfile.write(f"- first difference between predictions at {diff_pos_between_preds}: changed from {i1[diff_pos_between_preds]} to {i2[diff_pos_between_preds]} (first error: {diff_pos_to_truth == diff_pos_between_preds}, correct: {true_seq == i2})\n")

            if len(i1) > diff_pos_between_preds and len(i2) > diff_pos_between_preds  and len(true_seq) > diff_pos_between_preds:
                rel_ids = [i1[diff_pos_between_preds - 1], true_seq[diff_pos_between_preds], i1[diff_pos_between_preds], i2[diff_pos_between_preds]]
                rel_tokens = target_tokenizer.decode(rel_ids)
                rel_tokens = rel_tokens.split()

                rows = ["previous token", "true token", "predicted token", "type-corrected token"]
                table = []
                for i in range(4):
                    parts = rel_tokens[i].split("#",3)
                    table.append([rows[i], rel_ids[i]] + parts[:3])

                outfile.write(tabulate(table, headers=["relevant tokens", "id", "token", "type", "first arg type"]))

            outfile.flush()

        inst += 1
outfile.close()

