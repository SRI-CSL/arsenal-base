#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
import datasets
import torch
from tqdm import tqdm
from transformers import EncoderDecoderModel, BertTokenizerFast
from transformers.trainer_utils import get_last_checkpoint

from arsenal_tokenizer import PreTrainedArsenalTokenizer

torch.set_printoptions(threshold=5000)

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
parser.add_argument("-data_dir",       type=str,       default="../arsenal/large_files/datasets/2021-07-30T0004", help="location of the data directory")
parser.add_argument("-fp16",           type=str2bool,  default=True,                    help="Mixed precision training, can only be used on CUDA devices")
parser.add_argument("-model_dir",      type=str,       default="../arsenal/large_files/models/transformers/04-29-2021/translation_model",  help="location of the trained translation model")
parser.add_argument("-num_beams",      type=int,       default=1,                       help="number of beams to use in beam search when generating predictions")
parser.add_argument("-num_outputs",    type=int,       default=1,                       help="number of beam search outputs to generate for each instance")

args = parser.parse_args()
data_dir = args.data_dir
model_dir = args.model_dir
num_beams = args.num_beams
num_outputs = args.num_outputs

print(f"using model from {get_last_checkpoint(model_dir)} and test data from {data_dir} to generate predictions")

dataset_properties = json.load(open(os.path.join(model_dir, "dataset_properties.json")))
special_tokens = dataset_properties["special_tokens"]
target_vocab = dataset_properties["target_vocab"]
max_length = dataset_properties["encoder_max_len"]

source_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

bert2arsenal = EncoderDecoderModel.from_pretrained(get_last_checkpoint(model_dir))

val_data = datasets.load_from_disk(os.path.join(data_dir, "arsenal_val"))

runid, _, checkpoint = get_last_checkpoint((model_dir)).split("/")[-3:]
outfile = open(os.path.join(Path(model_dir).parent, f"predictions_{runid}_{checkpoint}.txt"), "w")

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert2arsenal.to(torch_device)

batch_size = 10
num_batches = int(val_data.num_rows / batch_size)

for i in tqdm(range(num_batches)):

    batch_range = range(i*batch_size, (i+1)*batch_size)
    batch = val_data.select(list(batch_range))

    batch_ids = torch.tensor(batch["input_ids"], device=torch_device)
    batch_masks = torch.tensor(batch["attention_mask"], device=torch_device)
    outputs = bert2arsenal.generate(batch_ids, attention_mask=batch_masks,
                                    decoder_start_token_id=target_tokenizer.cls_token_id, num_beams=num_beams, num_return_sequences=num_outputs,
                                    # type_forcing_vocab=None,
                                    no_repeat_ngram_size=0)

    # apparently batch instances and return sequences per instance are stacked along a single dimension
    for j in range(batch_size):
        input = [t for t in batch["input_ids"][j] if t != 0]
        true_seq = [t for t in batch['labels'][j] if t != -100]
        outfile.write(f"{input}\t{true_seq}")
        for k in range(j*num_outputs, (j+1)*num_outputs):
            pred_seq = [t for t in outputs[k].tolist() if t != 0]
            outfile.write(f"\t{pred_seq}")
        outfile.write("\n")
    outfile.flush()
outfile.close()

