#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
import datasets
import torch
from tabulate import tabulate
from tqdm import tqdm
from transformers import EncoderDecoderModel, BertTokenizerFast
from transformers.trainer_utils import get_last_checkpoint

from args import parse_arguments
from arsenal_tokenizer import PreTrainedArsenalTokenizer

torch.set_printoptions(threshold=5000)

# def str2bool(v):
#     if isinstance(v, bool):
#        return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
#
# parser = argparse.ArgumentParser(description="Relation Extraction")
# parser.add_argument("-data_dir",       type=str,       default="../arsenal/large_files/datasets/2021-07-30T0004", help="location of the data directory")
# parser.add_argument("-fp16",           type=str2bool,  default=True,                    help="Mixed precision training, can only be used on CUDA devices")
# parser.add_argument("-model_dir",      type=str,       default="../arsenal/large_files/models/transformers/04-29-2021/translation_model",  help="location of the trained translation model")
# parser.add_argument("-num_beams",      type=int,       default=1,                       help="number of beams to use in beam search when generating predictions")
# parser.add_argument("-num_outputs",    type=int,       default=1,                       help="number of beam search outputs to generate for each instance")
#
# args = parser.parse_args()
# data_dir = args.data_dir
# model_dir = args.model_dir
# num_beams = args.num_beams
# num_outputs = args.num_outputs

def generate_predictions(args):

    model_dir = os.path.join(args.model_root_dir, args.run_id, args.translation_model_name)
    print(f"model dir: {model_dir}")
    val_data_path = os.path.join(args.data_out_dir, args.val_dataset_name)
    print(f"using model from {model_dir} and test data from {val_data_path} to generate predictions")

    dataset_properties = json.load(open(os.path.join(model_dir, "dataset_properties.json")))
    special_tokens = dataset_properties["special_tokens"]
    target_vocab = dataset_properties["target_vocab"]

    source_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

    bert2arsenal = EncoderDecoderModel.from_pretrained(model_dir)

    val_data = datasets.load_from_disk(val_data_path)

    runid, _, checkpoint = get_last_checkpoint((model_dir)).split("/")[-3:]
    outfile = open(os.path.join(Path(model_dir).parent, f"predictions_{runid}_{checkpoint}.txt"), "w")

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert2arsenal.to(torch_device)

    batch_size = args.batch_size
    num_batches = int(val_data.num_rows / batch_size)

    type_forcing_vocab = target_tokenizer.id2vocab if args.type_forcing else None

    for i in tqdm(range(num_batches)):

        batch_range = range(i*batch_size, (i+1)*batch_size)
        batch = val_data.select(list(batch_range))

        batch_ids = torch.tensor(batch["input_ids"], device=torch_device)
        batch_masks = torch.tensor(batch["attention_mask"], device=torch_device)

        # take this little detour with the args for generate() so that we can decide whether
        # we want to add the argument for the type forcing vocab (if using an unpatched
        # transformers version, adding anything about typeforcing (even if disabled) would
        # cause errors about unrecognized arguments)
        generate_args = {
            "input_ids": batch_ids,
            "attention_mask": batch_masks,
            "decoder_start_token_id": target_tokenizer.cls_token_id,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_outputs,
            "no_repeat_ngram_size": 0
        }
        if args.type_forcing:
            generate_args["type_forcing_vocab"] = type_forcing_vocab

        outputs = bert2arsenal.generate(**generate_args)

        # apparently batch instances and return sequences per instance are stacked along a single dimension
        for j in range(batch_size):
            input = [t for t in batch["input_ids"][j] if t != 0]
            true_seq = [t for t in batch['labels'][j] if t != -100]
            outfile.write(f"{input}\t{true_seq}")
            for k in range(j*args.num_outputs, (j+1)*args.num_outputs):
                pred_seq = [t for t in outputs[k].tolist() if t != 0]
                outfile.write(f"\t{pred_seq}")
            outfile.write("\n")
        outfile.flush()
    outfile.close()
#

if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    print(tabulate(vars(args).items(), headers={"parameter", "value"}))
    generate_predictions(args)


