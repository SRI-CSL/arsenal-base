#!/usr/bin/env python
import argparse
import json
import os
from datasets import Dataset, tqdm
import random
from transformers import BertTokenizerFast, GPT2TokenizerFast
from arsenal_tokenizer import PreTrainedArsenalTokenizer

# for dataset 04/14: max input length: 192, max output length: 122

parser = argparse.ArgumentParser(description="Dataset preprocessor")
parser.add_argument("-data_dir",        type=str, default="../arsenal/large_files/datasets/2021-04-14T1922",    help="location of the generated data")
parser.add_argument("-train_file",      type=str, default="eng-pn.train.txt",                                   help="name of training data file")
parser.add_argument("-val_file",        type=str, default="eng-pn.val.txt",                                     help="name of validation data file")
parser.add_argument("-out_dir",         type=str,                                                               help="location of the output data directory")
parser.add_argument("-max_source_len",  type=int, default=75,                                                   help="maximum number of words in the English sentences "
                                                                                                                "(all instances above this threshold will be discarded")
args = parser.parse_args()

data_dir = args.data_dir
out_dir = args.out_dir

if out_dir is None:
    out_dir = data_dir

max_word_len = args.max_source_len
print(f"discarding all instances with more than {max_word_len} words in the source sentences")

def process_input(filename):

    print(f"processing file {filename}...")
    with open(os.path.join(data_dir, filename), "r") as f:
        instances = f.read().splitlines()
        #instances = [instances[i] for i in range(0,len(instances),2)]

    dataset = {}
    source_vocab = []
    target_vocab = []
    special_tokens = [] # all words containing '_' are treated as special tokens and added to the tokenizer

    # read input file, create:
    # - dataset: dict of source/target sentences
    #   (the target entry is actually a list of sentences, b/c. the data set contains multiple identical source entries)
    # - special_tokens: list of special tokens (ARSENAL entity place holders)
    # - target_vocab: list of words used in the target language
    for inst in instances:
        [source, target] = inst.split("\t")
        source_words = source.split(" ")

        if len(source_words) < max_word_len:
            source_vocab.extend(source_words)

            for word in source_words:
                if "_" in word:
                    special_tokens.append(word)
            if source not in dataset:
                dataset[source] = [target]
            else:
                dataset[source].append(target)
            target_vocab.extend(target.split(" "))

    # clean up set of special tokens and target vocab
    special_tokens = [x.replace(",", "") for x in special_tokens]
    special_tokens = list(set(special_tokens))
    special_tokens.sort()
    source_vocab = [x.replace(",", "") for x in source_vocab]
    source_vocab = list(set(source_vocab))
    source_vocab.sort()
    target_vocab = list(set(target_vocab))
    target_vocab.sort()

    total_instances = 0
    unique_instances = 0
    non_unique_inputs = []

    ambig_set = dict()
    # remove redundant instances, collect some statistics
    for s, t in dataset.items():
        total_instances += len(t)
        t = list(set(t))
        if len(t) > 1:
            non_unique_inputs.append(s)
            ambig_set[s] = t
            # ambig_file.write(f"{s}\n")
            # for ti in t:
            #     ambig_file.write(f"\t{ti}\n")
        unique_instances += len(t)
        dataset[s] = t

    if len(non_unique_inputs) > 0:
        out_json = open(os.path.join(out_dir,filename.replace(".txt","") + "_ambiguities.json"), "w")
        json.dump(ambig_set, out_json)
        out_json.close()

    # reduce multi-valued outputs to their first instance
    # (To simplify the dataset for first experiments, this should probably be
    # handled differently later. Maybe we can simply omit this step?)
    # At least when using accuracy as a metric, allowing for non-unique
    # outputs seems troublesome.
    for s, t in dataset.items():
        dataset[s] = list(t)[0]

    print(f" -number of special tokens: {len(special_tokens)}")
    print(f" -size of output vocab: {len(target_vocab)}")
    print(f" -number of unique inputs: {len(dataset.keys())}")

    print(f" -total number of outputs: {total_instances}")
    print(f" -unique number of outputs: {unique_instances}")
    print(f" -non-unique number of inputs: {len(non_unique_inputs)}")

    return dataset, special_tokens, source_vocab, target_vocab

train_dataset, special_tokens, source_vocab, target_vocab = process_input(args.train_file)
val_dataset, val_special_tokens, val_source_vocab, val_target_vocab = process_input(args.val_file)

diffs = {}

diffs["source_vocab_train_only"] = [word for word in source_vocab if word not in val_source_vocab]
diffs["source_vocab_val_only"] = [word for word in val_source_vocab if word not in source_vocab]

diffs["target_vocab_train_only"] = [ word for word in target_vocab if word not in val_target_vocab]
diffs["target_vocab_val_only"] = [ word for word in val_target_vocab if word not in target_vocab]

diffs["special_tokens_train_only"] = [ token for token in special_tokens if token not in val_special_tokens]
diffs["special_tokens_val_only"] = [ token for token in val_special_tokens if token not in special_tokens]

print(f"sanity checks (should both be []):\n "
      f" -special tokens only in validation set: {diffs['special_tokens_val_only']},\n"
      f" -target words only in validation set: {diffs['target_vocab_val_only']}")

if diffs["special_tokens_train_only"] != []:
    print(f"the training set contains special tokens that don't appear in the validation set")
if diffs["target_vocab_train_only"] != []:
    print(f"the training set contains target words that don't appear in the validation set")

to_remove = []

for i in val_dataset:
    if i in train_dataset:
        to_remove.append(i)

for i in to_remove:
    val_dataset.pop(i)

if to_remove != []:
    print(f"found {len(to_remove)} overlapping instances in train and validation set, removed those from validation set. New size of validation set: {len(val_dataset)}")

source_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

# this is highly inefficient because all data is tokenized twice:
# - first to determine max token sequence length
# - again, to pad all inputs to max length
# one could optimize this by first padding all tokenized sequences to some large number,
# then check for the max payload length (e.g., max # of nonzero input ids) and then prune
# everything accordingly. But compared to the training time, the time spent tokenizing is
# fairly small, so this doesn't seem like a critical improvement.
determine_max_len = True
if determine_max_len:
    print("finding max token sequence length for input and output...")
    train_input_lengths = list(map(lambda x: len(source_tokenizer(x).input_ids), tqdm(train_dataset.keys())))
    val_input_lengths = list(map(lambda x: len(source_tokenizer(x).input_ids), tqdm(val_dataset.keys())))
    train_output_lengths = list(map(lambda x: len(target_tokenizer(x).input_ids), tqdm(train_dataset.values())))
    val_output_lengths = list(map(lambda x: len(target_tokenizer(x).input_ids), tqdm(val_dataset.values())))

    print(f"max train input: {max(train_input_lengths)}, max val input: {max(val_input_lengths)}")
    print(f"max train output: {max(train_output_lengths)}, max val output: {max(val_output_lengths)}")
    encoder_max_length = max(train_input_lengths+val_input_lengths)
    decoder_max_length = max(train_output_lengths+val_output_lengths)
    encoder_min_length = min(train_input_lengths + val_input_lengths)
    decoder_min_length = min(train_output_lengths + val_output_lengths)

    print(f"max input length: {encoder_max_length}, max output length: {decoder_max_length }")
    print(f"min input length: {encoder_min_length}, min output length: {decoder_min_length}")
else:
    encoder_max_length = 97
    decoder_max_length = 122
    encoder_min_length = 4
    decoder_min_length = 7

batch_size=4


# from https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing
def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = source_tokenizer(batch["input"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = target_tokenizer(batch["output"], padding="max_length", truncation=True, max_length=decoder_max_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == source_tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch

dataset_properties = {}

dataset_properties["max_source_word_len"] = max_word_len
dataset_properties["source_vocab"] = source_vocab
dataset_properties["target_vocab"] = target_vocab
dataset_properties["encoder_max_len"] = encoder_max_length
dataset_properties["decoder_max_len"] = decoder_max_length
dataset_properties["encoder_min_len"] = encoder_min_length
dataset_properties["decoder_min_len"] = decoder_min_length
dataset_properties["special_tokens"] = special_tokens
dataset_properties["train_val_differences"] = diffs


with open(os.path.join(out_dir, "dataset_properties.json"), "w") as f:
    json.dump(dataset_properties, f,  indent=3)

source_tokenizer.save_pretrained(os.path.join(out_dir, "source_tokenizer"))
target_tokenizer.save_pretrained((os.path.join(out_dir, "target_tokenizer")))

splits = {"train": train_dataset, "val": val_dataset}

for name, dataset in splits.items():
    print(f"tokeninzing {name} data (note that this is processed in batches, so the shown total number doesn't correspond to total instances)")
    ds = Dataset.from_dict({"input": dataset.keys(), "output": dataset.values()})
    ds = ds.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
    )
    ds.save_to_disk(os.path.join(out_dir, "arsenal_" + name))


###############
# create inputs for LM evaluation (based on GPT2 LM)

# the training set is basically the same as the input side of the seq2seq model, except that (for now)
# it is tokenized with a GPT-specific tokenizer
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

gpt_train_ds = Dataset.from_dict(gpt_tokenizer(list(train_dataset.keys()), padding="max_length", truncation=True, max_length=decoder_max_length))
gpt_train_ds.save_to_disk(os.path.join(out_dir, "lm_train"))








