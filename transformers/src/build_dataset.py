#!/usr/bin/env python
import argparse
import json
import os
import re
import sys

from datasets import Dataset, tqdm
from transformers import BertTokenizerFast, GPT2TokenizerFast

from args import parse_arguments
from arsenal_tokenizer import PreTrainedArsenalTokenizer

def process_input(data_dir, out_dir, filename, max_word_len, ignore_prefixes, ignore_suffixes, check_balance):

    print(f"processing file {filename}...")
    with open(os.path.join(data_dir, filename), "r") as f:
        instances = f.read().splitlines()

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

    # the prefix used for entity placeholders - use this to split compound tokens below
    grammar_prefix = re.match(".*?(_\w*)", special_tokens[0]).group(1)

    #  clean special tokens according to ignore special prefix/suffix chars, trailing periods, split on parentheses etc
    cleaned_special_tokens = []
    for t in special_tokens:

        t = t.replace(",", "")

        modified = True
        skip = False
        while(modified):
            if t == "":
                skip = True
                break

            modified = False
            # remove trailing periods
            if t.endswith("."):
                t = t[:-1]
                modified = True
                continue

            if check_balance:
                # remove balanced enclosing brackets and parentheses
                if t.startswith("(") and t.endswith(")"):
                    t = t[1:-1]
                    modified = True
                    continue
                if t.startswith("[") and t.endswith("]"):
                    t = t[1:-1]
                    modified = True
                    continue
            else:
                # remove all enclosing brackets and parentheses, even if unbalanced
                if t[0] in ["(", "["]:
                    t = t[1:]
                    modified = True
                    continue
                if t[-1] in [")", "]"]:
                    t = t[1:]
                    modified = True
                    continue

            # remove special ignore prefix/suffix characters
            if t[-1] in ignore_suffixes:
                t = t[:-1]
                modified = True
                continue
            if t[0] in ignore_prefixes:
                t = t[1:]
                modified = True
                continue

            # if we have parentheses and brackets inside of special tokens (e.g., for function applications)
            # split across opening parentheses and treat 'inner' and 'outer' part as separate tokens
            # (by appending them to the end of the queue and skipping the compound token)
            # - only split after all enclosing parentheses have been removed
            if not t.startswith("(") and "(" in t:
                outer, inner = t.split("(", 1)
                skip = True
                special_tokens.append(inner)
                special_tokens.append(outer)

            if not t.startswith("[") and "[" in t:
                outer, inner = t.split("[", 1)
                skip = True
                special_tokens.append(inner)
                special_tokens.append(outer)

            # we separate all the different entities in the same string
            if len(t.split(grammar_prefix)) > 2:
                compound_terms = t.split(grammar_prefix)[1:]
                compound_tokens = [grammar_prefix + term for term in compound_terms]
                special_tokens.extend(compound_tokens)
                skip = True

        if not skip:
            cleaned_special_tokens.append(t)

    special_tokens = list(set(cleaned_special_tokens))
    special_tokens.sort()
    source_vocab = [x.replace(",", "") for x in source_vocab]
    source_vocab = list(set(source_vocab))
    source_vocab.sort()
    target_vocab = list(set(target_vocab))
    target_vocab.sort()

    #  sanity check: look for special tokens that appear in the source, but not in the target
    # (can't directly check for equality, because special tokens in the source language start
    # with an underscore and have additional type information)
    both = []
    source_only = []
    for t1 in special_tokens:

        tmp = t1[1:] # strip away leading underscore
        found = False
        for t2 in target_vocab:
            if tmp in t2:
                both.append(t1)
                found = True
                break

        if not found:
            source_only.append(t1)

    if len(source_only) > 0:
        print(f"The source sentences contain {len(source_only)}/{len(special_tokens)} special tokens that don't appear in the target sentences:")
        for t in source_only:
            print(t)


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


def build_dataset(args):

    data_dir = args.data_dir
    out_dir = args.data_out_dir
    train_file = args.train_file
    val_file = args.val_file
    max_word_len = args.max_source_len
    batch_size = args.batch_size
    ignore_suffixes = args.strip_suffix_chars.split()
    ignore_prefixes = args.strip_prefix_chars.split()
    check_balance = args.check_balance


    print(f"discarding all instances with more than {max_word_len} words in the source sentences")

    train_dataset, special_tokens, source_vocab, target_vocab = process_input(data_dir, out_dir, train_file, max_word_len, ignore_prefixes, ignore_suffixes, check_balance)
    val_dataset, val_special_tokens, val_source_vocab, val_target_vocab = process_input(data_dir, out_dir, val_file, max_word_len, ignore_prefixes, ignore_suffixes, check_balance)

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

    source_tokenizer = BertTokenizerFast.from_pretrained(args.source_model)
    source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

    # source_tokenizer.save_pretrained(os.path.join(out_dir, "source_tokenizer"))
    # target_tokenizer.save_pretrained((os.path.join(out_dir, "target_tokenizer")))

    splits = {"train": train_dataset, "val": val_dataset}

    # this is highly inefficient because all data is tokenized twice:
    # - first to determine max token sequence length
    # - again, to pad all inputs to max length
    # one could optimize this by first padding all tokenized sequences to some large number,
    # then check for the max payload length (e.g., max # of nonzero input ids) and then prune
    # everything accordingly. But compared to the training time, the time spent tokenizing is
    # fairly small, so this doesn't seem like a critical improvement.
    # In principle, it should not need to be required to padd *all* instances to the same length,
    # but instead it should suffice to make sure that all instances within one batch have the same
    # length (and make sure that the batch size for the actual model training is the same as for
    # dataset generation). This can be achieved by setting padding="longest", but for some reason
    # the resulting batches don't work with training the translation model (but they do work for the
    # target LM training). Could be further investigated...
    determine_max_len = True
    if determine_max_len:
        print("finding max token sequence length for input and output...")
        train_input_lengths = list(map(lambda x: len(source_tokenizer(x).input_ids), tqdm(train_dataset.keys())))
        val_input_lengths = list(map(lambda x: len(source_tokenizer(x).input_ids), tqdm(val_dataset.keys())))
        train_output_lengths = list(map(lambda x: len(target_tokenizer(x).input_ids), tqdm(train_dataset.values())))
        val_output_lengths = list(map(lambda x: len(target_tokenizer(x).input_ids), tqdm(val_dataset.values())))

        print(f"max train input: {max(train_input_lengths)}, max val input: {max(val_input_lengths)}")
        print(f"max train output: {max(train_output_lengths)}, max val output: {max(val_output_lengths)}")
        encoder_max_length = max(train_input_lengths + val_input_lengths)
        decoder_max_length = max(train_output_lengths + val_output_lengths)
        encoder_min_length = min(train_input_lengths + val_input_lengths)
        decoder_min_length = min(train_output_lengths + val_output_lengths)

        print(f"max input length: {encoder_max_length}, max output length: {decoder_max_length}")
        print(f"min input length: {encoder_min_length}, min output length: {decoder_min_length}")
    else:
        encoder_max_length = 97
        decoder_max_length = 122
        encoder_min_length = 4
        decoder_min_length = 7

    # from https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing
    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        # note: we use the source tokenizer's model max length here for both tokenizers, b/c the target tokenizer
        # doesn't have a useful max length. This shouldn't matter much b/c we use this setting here only to make sure
        # that
        inputs = source_tokenizer(batch["input"], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = target_tokenizer(batch["output"], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == source_tokenizer.pad_token_id else token for token in labels] for labels in
                           batch["labels"]]
        return batch

    # encoder_max_len = 0
    # decoder_max_len = 0
    # encoder_min_len = source_tokenizer.model_max_length
    # decoder_min_len = source_tokenizer.model_max_length
    for name, dataset in splits.items():
        print(f"tokeninzing {name} data")
        ds = Dataset.from_dict({"input": dataset.keys(), "output": dataset.values()})
        ds = ds.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
        )

        # encoder_max_len = max(encoder_max_len, max(len(ids) for ids in ds["input_ids"]))
        # decoder_max_len = max(decoder_max_len, max(len(ids) for ids in ds["decoder_input_ids"]))
        # encoder_min_len = min(encoder_min_len, min(len(ids) for ids in ds["input_ids"]))
        # decoder_min_len = min(decoder_min_len, min(len(ids) for ids in ds["decoder_input_ids"]))

        ds.save_to_disk(os.path.join(out_dir, "arsenal_" + name))

    dataset_properties = {}

    dataset_properties["max_source_word_len"] = max_word_len
    dataset_properties["source_vocab"] = source_vocab
    dataset_properties["target_vocab"] = target_vocab
    dataset_properties["special_tokens"] = special_tokens
    dataset_properties["train_val_differences"] = diffs
    dataset_properties["encoder_max_len"] = encoder_max_length
    dataset_properties["decoder_max_len"] = decoder_max_length
    dataset_properties["encoder_min_len"] = encoder_min_length
    dataset_properties["decoder_min_len"] = decoder_min_length

    with open(os.path.join(out_dir, "dataset_properties.json"), "w") as f:
        json.dump(dataset_properties, f, indent=3)


    ###############
    # create inputs for LM evaluation (based on GPT2 LM)

    # the training set is basically the same as the input side of the seq2seq model, except that  it is tokenized with a GPT-specific tokenizer
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # max_length is set somewhat arbitrarily here: to find the best value, we'd need to tokenize twice, once without padding to find
    # the longest generated sequence, and then a second time to use that value. In the dataset from 2017-07-30, the longest generated
    # sequence was 291, so using 512 should give us sufficient capacity for meaningful experiments.
    gpt_train_ds = Dataset.from_dict(gpt_tokenizer(list(train_dataset.keys()), padding="max_length", truncation=True, max_length=512))
    gpt_train_ds.save_to_disk(os.path.join(out_dir, "lm_train"))


if __name__ == "__main__":

    args = parse_arguments(sys.argv)
    build_dataset(args)

