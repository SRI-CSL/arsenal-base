#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
import tqdm
from datasets import Dataset
import random
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
    ent_freqs = {"original": {}, "filtered": {}} # collect statistics about the occurrences of each entity placeholder (for each type and count)

    # read input file, create:
    # - dataset: dict of source/target sentences
    #   (the target entry is actually a list of sentences, b/c. the data set contains multiple identical source entries)
    # - special_tokens: list of special tokens (ARSENAL entity place holders)
    # - target_vocab: list of words used in the target language
    for inst in instances:
        [source, target] = inst.split("\t")
        source_words = source.split(" ")

         # collect statistics for this sentence (before deciding whether to discard it b/c of length)
        for word in source_words:
            m = re.match(".*/([A-Za-z]+)_(\d+).*", word)
            if m:
                typ = m.group(1)
                idx = m.group(2)

                # statistics for the original dataset
                if typ not in ent_freqs["original"]:
                    ent_freqs["original"][typ] = {}
                if idx not in ent_freqs["original"][typ]:
                    ent_freqs["original"][typ][idx] = 0

                ent_freqs["original"][typ][idx] += 1

                # statistics for the filtered dataset
                if len(source_words) < max_word_len:
                    if typ not in ent_freqs["filtered"]:
                        ent_freqs["filtered"][typ] = {}
                    if idx not in ent_freqs["filtered"][typ]:
                        ent_freqs["filtered"][typ][idx] = 0

                    ent_freqs["filtered"][typ][idx] += 1


        if len(source_words) < max_word_len:
            source_vocab.extend(source_words)
           
            for word in source_words:
                if "_" in word:

                    # the dataset might have some compound words separated by "-" (e.g., "the DATA_000-encoding")
                    # we split on "-" to make sure that we don't generate special tokens for these subwords
                    # we need to be careful that "-" is not used in the internal arsenal representation anywhere
                    subwords = word.split("-")
                    for s in subwords:
                        if "_" in s:
                            special_tokens.append(s)

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

    return dataset, special_tokens, source_vocab, target_vocab, ent_freqs

def add_noise(datasets, source_vocab, tokenizer_vocab, ratio, max_noise_length=3, max_noise_instances=3):
    r"""
        Injects noise into the given dataset. Noise is in the form of any additional words that appear in the tokenizer
        vocab (e.g., any English word) and not in the source vocab.

        Args:
            datasets:            a list of datasets to be noised
            source_vocab:        the list of words appearing on the source side of the training data sets (i.e., any words 
                                 supported by the grammar)
            tokenizer_vocab:     the entire language vocab from the respective tokenizer; any words from tokenizer_vocab
                                 that are not in source_vocab can be used as noise
            ratio:               the ratio by which to extend the original dataset with noise instances (e.g., for a ratio 
                                 of 0.2 and a dataset size of 100, 20 noise instances will be added to the dataset)
            max_noise_length:    the maximum length (in words) of any injected noise instance
            max_noise instances: the maximum number of noise instances to insert in any sentence
    """
    noise_vocab = [v for v in tokenizer_vocab if v not in source_vocab]

    for ds_idx, ds in enumerate(datasets):
        source_sentences = list(ds.keys())
        ds_size = len(ds)
        num_noisy_sents = int(ds_size*ratio)
        noisy_sent_idxs = random.sample(range(ds_size), num_noisy_sents)
        noisy_sents = []

        for idx in noisy_sent_idxs:
            sent = source_sentences[idx].split()
            # generate a random number of noise instances of random lengths to add to this sentence
            num_noise_insts = random.randint(1,max_noise_instances) 
            num_noise_insts = min(len(sent)+1, num_noise_insts)
            noise_insts = []
            for _ in range(num_noise_insts):
                noise_inst = []
                len_noise = random.randint(1,max_noise_length)
                for _ in range(len_noise):
                    noise_inst.append(noise_vocab[random.randint(0, len(noise_vocab)-1)])
                noise_insts.append(noise_inst)
                
            # pick random positions in the original sentence for noise injection
            try:
                noise_positions = random.sample(range(len(sent)+1), num_noise_insts)
            
            except :
                print(f"idx: {idx}, len dataset: {len(ds)}, len sent: {len(sent)}")
                raise
            noise_positions.sort()

            # create a new sentence from the original sentence by injecting the generated noise
            # at the chosen positions
            noisy_sent = []
            prev_pos = 0
            for noise, pos in zip(noise_insts, noise_positions):
                noisy_sent.extend(sent[prev_pos:pos])
                noisy_sent.extend(noise)
                prev_pos = pos
            if prev_pos < len(sent):
                noisy_sent.extend(sent[prev_pos:])

            noisy_sents.append(" ".join(noisy_sent))

        for s in noisy_sents:
            ds[s] = "UNSUPPORTED"
        datasets[ds_idx] = ds
    return datasets


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
    inject_noise = args.inject_noise
    noise_ratio = args.noise_ratio

    print(f"train file: {train_file}")
    print(f"data_dir: {data_dir}")

    # if we use curriculum learning, we'll have multiple files with training data
    # with different suffixes (e.g., "eng-pn.train_1.txt", "eng-pn.train_2.txt", etc.).
    # We identify all these files and sort them in ascending order so that we get the 
    # most comprehensive set (representing the entire grammar) last.
    for (_, _, train_files) in os.walk(data_dir):
        break
    train_file_stub = os.path.splitext(train_file)[0]
    train_files = [f for f in train_files if train_file_stub in f]
    train_files.sort(reverse=False)
    
    print(f"discarding all instances with more than {max_word_len} words in the source sentences")

    build_val = os.path.exists(os.path.join(data_dir, val_file))

    train_datasets = [] 

    # note: we keep all resulting 'train_dataset's, so that we can use them for curriculum learning
    # all other information from process_input is overwritten in each iteration, b/c. we only care about the 
    # information from the last (most comprehensive) set
    for train_file in train_files:
        print(f"processing {train_file}")
        train_dataset, special_tokens, source_vocab, target_vocab, train_ent_freqs = process_input(data_dir, out_dir, train_file, max_word_len, ignore_prefixes, ignore_suffixes, check_balance)
        train_datasets.append(train_dataset)

    # we don't really need to dump this information here, because later it will be included in dataset_properties later
    # this is done here only to get these results quickly without having to wait for the lengthy process of tokenization
    with open(os.path.join(out_dir, "training_entity_frequencies.json"), "w") as f:
        json.dump(train_ent_freqs, f, indent=3)

    if build_val:
        val_dataset, val_special_tokens, val_source_vocab, val_target_vocab, val_ent_freqs = process_input(data_dir, out_dir, val_file, max_word_len, ignore_prefixes, ignore_suffixes, check_balance)
    else:
        val_dataset = {}
        val_special_tokens = []
        val_source_vocab = []
        val_target_vocab = []

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

    if inject_noise:
        tokenizer_vocab = source_tokenizer.get_vocab().keys()
        tokenizer_vocab = [v for v in tokenizer_vocab if v[0] not in ["#", "[]"]] #skip special tokens (e.g., subwords)
        train_datasets = add_noise(train_datasets, source_vocab, tokenizer_vocab, noise_ratio)
        
        # not required, but we'll write the clear-text versions of the noisy datasets to file for debugging
        noise_output_dir = os.path.join(data_dir, "noisy")
        if not os.path.exists(noise_output_dir):
            os.mkdir(noise_output_dir)
        
        for train_file, train_dataset in zip(train_files, train_datasets):
            with open(os.path.join(noise_output_dir, train_file.replace(".txt", "_noisy.txt")), "w") as f:
                for s, t in train_dataset.items():
                    f.write(f"{s}\t{t}\n")



    # source_tokenizer.save_pretrained(os.path.join(out_dir, "source_tokenizer"))
    # target_tokenizer.save_pretrained((os.path.join(out_dir, "target_tokenizer")))

    splits = {}

    for i, s in enumerate(train_datasets):
        splits[f"train_{i}"] = s

    if build_val:
        splits["val"] = val_dataset

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
        train_input_lengths = list(map(lambda x: len(source_tokenizer(x).input_ids), train_dataset.keys()))
        train_output_lengths = list(map(lambda x: len(target_tokenizer(x).input_ids), train_dataset.values()))

        if build_val:
            val_input_lengths = list(map(lambda x: len(source_tokenizer(x).input_ids), val_dataset.keys()))
            val_output_lengths = list(map(lambda x: len(target_tokenizer(x).input_ids), val_dataset.values()))
        else:
            val_input_lengths = []
            val_output_lengths = []

        # print(f"max train input: {max(train_input_lengths)}, max val input: {max(val_input_lengths)}")
        # print(f"max train output: {max(train_output_lengths)}, max val output: {max(val_output_lengths)}")
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
    dataset_properties["training_size"] = [len(ds.keys()) for ds in train_datasets]
    dataset_properties["training_entity_frequencies"] = train_ent_freqs
    if build_val:
        dataset_properties["validation_size"] = len(val_dataset.keys())
        dataset_properties["validation_entity_frequencies"] = val_ent_freqs

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

