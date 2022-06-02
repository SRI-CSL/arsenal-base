#!/usr/bin/env python

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime
import os
from pathlib import Path

import numpy as np

import datasets
import torch
from tabulate import tabulate
from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    GPT2TokenizerFast, TrainingArguments, Trainer, GPT2Config, GPT2Model, DataCollatorForLanguageModeling, \
    GPT2LMHeadModel, BertConfig, AutoModelForMaskedLM
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import tensorboard

'''
Scores real sentences based on the perplexity of a causal LM trained on synthetic data. This should indicate coverage
of the grammar which was used to create the synthetic data.

Main argumentss:
- data_basedir:   The root location in which datasets are stored
- dataset:        Name of the dataset that is used for training the CLM. If none is provided, the latest from 
                  data_basedir is used

The trained model is used to score sentences from three different datasets:
- val_set:        This is the synthetic test set (created together with the synthetic training set used for CLM training).
                  Scores from this set are used to obtain an (average) reference score for covered examples.
- test_set:       All (entity-processed) sentences from documents that exist in our corpus. This will contain mostly
                  uncovered sentences. Comparing with the reference score from the validation set should indicate which
                  sentences are covered and which are not.
- example_set:    A much smaller subset of the test_set that we are currently focusing on. Here, the ratio of covered
                  sentences should be much higher than in the test-set, and so hopefully we'll see better scores in 
                  in this set.

'''

logger = logging.getLogger(__name__)


def clean_set(instances, min_len=3):
    '''
    transform all sentences to lowercase, remove duplicates, and discard sentences shorter than min_len
    '''

    # lowercase everything (our bert model is uncased, and lowercasing everything will help to remove more duplicates)
    instances = [i.strip().lower() for i in instances]
    # remove duplicates
    instances = list(set(instances))

    cleaned = []
    discarded = []

    for i in instances:
        # strip away all html tags
        i = re.sub("<.*?>", " ", i)

        # discard all instances with less than 5 words
        if len(i.split()) < min_len:
            discarded.append(i)
        else:
            cleaned.append(i)

    return cleaned, discarded

parser = argparse.ArgumentParser(description="Build LM model from synthetic dataset to score coverage of real sentences")
parser.add_argument("-data_basedir",   type=str,       default="../../../large_files/datasets/", help="base directory in which data sets are stored")
parser.add_argument("-dataset",        type=str,                                help="name of the dataset to be used. If none is provided, the latest dataset is used")
parser.add_argument("-test_file",      type=str,       default="../../../eval/lm_scoring/spec_sentences_clean.txt", help="location of the test file (real sentences from the specs)")
parser.add_argument("-example_dir",    type=str,       default="../../../../generated-files/csts", help="location of the (entity-processed) example snippets directory")
parser.add_argument("-val_size",       type=int,       default=10000,          help="number of instances to use from the validation set (if none is provided, the entire val set is scored - this might take days!)")
parser.add_argument("-model_dir",      type=str,                               help="model location; if none is provided, ../arsenal/large_files/models/lm/[data_dir]_[model_type] is used for training")
parser.add_argument("-out_dir",        type=str,       default="../../../eval/lm_scores/",help="output location")
parser.add_argument("-source_model",   type=str,       default="bert-base-uncased",help="The pretrained language model for the source language (from https://huggingface.co/models)")
parser.add_argument("-epochs",         type=int,       default=1,              help="number of training epochs")
parser.add_argument("-resume",                         action='store_true',    help="tries to resume training from latest model/checkpoint")
parser.add_argument("-skiptrain",                      action='store_true',    help="skip model training")
parser.add_argument("-skipeval",                       action='store_true',    help="skip evaluation")
parser.add_argument("-model_type",     type=str,       default="gpt2",         help="the type of model to use, gpt2 or bert")
parser.add_argument("-cuda_devices",   type=str,       default="6,7",          help="GPUs to use (as a list of comma-separated numbers)")
parser.add_argument("-batch_size",     type=int,       default=1,              help="batch size (should be 1 b/c scoring doesn't seem to work in batches)")
parser.add_argument("-min_length",     type=int,       default=5,              help="minimum sentence length in words (shorter sentences will be discarded)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

if args.dataset is None:
    for (_, data_sets, _) in os.walk(args.data_basedir):
        break
    data_sets = [d for d in data_sets if d.startswith("20")]
    data_sets.sort()
    args.dataset = data_sets[-1]


if args.model_dir is None:
    args.model_dir = os.path.join("../../../large_files/models/lm-scoring/", args.dataset + "_" + args.model_type)

if args.val_size is None:
    val_size = -1

data_basedir = args.data_basedir
dataset = args.dataset
data_dir = os.path.join(data_basedir, dataset)
test_file = args.test_file
val_size = args.val_size
model_dir = args.model_dir
model_type = args.model_type
out_dir = args.out_dir
example_dir = args.example_dir
val_size = args.val_size
train = not args.skiptrain
eval = not args.skipeval
min_len = args.min_length

print(tabulate(vars(args).items(), headers={"parameter", "value"}))
with open(os.path.join(Path(out_dir), "args.txt"), "w") as f:
    print(tabulate(vars(args).items(), headers={"parameter", "value"}), file=f)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
logging_dir=os.path.join(model_dir, "logs")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

if model_type == "gpt2":
    # the huggingface servers had temporary problems to serve the tokenizer, so we saved it for
    # offline use - replacing "gpt2_offline" with "gpt2" will download the latest again. The one
    # saved for offline use should be identical to the regular one.
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if train:
        train_dataset = datasets.Dataset.load_from_disk(os.path.join(data_dir, "lm_train"))


elif model_type == "bert":
    dataset_properties = json.load(open(os.path.join(data_dir, "dataset_properties.json")))
    special_tokens = dataset_properties["special_tokens"]
    tokenizer = BertTokenizerFast.from_pretrained(args.source_model)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    config = BertConfig()
    config.vocab_size = len(tokenizer)

    model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # the NL inputs for the train dataset are the same for BERT and GPT-2 models, but they are tokenized
    # differently (using the corresponding BERT and GPT-2 tokenizers, respectively). The standard training
    # set is already tokenized with the BERT tokenizer, so we can reuse that set here.
    if train:
        train_dataset = datasets.Dataset.load_from_disk(os.path.join(data_dir, "arsenal_train"))


else:
    raise("unknown model type")

# the train part is pretty much identical to the one from training the target language model (cf. target_model.py)
# except that we train a GPT2 instead of a BERT model
if train:
    print("training")

    training_args = TrainingArguments(
        output_dir=model_dir,
        logging_dir=logging_dir,
        num_train_epochs=args.epochs,    # total # of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_steps=200,
        save_total_limit=1,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    model.save_pretrained(model_dir)

def eval_scores(instances):
    if model_type == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(get_last_checkpoint(model_dir))
    elif model_type == "bert":
        model = AutoModelForMaskedLM.from_pretrained(get_last_checkpoint(model_dir))

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scores = []

    # It seems that batch processing returns aggregated losses over the entire batch.
    # Since the goal is to obtain loss scores for each instance, we consequently can't
    # use batch processing here.
    for i in datasets.tqdm(instances):

        if i == "":
            print(f"skipping empty sentence {i}")
            continue

        tokens = tokenizer(i, return_tensors="pt", truncation=True)["input_ids"]
        tokens.to(torch_device)

        # this generates cross entropy losses, so no need to modify anything inside the huggingface library?
        # The loss is specified like this:
        #   loss_fct = CrossEntropyLoss()
        #   loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # (cf. models/gpt2/modeling_gpt2.py: GPT2LMHeadModel.forward())

        try:
            # print(f"trying to send '{i}'")
            outputs = model(tokens, labels=tokens)
        except Exception as e:
            print(f"couldn't score sentence '{i}': {e}")

        score = outputs["loss"]

        # # this transforms the (cross entropy) loss into perplexity - but do we need this step at all?
        # # We are mainly interested in getting the relative differences between scores to figure out which
        # # sentences are well-covered by our model (and thus by our grammar) and which are not. Simply looking
        # # at the loss would already give us that information, the monotonic exp transformation doesn't give
        # # any additional benefit here, right?
        # score = torch.exp(score)

        scores.append({"sentence": i, "score": float(score)})

    return scores

if eval:

    # the artificially generated validation set
    val_file = os.path.join(data_dir, "eng-pn.val.txt")
    val_set = []
    with open(val_file, "r") as f:
        for l in f.read().splitlines()[:val_size]:
            val_set.append(l.split("\t")[0])

    # the set of all collected (and entity-processed) sentences from the standards docs
    with open(test_file, "r") as f:
        test_set = f.read().splitlines()

    # the set of selected examples to focus on (should be
    example_set = []

    for (_, _, example_files) in os.walk(example_dir):
        break
    example_files = [f for f in example_files if f.endswith(".json")]

    for example_file in example_files:
        with open(os.path.join(example_dir, example_file), "r") as f:
            instances = json.load(f)
            example_set.extend([i["cleaned-text"] for i in instances])

    val_set, _ = clean_set(val_set, min_len)
    test_set, _ = clean_set(test_set, min_len)
    example_set, _ = clean_set(example_set, min_len)

    results = {}
    print(f"evaluating {val_file}")

    results["val"] = eval_scores(val_set)

    print(f"evaluating {test_file}")
    results["test"] = eval_scores(test_set)

    print(f"evaluating {example_dir}")
    results["examples"] = eval_scores(example_set)


    for r in ["val", "test", "examples"]:
        stats = {}
        instances = results[r]
        instances.sort(key= lambda x: x["score"])

        scores = []
        for i in instances:
            if math.isnan(i["score"]):
                print(f"no score for sentence '{i['sentence']}'")
            else:
                scores.append(i["score"])

        model_id = os.path.basename(os.path.normpath(args.model_dir))
        stats["model"] = model_id
        stats['dataset'] = r
        stats['mean'] = np.mean(scores)
        stats["median"] = np.median(scores)
        stats['std'] = np.std(scores)
        stats['min'] = np.min(scores)
        stats['max'] = np.max(scores)

        print(f"results for {r} set:")
        print(tabulate(stats.items()))
        print("\n")

        file_prefix = os.path.join(out_dir, f"lm_scores_{model_id}_{r}")

        # with open(f"{file_prefix}_scores.json", "w") as f:
        #     json.dump(instances, f, indent=3)

        # same as previous, but better readable for manual inspection
        with open(f"{file_prefix}_scores.txt", "w") as f:
            for i in instances:
                f.write(f"{i['score']}\t{i['sentence']}\n")

        with open(f"{file_prefix}_stats.json", "w") as f:
            json.dump(stats, f, indent=3)





