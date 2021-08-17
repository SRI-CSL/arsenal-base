#!/usr/bin/env python
import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

import datasets
from transformers import CONFIG_MAPPING, AutoModelForMaskedLM, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertConfig
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from arsenal_tokenizer import PreTrainedArsenalTokenizer

# we don't *require* tensorboard here, but if it is not available, training progress won't be logged
# (and it might be really frustrating to realize this after days of training), so try to import it here
# to make sure progress is always logged
import tensorboard

logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Building the target language model")
parser.add_argument("-data_dir",            type=str,       default="./data",      help="location of the data directory")
parser.add_argument("-epochs",              type=int,       default=1,            help="number of training epochs")
parser.add_argument("-output_dir",          type=str,                              help="location of the output directory; if none is provided, ./results/target_model/[current_time] is used")
parser.add_argument("-train",               type=str2bool,  default=True,          help="Whether to train the model")
parser.add_argument("-eval",                type=str2bool,  default=False,         help="Whether to evaluate the model")
parser.add_argument("-batch_size",          type=int,       default=16,            help="batch size for training")
parser.add_argument("-hidden_size",         type=int,       default=768,           help="size of single token embedding")
parser.add_argument("-intermediate_size",   type=int,       default=3072,          help="size of feed-forward layer")
parser.add_argument("-num_hidden_layers",   type=int,       default=12,            help="number of hidden layers")
parser.add_argument("-num_attention_heads", type=int,       default=12,            help="number of LM attention heads")

args = parser.parse_args()
print(tabulate(vars(args).items(), headers={"parameter", "value"}))

data_dir = args.data_dir
output_dir = args.output_dir
if output_dir is None:
    output_dir=os.path.join("./results/target_model/", datetime.now().strftime("%b%d_%H-%M-%S"))
    print(f"output dir: {output_dir}")
else:
    os.makedirs(output_dir, exist_ok=True)
logging_dir=os.path.join(output_dir, "logs")

#f = open(os.path.join(Path(output_dir).parent, "target-args.txt"), "w")
with open(os.path.join(Path(output_dir).parent, "target-args.txt"), "w") as f:
    print(tabulate(vars(args).items(), headers={"parameter", "value"}), file=f)

dataset_properties = json.load(open(os.path.join(data_dir, "dataset_properties.json")))
target_vocab = dataset_properties["target_vocab"]

def prepare_dataset(name):
    dataset = datasets.Dataset.load_from_disk(os.path.join(data_dir, name))
    dataset = dataset.remove_columns(["input", "input_ids", "attention_mask", "labels", ])
    dataset = dataset.rename_column("decoder_input_ids", "input_ids")
    dataset = dataset.rename_column("decoder_attention_mask", "attention_mask")
    dataset = dataset.rename_column("output", "text")
    return dataset

train_dataset = prepare_dataset("arsenal_train")
eval_dataset = prepare_dataset("arsenal_val")

tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

config : BertConfig = CONFIG_MAPPING["bert"]()
config.max_position_embeddings = dataset_properties["decoder_max_len"]
config.vocab_size = len(tokenizer)
config.hidden_size = args.hidden_size
config.intermediate_size = args.intermediate_size
config.num_hidden_layers = args.num_hidden_layers
config.num_attention_heads = args.num_attention_heads

model = AutoModelForMaskedLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))
print(f"model config:\n{model.config}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    num_train_epochs=args.epochs,    # total # of training epochs
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    do_train=args.train,
    do_eval=args.eval,
    save_steps=10000,
    save_total_limit=1,
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,

)

last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Training
if training_args.do_train:
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    ## not yet incorporated in stable release
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

model.save_pretrained(output_dir)

# Evaluation
results = {}
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    results["perplexity"] = perplexity

    # trainer.log_metrics("eval", results)
    # trainer.save_metrics("eval", results)

    print(results)
