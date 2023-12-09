#!/usr/bin/env python
import json
import logging
import os
import sys
from pathlib import Path
from tabulate import tabulate
import datasets
from transformers import CONFIG_MAPPING, AutoModelForMaskedLM, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertConfig
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from args import parse_arguments
from arsenal_tokenizer import PreTrainedArsenalTokenizer

# we don't *require* tensorboard here, but if it is not available, training progress won't be logged
# (and it might be really frustrating to realize this after days of training), so try to import it here
# to make sure progress is always logged
import tensorboard

logger = logging.getLogger(__name__)


def prepare_dataset(file):
    dataset = datasets.Dataset.load_from_disk(file)
    dataset = dataset.remove_columns(["input", "input_ids", "attention_mask", "labels", ])
    dataset = dataset.rename_column("decoder_input_ids", "input_ids")
    dataset = dataset.rename_column("decoder_attention_mask", "attention_mask")
    dataset = dataset.rename_column("output", "text")
    return dataset

def train_targetmodel(args):
    dataset_properties = json.load(open(os.path.join(args.data_dir, "dataset_properties.json")))
    target_vocab = dataset_properties["target_vocab"]
    train_dataset = prepare_dataset(os.path.join(args.data_dir, args.train_dataset_name))

    tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_root_dir, args.run_id, args.target_model_name),
        logging_dir=os.path.join(args.model_root_dir, args.run_id, args.target_model_name, "logs"),
        num_train_epochs=args.target_epochs,  # total # of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
    )

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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if os.path.isdir(training_args.output_dir):
        if len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. ")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print(tabulate(vars(args).items(), headers={"parameter", "value"}))

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    train_targetmodel(args)


