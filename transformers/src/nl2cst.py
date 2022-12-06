#!/usr/bin/env python

import argparse
import json
import sys
from datetime import datetime
import os
from shutil import copyfile
from pathlib import Path
import datasets
import torch
from tabulate import tabulate
from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint

from args import parse_arguments
from arsenal_tokenizer import PreTrainedArsenalTokenizer

# we don't *require* tensorboard here, but if it is not available, training progress won't be logged
# (and it might be really frustrating to realize this after days of training), so try to import it here
# to make sure progress is always logged
import tensorboard

def train_translationmodel(args):
    dataset_properties = json.load(open(os.path.join(args.data_dir, "dataset_properties.json")))
    special_tokens = dataset_properties["special_tokens"]
    target_vocab = dataset_properties["target_vocab"]

    target_model = os.path.join(args.model_root_dir, args.run_id, args.target_model_name)
    output_dir = os.path.join(args.model_root_dir, args.run_id, args.translation_model_name)
    logging_dir = os.path.join(output_dir, "logs")
    if args.resume == False:
        checkpoint = None
        os.mkdir(output_dir)
        # copy info about dataset b/c we'll need that when running the dockerized model (among others, it contains the target vocab)
        copyfile(os.path.join(args.data_dir, "dataset_properties.json"), os.path.join(output_dir, "dataset_properties.json"))
    else:
        checkpoint = get_last_checkpoint(output_dir)
        print(f"trying to resume training from {checkpoint} in {output_dir}")

    # use mixed precision training on CUDA devices, otherwise disable it so that code can run on CPUs
    fp16 = True if torch.cuda.is_available() else False

    nl2cst = EncoderDecoderModel.from_encoder_decoder_pretrained(args.source_model, target_model)
    source_tokenizer = BertTokenizerFast.from_pretrained(args.source_model)
    source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # save it for later use s.t. we don't have to download anything for the runtime
    source_tokenizer.save_pretrained(os.path.join(output_dir, "source_tokenizer"))

    # only needed to get the id of the EOS token in the target language
    target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

    # Due to the additional special tokens, encoder token embeddings need to be resized.
    # The target model has been created specifically for the "Effigy Arsenal Language" so has already correct dims
    nl2cst.encoder.resize_token_embeddings(len(source_tokenizer))
    nl2cst.config.decoder_start_token_id = source_tokenizer.cls_token_id
    nl2cst.config.eos_token_id = target_tokenizer.sep_token_id

    # not sure whether these settings are relevant? (At least they shouldn't be harmful)
    nl2cst.config.encoder.eos_token_id = source_tokenizer.sep_token_id
    nl2cst.config.decoder.eos_token_id = target_tokenizer.sep_token_id

    nl2cst.config.pad_token_id = source_tokenizer.pad_token_id
    nl2cst.config.vocab_size = nl2cst.encoder.vocab_size
    nl2cst.config.encoder.vocab_size = nl2cst.encoder.vocab_size

    # the model has min/max length settings in three places: for the main moder (EncoderDecoder) and both encoder
    # and decoder as submodels. Settings in the latter two parts seem to be completely irrelevant (unless one would
    # try to use the trained encoder or decoder parts from the translation model in isolation).
    nl2cst.config.max_length = dataset_properties["decoder_max_len"]
    nl2cst.config.min_length = dataset_properties["decoder_min_len"]

    # Don't prevent any n-gram repetitions! This would have a significant negative influence on
    # the translations (especially for longer sentences), because the correct CSTs may contain n-gram repetitions
    nl2cst.config.no_repeat_ngram_size = 0
    nl2cst.config.early_stopping = True
    nl2cst.config.length_penalty = 2.0
    nl2cst.config.num_beams = 4
    # nl2cst.config.add_cross_attention
    # nl2cst.config.num_return_sequences = 5 # this can be used to set the number of return sequences

    print(f"model config:\n{nl2cst.config}")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=fp16,
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        num_train_epochs=args.translation_epochs,
    )

    nl2cst.config.to_json_file(os.path.join(output_dir, "model_config.json"))
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        f.write(str(training_args.to_json_string()))

    train_data = datasets.Dataset.load_from_disk(os.path.join(args.data_dir, args.train_dataset_name))

    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    trainer = Seq2SeqTrainer(
        model=nl2cst,
        args=training_args,
        train_dataset=train_data,
        tokenizer=source_tokenizer
    )
    print(f"start training at {datetime.now().strftime('%b%d_%H-%M-%S')}")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    print(tabulate(vars(args).items(), headers={"parameter", "value"}))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    train_translationmodel(args)


