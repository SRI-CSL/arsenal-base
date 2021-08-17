#!/usr/bin/env python

# mostly followed tutorial from https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing
import argparse
import json
from datetime import datetime
import os
from shutil import copyfile
from pathlib import Path

import datasets
import torch
from tabulate import tabulate
from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint

from arsenal_tokenizer import PreTrainedArsenalTokenizer

# we don't *require* tensorboard here, but if it is not available, training progress won't be logged
# (and it might be really frustrating to realize this after days of training), so try to import it here
# to make sure progress is always logged
import tensorboard

from utils import get_latest_trainingrun, get_latest_checkpoint


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Building the translation model form NL to AST")
parser.add_argument("-data_dir",       type=str,       default="../arsenal/large_files/datasets/2021-04-14T1922",       help="location of the data directory")
parser.add_argument("-epochs",         type=int,       default=1,             help="number of training epochs")
# parser.add_argument("-fp16",           type=str2bool,  default=True,           help="Mixed precision training, can only be used on CUDA devices")
parser.add_argument("-target_model",   type=str,       default="arsenal_model",help="location of the pretrained target model")
parser.add_argument("-output_dir",     type=str,                               help="location of the output directory; if none is provided, ./results/translation/[current_time] is used")
parser.add_argument("-batch_size",     type=int,       default=2,              help="size of training batches")
parser.add_argument("-resume",         type=str2bool,  default=False,          help="tries to resume training from latest model/checkpoint")

args = parser.parse_args()
print(tabulate(vars(args).items(), headers={"parameter", "value"}))
data_dir = args.data_dir
batch_size = args.batch_size

# use mixed precision training on CUDA devices, otherwise disable it so that code can run on CPUs
if torch.cuda.is_available():
    fp16 = True
else:
    fp16 = False


if args.resume == False:
    output_dir = args.output_dir
    os.makedirs(output_dir)
    checkpoint = None
    # copy info about dataset b/c we'll need then when running the model (among others, it contains the target vocab)
    copyfile(os.path.join(data_dir, "dataset_properties.json"), os.path.join(output_dir, "dataset_properties.json"))
else:
    run_id = get_latest_trainingrun(args.results_dir)
    output_dir = os.path.join(args.results_dir, run_id, "translation_model")
    checkpoint = get_last_checkpoint(output_dir)
    print(f"trying to resume training from {checkpoint} in {output_dir}")


if output_dir is None:
    output_dir=os.path.join("./results/translation/", datetime.now().strftime("%b%d_%H-%M-%S"))
logging_dir=os.path.join(output_dir, "logs")

with open(os.path.join(Path(output_dir).parent, "translation-args.txt"), "w") as f:
    print(tabulate(vars(args).items(), headers={"parameter", "value"}), file=f)

dataset_properties = json.load(open(os.path.join(data_dir, "dataset_properties.json")))
special_tokens = dataset_properties["special_tokens"]
target_vocab = dataset_properties["target_vocab"]

bert2arsenal = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", args.target_model)
source_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# only needed to get the id of the EOS token in the target language
target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)

# Due to the additional special tokens, encoder token embeddings need to be resized.
# The target model has been created specifically for the "Effigy Arsenal Language" so has already correct dims
bert2arsenal.encoder.resize_token_embeddings(len(source_tokenizer))
bert2arsenal.config.decoder_start_token_id = source_tokenizer.cls_token_id
bert2arsenal.config.eos_token_id = target_tokenizer.sep_token_id

# not sure whether these settings are relevant? (At least they shouldn't be harmful)
bert2arsenal.config.encoder.eos_token_id = source_tokenizer.sep_token_id
bert2arsenal.config.decoder.eos_token_id = target_tokenizer.sep_token_id

bert2arsenal.config.pad_token_id = source_tokenizer.pad_token_id
bert2arsenal.config.vocab_size = bert2arsenal.encoder.vocab_size
bert2arsenal.config.encoder.vocab_size = bert2arsenal.encoder.vocab_size

# todo: verify that these parameters actually refer to target/decoder sequences
bert2arsenal.config.max_length = dataset_properties["decoder_max_len"]
bert2arsenal.config.min_length = dataset_properties["decoder_min_len"]

# prevents 3 (or more) repetitions of the same n-gram in the output by manually setting the probability of
# words that could represent a third n-gram to zero.
# Todo: inspect arsenal's effigy grammar to see if/what setting for this makes sense
bert2arsenal.config.no_repeat_ngram_size = 0
bert2arsenal.config.early_stopping = True
bert2arsenal.config.length_penalty = 2.0
bert2arsenal.config.num_beams = 4
# bert2arsenal.config.add_cross_attention
# bert2arsenal.config.num_return_sequences = 5 # this can be used to set the number of return sequences

print(f"model config:\n{bert2arsenal.config}")

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    # evaluation_strategy="epoch",
    # eval_steps=4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=fp16,
    output_dir=output_dir,
    logging_dir=logging_dir,
    logging_steps=100,
    save_steps=10000,
    warmup_steps=2000,
    save_total_limit=1,
    num_train_epochs=args.epochs,
    do_train=True,
    do_eval=False
)

bert2arsenal.config.to_json_file(os.path.join(output_dir, "model_config.json"))
with open(os.path.join(output_dir, "training_args.json"), "w") as f:
    f.write(str(training_args.to_json_string()))

train_data = datasets.Dataset.load_from_disk(os.path.join(data_dir, "arsenal_train"))
val_data = datasets.Dataset.load_from_disk(os.path.join(data_dir, "arsenal_val"))

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


trainer = Seq2SeqTrainer(
    model=bert2arsenal,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_data,
    # eval_dataset=val_data,
    tokenizer=source_tokenizer
)
print(f"start training at {datetime.now().strftime('%b%d_%H-%M-%S')}")
trainer.train(resume_from_checkpoint=checkpoint)
