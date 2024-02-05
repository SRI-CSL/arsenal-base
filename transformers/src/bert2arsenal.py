#!/usr/bin/env python
import shutil
import json
import sys
from datetime import datetime
import os
from shutil import copyfile
from pathlib import Path
import datasets
import torch
from tabulate import tabulate
from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, IntervalStrategy, GenerationConfig
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

        if os.path.exists(output_dir):
            choice = None
            while True:
                choice = "y" if args.y else input(f"output dir {output_dir} already exists. Delete? (y/n)")
                if choice.lower() == "y":
                    shutil.rmtree(output_dir)
                    break
                elif choice.lower() == "n":
                    sys.exit(0)
                else:
                    print("unrecognized input")

        os.mkdir(output_dir)
        # copy info about dataset b/c we'll need that when running the dockerized model (among others, it contains the target vocab)
        copyfile(os.path.join(args.data_dir, "dataset_properties.json"), os.path.join(output_dir, "dataset_properties.json"))
    else:
        checkpoint = get_last_checkpoint(output_dir)
        print(f"trying to resume training from {checkpoint} in {output_dir}")

    # use mixed precision training on CUDA devices, otherwise disable it so that code can run on CPUs
    fp16 = True if torch.cuda.is_available() else False

    bert2arsenal = EncoderDecoderModel.from_encoder_decoder_pretrained(args.source_model, target_model)
    source_tokenizer = BertTokenizerFast.from_pretrained(args.source_model)
    source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # save it for later use s.t. we don't have to download anything for the runtime
    source_tokenizer.save_pretrained(os.path.join(output_dir, "source_tokenizer"))

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

    # Don't prevent any n-gram repetitions! This would have a significant negative influence on
    # the translations (especially for longer sentences), because the correct CSTs may contain n-gram repetitions
    bert2arsenal.config.no_repeat_ngram_size = 0


    # Don't prevent any n-gram repetitions! This would have a significant negative influence on
    # the translations (especially for longer sentences), because the correct CSTs may contain n-gram repetitions
    bert2arsenal.config.no_repeat_ngram_size = 0

    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        num_return_sequences=args.num_outputs,
        no_repeat_ngram_size=0,
        decoder_start_token_id=target_tokenizer.cls_token_id,
        eos_token_id=bert2arsenal.config.eos_token_id,
        pad_token_id=bert2arsenal.config.pad_token_id,
        min_length = dataset_properties["decoder_min_len"],
        max_length = dataset_properties["decoder_max_len"],
        max_new_tokens = dataset_properties["decoder_max_len"],
    )
    bert2arsenal.generation_config = generation_config

    print(f"model config:\n{bert2arsenal.config}")

    if args.early_stopping:
        args.do_validation = True

    training_args = {}
    if args.do_validation:
        training_args["evaluation_strategy"] = IntervalStrategy.STEPS
        training_args["eval_steps"] = args.eval_steps
        if args.early_stopping:
            training_args["load_best_model_at_end"] = True

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=fp16,
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        num_train_epochs=args.translation_epochs,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=args.eval_steps, # saving steps need to match eval steps s.t. best model can be loaded at the end
        save_total_limit=args.save_total_limit,
        **training_args
    )

    bert2arsenal.config.to_json_file(os.path.join(output_dir, "model_config.json"))
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        f.write(str(training_args.to_json_string()))

    train_data = datasets.Dataset.load_from_disk(os.path.join(args.data_dir, args.train_dataset_name))
    
    # transformers 4.12 changed handling of EncoderDecoderModels. 
    # - `decoder_input_ids` and `decoder_attention_mask` were required in earlier versions but MUST be removed with
    #    transformers >= 4.12 otherwise training will produce nonsense results (repeating the same token forever with
    #    a loss of 0.0).
    # - `the first token of the labels (sequence start token) must be removed from the labels, because a built-in 
    #    function within `forward` automatically prepends this token. So the actual model inputs and outputs still
    #    contain this token. Hence, training and validation data needs to be stripped of this token, test data (where
    #    the expected result is not fed through the forward loop) need to keep it.
    train_data = train_data.remove_columns(["decoder_input_ids", "decoder_attention_mask"])

    def remove_start_label(x):
        x["labels"] = x["labels"][1:]
        return x

    train_data = train_data.map(remove_start_label)

    trainer_args = {}

    if args.do_validation:
        val_data = datasets.Dataset.load_from_disk(os.path.join(args.data_dir, args.val_dataset_name))
        val_data = val_data.remove_columns(["decoder_input_ids", "decoder_attention_mask"])
        val_data = val_data.map(remove_start_label)
        trainer_args["eval_dataset"] = val_data
        if args.early_stopping:
            trainer_args["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    
    trainer = Seq2SeqTrainer(
        model=bert2arsenal,
        args=training_args,
        train_dataset=train_data,
        tokenizer=source_tokenizer,
        **trainer_args

    )

    print(f"start training at {datetime.now().strftime('%b%d_%H-%M-%S')}")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    print(tabulate(vars(args).items(), headers={"parameter", "value"}))
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    train_translationmodel(args)


