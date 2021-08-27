import argparse
import sys
from datetime import datetime


def parse_arguments(input_args):

    print(sys.argv)
    parser = argparse.ArgumentParser(description="Arsenal transformer pipeline")

    # environment settings (paths, file/folder names, CUDA devices, etc.)
    parser.add_argument("-data_dir",                type=str,   default="../../../large_files/datasets/2021-07-30T0004",
                                                                                            help="location of the input data")
    parser.add_argument("-data_out_dir",            type=str,                               help="location for the generated datasets (if none is provided, data_dir is used")
    parser.add_argument("-train_file",              type=str,   default="eng-pn.train.txt", help="name of iput training data file")
    parser.add_argument("-val_file",                type=str,   default="eng-pn.val.txt",   help="name of input validation data file")
    parser.add_argument("-train_dataset_name",      type=str,   default="arsenal_train",    help="name of the generated training dataset")
    parser.add_argument("-val_dataset_name",        type=str,   default="arsenal_val",      help="name of the generated validation dataset")
    parser.add_argument("-model_root_dir",          type=str,   default="../../../large_files/models/transformers",
                                                                                            help="root location of all generated models")
    parser.add_argument("-run_id",                  type=str,                               help="name of the folder below root dir (the dir in which "
                                                                 "both target LM and translation model of a single run will be stored. If none is provided, "
                                                                 "MM-DD-YYYY based on the current date is used")
    parser.add_argument("-target_model_name",       type=str,   default="target_model",     help="folder base name where the target LM will be stored")
    parser.add_argument("-translation_model_name",  type=str,   default="translation_model",help="folder base name where the translation model will be stored")
    parser.add_argument("-cuda_devices",            type=str, default="6,7",                help="GPUs to use (as a list of comma-separated numbers)")

    # settings for dataset preparation
    parser.add_argument("-max_source_len",          type=int,   default=75,                 help="maximum number of words in the English sentences (all instances above this threshold will be discarded")

    # model configuration for target LM model
    # note: the source LM model uses a pretrained BERT model and thus can't be configured separately
    parser.add_argument("-hidden_size",             type=int,   default=768,                help="size of single token embedding in target LM model")
    parser.add_argument("-intermediate_size",       type=int,   default=3072,               help="size of feed-forward layer in target LM model")
    parser.add_argument("-num_hidden_layers",       type=int,   default=12,                 help="number of hidden layers in target LM model")
    parser.add_argument("-num_attention_heads",     type=int,   default=12,                 help="number of LM attention heads in target LM model")

    # training process configuration
    parser.add_argument("-batch_size",              type=int,   default=4,                  help="batch size for tokenizing")
    parser.add_argument("-warmup_steps",            type=int,   default=500,                help="number of warmup steps for learning rate scheduler")
    parser.add_argument("-weight_decay",            type=float, default=0.01,               help="strength of weight decay")
    parser.add_argument("-logging_steps",           type=int,   default=100,                help="step interval to log progress")
    parser.add_argument("-save_steps",              type=int,   default=10000,              help="step interval to save checkpoint")
    parser.add_argument("-save_total_limit",        type=int,   default=1,                  help="number of checkpoints to keep")
    parser.add_argument("-target_epochs",           type=int,   default=1,                  help="number of training epochs for target LM")
    parser.add_argument("-translation_epochs",      type=int,   default=1,                  help="number of training epochs for translation model")
    parser.add_argument("-skip_databuild",                      action='store_true',        help="skip the dataset building step")
    parser.add_argument("-resume",                              action='store_true',        help="resume training of the translation model  from last checkpoint (automatically skips data build and target LM training)")

    # translation generation configuration (can be changed without changing anything to trained models)
    parser.add_argument("-num_beams",               type=int,   default=1,                  help="number of beams to use in beam search when generating predictions")
    parser.add_argument("-num_outputs",             type=int,   default=1,                  help="number of beam search outputs to generate for each instance")
    parser.add_argument("-type_forcing",            action='store_true',                    help="use grammar to enforce correctly typed translation outputs. This requires a specially patched version of huggingface's"
                                                                                                 "transformers library. This did not improve results in our experiments.")

    args = parser.parse_args(input_args)
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%m-%d-%Y")

    if args.data_out_dir is None:
        args.data_out_dir = args.data_dir

    return args