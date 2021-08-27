import shutil
import sys
from datetime import datetime
import os
from pathlib import Path
from tabulate import tabulate
from args import parse_arguments
from bert2arsenal import train_translationmodel
from build_dataset import build_dataset
from generate_predictions import generate_predictions
from target_model import train_targetmodel

if __name__ == "__main__":

    ########## some preliminary setup ##########

    args = parse_arguments(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    out_dir = os.path.join(args.model_root_dir, args.run_id)

    if os.path.exists(out_dir):
        choice = None
        while True:
            choice = input(f"output dir {out_dir} already exists. Delete? (y/n)")
            if choice.lower() == "y":
                shutil.rmtree(out_dir)
                break
            elif choice.lower() == "n":
                sys.exit(0)
            else:
                print("unrecognized input")
    os.mkdir(out_dir)

    print(tabulate(vars(args).items(), headers={"parameter", "value"}))
    with open(os.path.join(Path(out_dir), "args.txt"), "w") as f:
        print(tabulate(vars(args).items(), headers={"parameter", "value"}), file=f)


    ########## preparing the dataset ##########

    if not args.skip_databuild and not args.resume:
        print(f"\n*** {datetime.now()}: building data set ***")
        build_dataset(args)


    ########## training the target LM model ##########

    if not args.resume:
        print(f"\n\n*** {datetime.now()}: training target LM model ***\n")
        train_targetmodel(args)


    ########## training the translation model ##########

    print(f"\n\n*** {datetime.now()}: training translation model ***\n")
    train_translationmodel(args)


    ########## generating translations with trained model ##########

    print(f"\n\n*** {datetime.now()}: generating predictions ***\n")
    generate_predictions(args)

