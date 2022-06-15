#!/usr/bin/env python

import shutil
import sys
import time
from datetime import datetime
import os
from pathlib import Path
from tabulate import tabulate
from args import parse_arguments
from bert2arsenal import train_translationmodel
from build_dataset import build_dataset
from eval import evaluate_predictions
from generate_predictions import generate_predictions
from target_model import train_targetmodel

def print_td(t0, t1):
    delta = round(t1-t0)

    secs_day = 60 * 60 * 24
    secs_hour = 60 * 60
    secs_min = 60

    days = delta // secs_day
    hours = (delta - (days * secs_day)) // secs_hour
    minutes = (delta - (days * secs_day) - (hours * secs_hour)) // secs_min
    seconds = delta - (days * secs_day) - (hours * secs_hour) - minutes * secs_min

    return f"{days:02d}d:{hours:02d}h:{minutes:02d}m:{seconds:02d}s"

if __name__ == "__main__":

    ########## some preliminary setup ##########

    args = parse_arguments(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    out_dir = os.path.join(args.model_root_dir, args.run_id)

    if os.path.exists(out_dir):
        choice = None
        while True:
            choice = "y" if args.y else input(f"output dir {out_dir} already exists. Delete? (y/n)")
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


    t0 = time.time()

    ########## preparing the dataset ##########

    if not args.skip_databuild and not args.resume:
        print(f"\n*** {datetime.now()}: building data set ***")
        build_dataset(args)

    t1 = time.time()

    ########## training the target LM model ##########

    if not args.resume:
        print(f"\n\n*** {datetime.now()}: training target LM model ***\n")
        train_targetmodel(args)

    t2 = time.time()

    ########## training the translation model ##########

    print(f"\n\n*** {datetime.now()}: training translation model ***\n")
    train_translationmodel(args)

    t3 = time.time()

    ########## generating translations with trained model ##########

    print(f"\n\n*** {datetime.now()}: generating predictions ***\n")
    generate_predictions(args)

    t4 = time.time()

    ########## evaluation of generated predictions ##########

    print(f"\n\n*** {datetime.now()}: evaluating predictions ***\n")
    evaluate_predictions(args)

    t5 = time.time()

    ########## done - output timing information ##########

    print(f"\n\n*** {datetime.now()}: done ***\n")

    print(f"\ntime spent:\n")

    timing = {}

    timing["dataset construction"]          = print_td(t0, t1)
    timing["target LM model training"]      = print_td(t1, t2)
    timing["translation model training"]    = print_td(t2, t3)
    timing["prediction generation"]         = print_td(t3, t4)
    timing["prediction generation"]         = print_td(t4, t5)

    print(tabulate(timing.items(), headers=["total", print_td(t0, t5)]))




