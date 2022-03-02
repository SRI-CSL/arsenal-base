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

    ########## done - output timing information ##########

    print(f"\n\n*** {datetime.now()}: done ***\n")

    print(f"\ntime spent:\n")

    timing = {}
    timing["dataset construction"]       = time.strftime("%Hh:%Mm:%Ss", time.gmtime(t1-t0))
    timing["target LM model training"]   = time.strftime("%Hh:%Mm:%Ss", time.gmtime(t2-t1))
    timing["translation model training"] = time.strftime("%Hh:%Mm:%Ss", time.gmtime(t3-t2))
    timing["prediction generation"]      = time.strftime("%Hh:%Mm:%Ss", time.gmtime(t4-t3))

    print(tabulate(timing.items(), headers={"total", time.strftime("%Hh:%Mm:%Ss", time.gmtime(t4-t0))}))




