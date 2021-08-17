import argparse
import os
import re

from tabulate import tabulate

parser = argparse.ArgumentParser(description="Building the translation model form NL to AST")
parser.add_argument("-eval_file",  type=str,      default="../arsenal/large_files/datasets/lm/2021-06-28/spec_sentences.txt",         help="location of the test data directory")

args = parser.parse_args()
print(tabulate(vars(args).items(), headers={"parameter", "value"}))
eval_file = args.eval_file

def clean_eval_set(in_file):
    with open(in_file, "r") as f:
        instances = f.read().splitlines()

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
            if len(i.split()) < 5:
                discarded.append(i)
            else:
                cleaned.append(i)

    return cleaned, discarded

cleaned, discarded = clean_eval_set(eval_file)

cleaned.sort(key = lambda x: len(x.split()))
discarded.sort(key = lambda x: len(x.split()))

path = os.path.split(eval_file)[0]

with open(os.path.join(path, "spec_sentences_clean.txt"), "w") as f:
    for i in cleaned:
        f.write(i + "\n")

with open(os.path.join(path, "spec_sentences_discarded.txt"), "w") as f:
    for i in discarded:
        f.write(i + "\n")
