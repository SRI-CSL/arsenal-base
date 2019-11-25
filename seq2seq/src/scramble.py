import random
import sys


if __name__ == "__main__":
    input_file = sys.argv[1]
    tgt_file = sys.argv[2]    
    print("Processing file={}".format(input_file))
    lines = open(input_file).readlines()
    random.shuffle(lines)
    with open(tgt_file, "w") as f:
        f.writelines(lines)
