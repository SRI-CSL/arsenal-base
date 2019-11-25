# Code snippet from https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/7
# Returns the GPU with the most free memory
import os
import numpy as np
import random

def get_freer_gpu():
    tmp_suffix = random.randint(10000,99999)
    tmp_fname = "tmp.{}.txt".format(tmp_suffix)
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{}".format(tmp_fname))
    memory_available = [int(x.split()[2]) for x in open(tmp_fname, 'r').readlines()]
    os.system("rm {}".format(tmp_fname))
    return np.argmax(memory_available)
