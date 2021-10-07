import os

import datasets
import matplotlib.pyplot as plt

real_dir = "../../../../source-documents/extracted/"
gen_dir = "../../../large_files/datasets/2021-09-30T1326/"
tmp_dir = "./tmp/"

file_suffix = "_sentences.txt"

## we shouldn't collect source documents automatically (at least not in this simplistic way)
## because we have some documents with multiple versions, which would impact the statistics
# for (_, _, filenames) in os.walk(json_dir):
#     break
# filenames = [f for f in filenames if f.endswith(".json")]
#

filenames = ['21905-h00.json', '29503-h20.json', '33102-g00.json', '33220-h00.json', '33402-g00.json', '33501-g50.json']

num_docs = len(filenames)
for i, filename in enumerate(filenames):
    os.system(f" jq -r '.. | select(.node_type? == \"sentence\").content ' "
              f"{os.path.join(real_dir, filename)} > "
              f"{os.path.join(tmp_dir, filename.replace('.json', file_suffix))}")

for (_, _, filenames) in os.walk(tmp_dir):
    break
filenames = [f for f in filenames if f.endswith(file_suffix)]


real_sentences = []
for i, filename in enumerate(filenames):
    with open(os.path.join(tmp_dir, filename), "r") as f:

        filename = filename[:9]
        print(f"\nreading {filename}")
        real_sentences += f.read().splitlines()

gen_sentences = []
with open(os.path.join(gen_dir, "eng-pn.val.txt")) as f:
    for line in f.readlines():
        sentence, _ = line.split("\t")
        gen_sentences.append(sentence)

def create_histogram(sentences):
    histogram = {}
    max_len = 0
    longest = ""
    for sentence in sentences:
        curr_len = len(sentence.split(" "))

        if curr_len in histogram:
            histogram[curr_len] += 1
        else:
            histogram[curr_len] = 1

        if curr_len > max_len:
            max_len = curr_len
            longest = sentence
    print(f"longest sentence has length {max_len}:\n{longest}")
    return histogram

real_histogram = create_histogram(real_sentences)
gen_histogram = create_histogram(gen_sentences)

fig, axs = plt.subplots(2)
fig.suptitle('Sentence length distributions')


def plot_histogram(histogram, axis, color, legend, xlabel=None):

    bars = list(histogram.keys())
    bars.sort()
    y = [histogram[b] for b in bars]
    axis.bar(bars, y, align="center", color=color)
    axis.set_ylabel("count")
    axis.set_xlabel(xlabel)
    axis.legend(legend)
    axis.set_xlim(0,80)

plot_histogram(real_histogram, axs[0], "blue", ["selection from standard documents"])
plot_histogram(gen_histogram, axs[1], "red", ["artificially generated validation data set"], "length (#words)")


plt.show()
