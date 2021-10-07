import argparse
import collections
import sys
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizerFast

np.set_printoptions(threshold=sys.maxsize, suppress=True)

from arsenal_tokenizer import PreTrainedArsenalTokenizer

parser = argparse.ArgumentParser(description="Evaluate generated predictions")
parser.add_argument("-model_dir",       type=str,       default="../../../large_files/models/transformers/09-30-2021/translation_model",  help="location of the trained translation model")
parser.add_argument("-out_dir",         type=str,       default="../../../eval",  help="location of the trained translation model")
parser.add_argument("-prediction_file", type=str,       help="file with generated predictions (if none provided, the model dir is searched for a suitable file")

args = parser.parse_args()
model_dir = args.model_dir
prediction_file = args.prediction_file
out_dir = args.out_dir

if prediction_file is None:
    for (_, _, files) in os.walk(Path(model_dir).parent):
        break

    for file in files:
        if file.startswith("prediction"):
            prediction_file = os.path.join(Path(model_dir).parent, file)
            break
    if prediction_file is None:
        raise FileNotFoundError("no prediction file")

print(f"loading predictions from {prediction_file}")

dataset_properties = json.load(open(os.path.join(model_dir, "dataset_properties.json")))
target_vocab = dataset_properties["target_vocab"]
special_tokens = dataset_properties["special_tokens"]

tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)
source_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# collect per sentence length:
# - similarities (similar to edit distance)
# - accuracies
similarities = collections.defaultdict(list)
accuracies = collections.defaultdict(list)

# also collect confusions (i.e., information about how tokens got wrongly predicted)
confusions = {}

# 3 is the SEP token (which marks end of the sequence) - if the only difference between prediction
# and true sequence is the output of additional token(s) after the complete correct sequence has been
# predicted, the confusion lies in the SEP token, so we'll need to add this.
# (This doesn't seem to have noteworthy impact on our analysis results, we'll mainly add this to avoid
# key errors in rare occurrences.)
confusions[3] = {"total_correct": 0}

for v in target_vocab:
    # the tokenizer always creates sequences "[CLS] token [SEP]", so for a single "word" we only need to look at token 1
    true_token_id = tokenizer(v)['input_ids'][1]
    confusions[true_token_id] = {"total_correct": 0}


with open(prediction_file, "r") as f:
    instances = f.read().splitlines()

# count instances with UNK token
unk_cnt = 0

for i, instance in enumerate(instances):
    [source, truth, preds] = instance.split("\t", 2)
    preds = preds.split("\t")

    # The generation script was changed at some time to record lists of token ids instead of the actual NL sentence.
    # We make up for this by decoding the source sequences. This is in particular relevant because we are looking at
    # sentence lengths for the evaluation, and the token sequence length does not necessarily correspond to the nl
    # sentence length.
    if source.startswith("["):
        source = [int(x) for x in source[1:-1].split(",")]

        source = source[1:-1]
        nl_text = source_tokenizer.decode(source)
        l = len(nl_text.split())

    truth = [int(x) for x in truth[1:-1].split(", ")]
    predictions = []
    for predicted_token_id in preds:
        predictions.append([int(x) for x in predicted_token_id[1:-1].split(", ")])

    # cf. https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher
    # the similarity is somewhat of an odd measure b/c it is dependent on the order
    # of arguments
    # ratio is defined as 2*M/T with
    # - T total number of elements in  both sequences
    # - M number of matches
    s = SequenceMatcher(None, truth, predictions[0])
    similarities[l].append(s.ratio())

    # use this to count correct token occurrences below
    seq_len = (len(truth)) -1

    prediction = predictions[0]
    if truth != prediction:

        # uncomment to evaluate multiple alternative predictions (e.g., from typeforcing)
        # best = s.ratio()
        # for p in predictions[1:]:
        #     s2 = SequenceMatcher(None, truth, p)
        #     if s2.ratio() > best:
        #         best = s2.ratio()
        # # print(f"similarity of first prediction: {s.ratio():.3f}, best similarity: {best:.3f}, difference: {best-s.ratio():.3f}")

        # get the first token where true and predicted sequence deviate

        for p in range(len(truth)):
            if truth[p] != prediction[p]:
                seq_len = p
                break

        # seq_len = s.get_matching_blocks()[0].size
        true_token_id = truth[seq_len]
        predicted_token_id = predictions[0][seq_len]

        # the UNK token - this originates from problems with the dataset generation: there is some instance in the
        # validation set with a word that never appeared in the training set, and thus hasn't been tokenized
        if true_token_id == 1:
            unk_cnt += 1
            continue

        try:
            if predicted_token_id in confusions[true_token_id]:
                confusions[true_token_id][predicted_token_id] += 1
            else:
                confusions[true_token_id][predicted_token_id] = 1
        except:
            print(f"key error ({true_token_id}) for")
            print(truth)
            print(predictions[0])

    else:
        confusions[3]["total_correct"] += 1 # correctly predicted SEP token

    for true_token_id in truth[1:seq_len]:
        confusions[true_token_id]["total_correct"] += 1

    acc = sum(1 for x, y in zip(truth, predictions[0]) if x == y) / float(len(truth))
    accuracies[l].append(acc)

print(f"found {unk_cnt} instances with unknown tokens")
### produce a better-readable version of confusions ####

ids = list(tokenizer.id2vocab.keys())
ids.sort()
out_conf = {}

# tried to only focus on the "base word" of the grammar construct, but there are multiple terms that have
# the same base word followed by different properties
def baseword(token_id):
    word = tokenizer.id2vocab[token_id]
    parts = word.split("#")
    # return parts[0]
    return word

for true_id in confusions:
    true_word = baseword(true_id)
    inst_dict = {}
    inst_dict["total_correct"] = confusions[true_id]["total_correct"]
    # inst_dict["id"] = true_id
    for pred_id, count in sorted(confusions[true_id].items(), key=lambda item: item[1], reverse=True):
        if pred_id != "total_correct":
            # inst_dict[baseword(pred_id)] = {"id": pred_id, "count": count}
            inst_dict[baseword(pred_id)] = count
    out_conf[true_word] = inst_dict

with open(os.path.join(out_dir, f"confusions_{str(Path(model_dir).parent).split('/')[-1]}.json"), "w") as f:
    json.dump(out_conf, f, indent=3)


### analyze and plot similarity and accuracy ###

lengths = list(accuracies.keys())
lengths.sort()

total_avg_sim = sum([sum(similarities[x]) for x in lengths]) / len(instances)
total_avg_acc = sum([sum(accuracies[x]) for x in lengths]) / len(instances)
total_avg_bin_acc = sum([sum(i for i in accuracies[x] if i == 1) for x in lengths]) / len(instances)

print(f"total average similarity: {total_avg_sim}")
print(f"total average accuracy: {total_avg_acc}")
print(f"total average exact matches: {total_avg_bin_acc}")


avg_sim = [sum(similarities[x]) / len(similarities[x]) for x in lengths]
avg_acc = [sum(accuracies[x] ) / len(accuracies[x]) for x in lengths]
avg_bin_acc = [sum(i for i in accuracies[x] if i == 1) / len(accuracies[x]) for x in lengths]
counts = [len(accuracies[x]) for x in lengths]

fig, axs = plt.subplots(2)

axs[0].bar(lengths, avg_sim, align="center")
axs[0].bar(lengths, avg_acc, align="center")
axs[0].bar(lengths, avg_bin_acc, align="center")
axs[0].set_xlabel("length (#words)")
axs[0].set_ylabel("average per sentence length")
axs[0].set_title("comparison between true and predicted 'sentences' (CSTs in polish notation)")
axs[0].set_xlim(0,80)
axs[0].set_ylim(0,1)
axs[0].legend(["similarity", "accuracy", "exact matches"], loc="lower left")

axs[1].bar(lengths, counts, align="center")
axs[1].set_xlabel("length (#words)")
axs[1].set_ylabel("# instances")
axs[1].set_title("instance counts")
axs[1].set_xlim(0,80)

fig.suptitle(Path(model_dir).parent)
fig.set_size_inches(10, 10)

plt.savefig(os.path.join(out_dir,  f"eval_{str(Path(model_dir).parent).split('/')[-1]}.png"), bbox_inches="tight")
