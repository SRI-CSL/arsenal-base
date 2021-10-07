import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("-score_dir", type=str, default="../../../eval/", help="location of the scoring data")
parser.add_argument("-old_score_file", default="lm_scores_2021-07-30T0004_gpt2_test_scores.txt", type=str, help="file containing old scores")
parser.add_argument("-new_score_file", default="lm_scores_2021-09-30T1326_gpt2_test_scores.txt", type=str, help="file containing new scores")

args = parser.parse_args()

score_dir = args.score_dir
old_scores_filename = args.old_score_file
new_scores_filename = args.new_score_file

scores = {}
with open(os.path.join(score_dir, old_scores_filename)) as f:
    lines = f.readlines()
    print(f"{len(lines)} scores in old set")
    for line in lines:
        score, sentence = line.split("\t")
        sentence = sentence[:-1]
        score = float(score)
        scores[sentence] = {"old": score}

with open(os.path.join(score_dir, new_scores_filename)) as f:
    lines = f.readlines()
    print(f"{len(lines)} scores in new set")
    for line in lines:
        score, sentence = line.split("\t")
        sentence = sentence[:-1]
        score = float(score)
        if sentence in scores:
            scores[sentence]["new"] = score
        else:
            scores[sentence] = {"new": score}

# print(json.dumps(scores, indent=3))

old_only_cnt = 0
new_only_cnt = 0
both_cnt = 0

same = {}
decreased = {}
increased = {}


for sentence, scores in scores.items():
    if len(scores) ==1:
        if "old" in scores:
            old_only_cnt += 1
        elif "new" in scores:
            new_only_cnt += 1
        else:
            print(scores)
    elif len(scores) == 2:
        both_cnt += 1

        old = scores["old"]
        new = scores["new"]
        if old == new:
            same[sentence] = scores
        elif old < new:
            increased[sentence] = scores
        else:
            decreased[sentence] = scores


decreased = dict(sorted(decreased.items(), key=lambda x: x[1]["old"] - x[1]["new"]))
increased = dict(sorted(increased.items(), key=lambda x: x[1]["new"] - x[1]["old"]))


with open(os.path.join(score_dir, "decreased.json"), "w") as f:
    json.dump(decreased, f, indent=3)

with open(os.path.join(score_dir, "increased.json"), "w") as f:
    json.dump(increased, f, indent=3)

with open(os.path.join(score_dir, "same.json"), "w") as f:
    json.dump(same, f, indent=3)


print(f"{both_cnt} scored in both, {old_only_cnt} only in old dataset, {new_only_cnt} only in new dataset")



