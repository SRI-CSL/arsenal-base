import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("-score_dir", type=str, default="../../../eval/", help="location of the scoring data")
parser.add_argument("-old_score_file", default="lm_scores_2021-07-30T0004_gpt2_test_scores.txt", type=str, help="file containing old scores")
parser.add_argument("-new_score_file", default="lm_scores_2022-01-04T1033_gpt2_test_scores.txt", type=str, help="file containing new scores")

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


for sentence, results in scores.items():
    if len(results) ==1:
        if "old" in results:
            old_only_cnt += 1
        elif "new" in results:
            new_only_cnt += 1
        else:
            print(results)
    elif len(results) == 2:
        both_cnt += 1

        old = results["old"]
        new = results["new"]
        if old == new:
            same[sentence] = results
        elif old < new:
            increased[sentence] = results
        else:
            decreased[sentence] = results


scores = dict(sorted(scores.items(), key=lambda x: x[1]["new"] - x[1]["old"]))
decreased = dict(sorted(decreased.items(), key=lambda x: x[1]["old"] - x[1]["new"], reverse=True))
increased = dict(sorted(increased.items(), key=lambda x: x[1]["new"] - x[1]["old"], reverse=True))



# with open(os.path.join(score_dir, "decreased.json"), "w") as f:
#     json.dump(decreased, f, indent=3)
#
# with open(os.path.join(score_dir, "increased.json"), "w") as f:
#     json.dump(increased, f, indent=3)
#
# with open(os.path.join(score_dir, "same.json"), "w") as f:
#     json.dump(same, f, indent=3)

with open(os.path.join(score_dir, "score_comparison.json"), "w") as f:
    json.dump(scores, f, indent=3)


with open(os.path.join(score_dir, "score_comparison.txt"), "w") as f:
    f.write("diff\told s\tnew s\tsentence\n")

    for sent, results in scores.items():
        diff = f"{results['new']-results['old']:6.3f}"
        s1 = f"{results['old']:6.3f}"
        s2 = f"{results['new']:6.3f}"

        f.write(f"{diff}\t{s1}\t{s2}\t{sent}\n")


print(f"{both_cnt} scored in both, {old_only_cnt} only in old dataset, {new_only_cnt} only in new dataset")



