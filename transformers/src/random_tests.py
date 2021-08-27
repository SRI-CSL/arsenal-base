import json
import math
import os

in_dir = "/Users/martiny/Projects/Effigy/GGGGG/arsenal/large_files/models/transformers/eval/"
for (_, _, score_files) in os.walk(in_dir):
    break

score_files = [f for f in score_files if f.startswith("lm_scores") and f.endswith(".json")]



results = {}
for score_file in score_files:
    print(score_file)
    outfile = score_file.replace(".json", ".txt")
    with open(os.path.join(in_dir, score_file)) as f:
        scores = json.load(f)

    nan_scores = [s for s in scores if math.isnan(s["score"])]
    scores = [s for s in scores if not math.isnan(s["score"])]

    scores.sort(key=lambda entry: entry["score"])

    recorded = []

    with open(os.path.join(in_dir, outfile), "w") as f:
        for entry in nan_scores:
            if entry['sentence'].lower() not in recorded:
                recorded.append(entry['sentence'].lower())
                f.write(f"NaN\t{entry['sentence']}\n")

        for i in range(len(scores)):
            if i == 0:
                f.write(f"{scores[i]['score']}\t{scores[i]['sentence']}\n")
            else:
                try:
                    assert(scores[i]["score"] >= scores[i-1]["score"])
                except:
                    print(f"score mismatch ({i}): {scores[i]['score']}, previous: {scores[i-1]['score']}")

                sent = scores[i]['sentence'].strip()
                if sent.lower() not in recorded:
                    recorded.append(sent.lower())
                    f.write(f"{scores[i]['score']}\t{sent}\n")


