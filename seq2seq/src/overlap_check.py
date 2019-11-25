import os

file1 = "./train_data/eng-ast.txt"
file2 = "./test_data/eng-ast.txt"

print("Running distinct instances overlap check between train and test")

def read_file(tgtpath):
    if not(os.path.isfile(tgtpath)):
        print("Warning, path={} is not a file".format(tgtpath))
    all_lines = []
    distinct_lines = set()
    with open(tgtpath, "r") as f:
        for line in f:
            all_lines.append(line)
            distinct_lines.add(line)
    print("{}, total={}, distinct={}".format(tgtpath, len(all_lines), len(distinct_lines)))
    return all_lines, distinct_lines


f1_lines, f1_distinct = read_file(file1)
f2_lines, f2_distinct = read_file(file2)

distinct_overlap = [x for x in f1_distinct if x in f2_distinct]
print("Total distinct overlap={}".format(len(distinct_overlap)))