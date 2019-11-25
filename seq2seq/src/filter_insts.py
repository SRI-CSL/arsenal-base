#
# Test, loads in the train/eng-dsl.txt, and pares down training pairs that are >= limit.
#
# LIMIT=20 # Good
# LIMIT=50 # Goes wonky
# LIMIT=35 # Good
# LIMIT = 42 # Goes wonky
# LIMIT = 38 # Not good any longer
LIMIT = 30

def process(root_dir):
    tok_hist = {}
    with open("./{}/eng-ast.txt".format(root_dir), "w") as r_out:
        with open("./{}/eng-ast.orig.txt".format(root_dir), "r") as r_in:
            for line in r_in:
                nl, ast = line.strip().split("\t")
                nl_toks = nl.split(" ")
                num_toks = len(nl_toks)
                tok_hist[num_toks] = tok_hist.get(num_toks, 0) + 1
                if len(nl_toks) <= LIMIT:
                    r_out.write(line)

    in_order = sorted(list(tok_hist.items()), key=lambda x: x[0])
    for length, count in in_order:
        print("{}\t{}".format(length, count))


process("data")
