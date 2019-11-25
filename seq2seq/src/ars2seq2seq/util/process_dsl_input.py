import os
import sys
import json
import traceback
import collections

TABU_KEYS = set(["nl"])

def copy_except(obj, tabu_keys):
    """
    Copies the dictionary, removing elements with the given tabu keys
    :param in_dict:
    :param tabu_keys:
    :return:
    """
    if isinstance(obj, collections.Mapping):
        return {k: copy_except(v, tabu_keys) for k, v in obj.items() if not(k in tabu_keys)}
    elif isinstance(obj, list):
        return [copy_except(child_obj, tabu_keys) for child_obj in obj]
    else:
        return obj


def dict2txt(obj):
    """ Converts the desired dict based representation into a space delimited form."""
    accum = []
    if isinstance(obj, collections.Mapping):
        accum.append("{")
        for k, v in obj.items():
            accum.append(k)
            accum.append(":")
            accum.extend(dict2txt(v))
        accum.append("}")
    elif isinstance(obj, list):
        accum.append("[")
        for child_obj in obj:
            accum.extend(dict2txt(child_obj))
        accum.append("]")
    else:
        accum.append(obj)
    return [str(x) for x in accum]

def txt2dict(strform):
    """  The opposite of dict2txt """
    ret = []
    syntax_toks = set(["{", "}", "[", "]", ":"])
    toks = strform.split(" ")
    comma_stack = 0
    for tok in toks:
        if tok not in syntax_toks:
            ret.append("\"{}\"".format(tok))
        else:
            ret.append(tok)
        if tok == "}" or tok == "]":
            ret.append(",")
            comma_stack = 0
        elif tok not in syntax_toks:
            if len(ret) >=2 and ret[-2] == ":":
                ret.append(",")
    return " ".join(ret)


def process_dsl_txtfile(fname, output_fname="output/eng-cst.txt", TOK_LIMIT=None):
    with open(output_fname, "w") as fout:
        with open(fname, "r") as f:
            for line in f:
                if len(line) > 0 and not(line.startswith("Generating")):
                    try:
                        dict = json.loads(line)
                        nl = dict['nl']
                        if not(TOK_LIMIT is None) and len(nl.split(" ")) > TOK_LIMIT:
                            continue  # Skip longer NL lines
                        dict2 = copy_except(dict, TABU_KEYS)
                        str_form = " ".join(dict2txt(dict2))
                        fout.write("{}\t{}\n".format(nl, str_form))
                    except:
                        err_class, err_type, tb = sys.exc_info()
                        print("{}, {}".format(err_class, err_type))
                        traceback.print_tb(tb)


def process_json_listfile(fname, output_fname="output/eng-cst.txt", TOK_LIMIT=None):
    """
    If the source is a JSON list, take each dict entry in the list
    and treat that as a pair source
    :param fname:
    :param output_fname:
    :return:
    """
    with open(fname, "r") as f:
        elts = json.load(f)

    with open(output_fname, "w") as fout:
        for dict in elts:
            nl = dict['nl']
            if not (TOK_LIMIT is None) and len(nl.split(" ")) > TOK_LIMIT:
                continue  # Skip longer NL lines
            dict2 = copy_except(dict, TABU_KEYS)
            str_form = " ".join(dict2txt(dict2))
            fout.write("{}\t{}\n".format(nl, str_form))


if __name__ == "__main__":
    custom_limit=30
    input_file = sys.argv[1]
    print("Processing file={}".format(input_file))
    if input_file.endswith(".json"):
        process_json_listfile(input_file, "eng-cst.txt", TOK_LIMIT=custom_limit)
    else:
        process_dsl_txtfile(input_file, "eng-cst.txt", TOK_LIMIT=custom_limit)
