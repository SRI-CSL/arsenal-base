import collections
import json

def dict2txt(obj):
    """ Converts the desired dict based representation into a space delimited simplified form.
    This is the opposite of gen_str2dict"""
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


SYNTAX_TOKS = set(["{", "}", "[", "]", ":"])
CTRL_TOKS = set(["<EOS>", "<SOS>"])

from queue import Queue

def convert2queue(toks):
    if isinstance(toks, str):
        toks = toks.split(" ")
        toks_q = Queue()
        for tok in toks:
            toks_q.put(tok)
        toks = toks_q
    elif isinstance(toks, list) or isinstance(toks, tuple):
        toks_q = Queue()
        for tok in toks:
            toks_q.put(tok)
        toks = toks_q
    return toks

def traverse(toks):
    """ First traversal through the docs, converting it into a tree form.
     Brackets are included in the child form."""
    ret = []
    toks = convert2queue(toks)
    while not(toks.empty()):
        tok = toks.get()
        if tok == "{" or tok == "[":
            ret.append([tok] + traverse(toks))
        elif tok == "}" or tok == "]":
            ret.append(tok)
            return ret
        else:
            ret.append(tok)
    return ret[0]

def chunk(toks):
    ret = []
    idx = 0
    while idx < len(toks):
        if idx < (len(toks) - 2) and toks[idx + 1] == ":":
            ret.append(" ".join(toks[idx:idx+3]))
            idx += 3
        else:
            ret.append(toks[idx])
            idx += 1
    return ret

def gen_toks2dict(toks, unquote_digits=True, unquote_boolean=False):
    """
    Converts the simplified json output from the seq2seq model into a dict.
    This is the opposite.
    :param toks:
    :return:
    """
    ret = []
    toks = convert2queue(toks)
    while not (toks.empty()):
        tok = toks.get()
        if tok == "{" or tok == "[":
            child_form = gen_toks2dict(toks)
            ret.append(tok + child_form)
        elif tok == "}" or tok == "]":
            ret = chunk(ret)
            return ",".join(ret) + tok
        else:
            if tok in CTRL_TOKS:
                pass
            elif unquote_digits and tok.isdigit():
                ret.append(tok)
            elif unquote_boolean and tok.lower() in set(["true", "false"]):
                ret.append(tok.lower().title())
            elif tok not in SYNTAX_TOKS:
                ret.append("\"{}\"".format(tok))
            else:
                ret.append(tok)
    ret = chunk(ret)
    return " ".join(ret)



if __name__ == "__main__":
    x = {"a": "b", "c": {"d": "e", "f": ["g", "h", "i", ], }, "j": ["k", ], }
    print(x)
    json.loads("{ \"a\" : \"b\" }")
    json.loads('{ "a" : "b" , "c" : { "d" : "e" , "f" : [ "g" , "h" , "i"  ]  } , "j" : [ "k"  ]  }')
    print("Past JSON loads sanity checks")

    src = { "a" : "b",
            "c": { "d": "e",
                   "f": ["g", "h", "i"]},
            "j":["k"]
            }

    #test = {"node-type" : "top-theorem","expr" : {"node-type" : "always-formula","expr" : {"node-type" : "nary-or-formula","exprs" : [{"node-type" : "predicate-assertion","left" : {"node-type" : "name-term","value" : "cavity_magnetron_mode"},"right" : {"node-type" : "arith-and-predications","exprs" : [{"node-type" : "inequality",
    # "op" : {"node-type" : "not-equal", "value" : "", "!=" },"right" : {"node-type" : "name-term","value" : "ENERGIZED"}}]}},
    # {"node-type" : "predicate-assertion","left" : {"node-type" : "name-term","value" : "door_closed_sensor"},"right" : {"node-type" : "true-value","value" : True}}]}}}

    tok_form = dict2txt(src)
    str_form = " ".join(tok_form)
    print(str_form)
    tok_tree = traverse(str_form)
    print(tok_tree)
    print(gen_toks2dict(str_form))
    print("passed last sanity")


