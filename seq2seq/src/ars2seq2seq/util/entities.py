"""
Routines for dealing with entities
"""

from ars2seq2seq.util.vocab import process_sentence
import re

special_toks = set(["cooking", "idle", "open", "suspended", "setup", "energized"])  # Left true and false and boolean, as these are significant types

tabu_toks = set(["true", "false"])

def is_entity(tok):
    if tok.lower() in tabu_toks:
        return False
    return tok.startswith("id__") or ("_" in tok) or tok.isupper() or tok in special_toks


def is_number(tok):
    if tok.isdigit():
        return True
    return False

def normalize_sal_entities(input1, input2, ent_prefix="id__"):
    """ Get the entity IDs in order, and issues new in-sequence IDs."""
    eid = 0
    toks1 = process_sentence(input1)
    toks2 = process_sentence(input2)
    oldent2newent = {}
    # First pass scan for entities
    for tok1 in toks1:
        if is_entity(tok1):
            if tok1 not in oldent2newent:
                new_entity = "{}{}".format(ent_prefix, eid)
                eid += 1
                oldent2newent[tok1] = new_entity
        elif is_number(tok1):
            if tok1 not in oldent2newent:
                new_entity = "number__{}".format(eid)
                eid += 1
                oldent2newent[tok1] = new_entity
    # Replace in lang1 and lang2
    new_toks1 = [oldent2newent[t] if t in oldent2newent else t for t in toks1]
    new_toks2 = [oldent2newent[t] if t in oldent2newent else t for t in toks2]
    rlookup = {}
    for oldent, newent in oldent2newent.items():
        rlookup[newent] = oldent
    return " ".join(new_toks1), " ".join(new_toks2), rlookup


#
# Code for placeholders, where strings start with an alphabetical character and
# # suffixed with "$DIGIT+" are treated as argument placeholders.
# We reorder these by lexical order.
#
PLACEHOLDER_SPECIAL = "$PLACEHOLDER$"  # placeholder special, used to mark location to insert

def is_numbered_placeholder(tok):
    """ Determines if the given token ends with a '_$DIGIT'.
    If the token is bracketed by double-quotes (for JSON), does this
    check within as well."""
    dpat = '^_(.+_)\d+$'  # Starts with "_", then any stuff, then _ and ends with digits
    is_quoted = False
    tok_txt = tok.strip()
    if tok_txt.startswith('"') and tok_txt.endswith('"'):
        is_quoted = True
        tok_txt = tok_txt[1:-1]
    m = re.search(dpat, tok_txt)
    if m is not None:
        if is_quoted:
            return '"' + m.groups()[0] + "{}" + '"'
        return m.groups()[0] + "{}"
    return None

def get_split(token):
    parts = token.split('#',1)
    if isinstance(parts,list) and len(parts)==2:
        return [parts[0],("#"+parts[1])]
    else:
        return [token,""]

def subsfirst(token, entmap):
    [stem,suffix] = get_split(token)
    stem_with_underscore = "_" + stem
    result = ""
    if stem in entmap:
        return entmap[stem_with_underscore] + suffix
    else:
        return stem+suffix
    
def reorder_numbered_placeholders(input1, input2, by_group=True):
    """ Given a stream, identifies placeholders that are suffixed by '_$DIGITS'.  These are considered to be
    argument placeholders, and will be renumbered with those prefixes in place.
    If we are reordering by group, numbering is done by counts in group"""
    eid = 0
    toks1 = process_sentence(input1)
    toks2 = process_sentence(input2)
    oldent2newent  = {} # map from old entities (incl "_" 1st character) to new entities (excl "_")
    group_hist = {}
    # First pass scan for entities.  Add in replacements 
    for tok1 in toks1:
        matched = is_numbered_placeholder(tok1)
        if matched is not None: # tok1 of the form _XXX_DDD
            if tok1 not in oldent2newent:
                if by_group:
                    if matched not in group_hist:
                        group_hist[matched] = 0
                    next_id = group_hist[matched]
                    group_hist[matched] += 1
                else:
                    next_id = eid
                new_entity = matched.format(next_id)
                eid += 1
                oldent2newent[tok1] = new_entity
    # Replace in lang1 and lang2
    new_toks1 = ["_" + oldent2newent[t] if t in oldent2newent else t for t in toks1]
    new_toks2 = [subsfirst(t,oldent2newent) for t in toks2]
    rlookup = {}
    # Computing the reverse map (for decoding renumbering back CST entities at runtime)
    for oldent, newent in oldent2newent.items():
        rlookup["_" + newent] = oldent[1:]
    return " ".join(new_toks1), " ".join(new_toks2), rlookup


def reinsert_from_lookup(gen_toks, rlookup):
    """ Given the generated token sequence, applies and replaces entity placeholders with their lookup values.
    Because we are now emitting to JSON directly, look for quotes around the term.  If this is present,
    we strip the bracketing quotes and do the lookup that way."""
    ret = []
    for gen_tok in gen_toks:
        if gen_tok.startswith('"') and gen_tok.endswith('"'):
            inner_tok = gen_tok[1:-1]
            ret.append("\"{}\"".format(subsfirst(inner_tok,rlookup)))
        else:
            ret.append("{}".format(subsfirst(gen_tok,rlookup)))
    return ret


def sanity_check1():
    input1 = "If id__3 is set to id__1 then activate id__2 and TRUE and microwave_mode."
    input2 = "{ node:type-test arg5: microwave_mode arg1: id__2  arg2: id__1 arg3: id__3 arg4: TRUE }"
    updated_input1, updated_input2, lookup = normalize_sal_entities(input1, input2, ent_prefix="id__")
    print(updated_input1)
    print(updated_input2)
    print(lookup)

def sanity_check2():
    input1 = "If entity3 is set to value1 then activate signal2 and TRUE and status4."
    input2 = "{ node:type-test arg5: status4 arg1: signal2  arg2: value1 arg3: entity3 arg4: TRUE }"
    updated_input1, updated_input2, lookup = reorder_numbered_placeholders(input1, input2)
    print(updated_input1)
    print(updated_input2)
    print(lookup)

def sanity_check3():
    input1 = "If entity3 is set to value1 then activate signal2 and TRUE and status4."
    input2 = "{ node:type-test arg5: \"status4\" arg1: \"signal2\"  arg2: value1 \"arg3\": \"entity3\" arg4: TRUE }"
    updated_input1, updated_input2, lookup = reorder_numbered_placeholders(input1, input2)
    print(updated_input1)
    print(updated_input2)
    print(lookup)

if __name__ == "__main__":
    print("Sanity 1")
    sanity_check1()
    print("\n\nSanity 2")
    sanity_check2()
    print("\n\nSanity 3")
    sanity_check3()

