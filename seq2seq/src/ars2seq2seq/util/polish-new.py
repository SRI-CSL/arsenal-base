import re

def get_pn_op_arity(token):
    parts = re.split('#',token)
    if isinstance(parts,list) and len(parts)==2:
        [op,arity_str] = parts
        arity = int(arity_str)
        return [op,arity]
    else:
        if isinstance(parts,list) and len(parts)==3:
            [op,arity_str,nodetype] = parts
            arity = int(arity_str)
            return [op + ":" + nodetype,arity]
        else:
            return None

def get_typedpn_op_arity(token):
    parts = re.split('#',token)
    if isinstance(parts,list) and len(parts)>=2:
        return [parts[0],len(parts)-2]
    else:
        return None

def get_op(token):
    parts = re.split(':',token)
    if isinstance(parts,list):
        return parts[0]
    else:
        return token

def convert_aux(toks,i):
    tok = toks[i]
    [op,arity] = get_pn_op_arity(tok)
    parts = op.split(':')
    if isinstance(parts,list) and len(parts)>=2:
        [ op, hastype ] = [ parts[0], True ]
    else:
        hastype = False
    result = ["{ "]
    if hastype:
        result.append("\"type\": \"{}\", ".format(parts[1]))
    k = i+1
    if op == "Some":
        [arg,k] = convert_aux(toks,k)
        result.append("\"subtrees\": {} }}".format(arg))
        return ["".join(result),k]
    if op == "None":
        result.append("\"subtrees\": null }")
        return ["".join(result),k]
    if op != "Nil" and op != "List":
        result.append("\"node\": \"{}\"".format(op))
        if op == "Nil" or arity > 0:
            result.append(", ")
    if op == "Nil" or arity > 0:
        result.append("\"subtrees\": [ ")
    for j in range(arity):
        [arg,k] = convert_aux(toks,k)
        if j == 0:
            result.append(arg)
        else:
            result.append(", {}".format(arg))
    if op == "Nil" or arity > 0:
        result.append(" ]")
    result.append(" }")
    return ["".join(result),k]

def convert_pn2json(pn_str):
    """Convert polish notation output to JSON"""
    print("Converting PN to JSON: {}".format(pn_str))
    toks = pn_str.split()
    [result,_] = convert_aux(toks,0)
    return result
