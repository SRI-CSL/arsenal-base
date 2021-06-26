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
    if op == "None":
        return ["null",i+1]
    if op == "Some":
        return convert_aux(toks,i+1)
    if op == "Nil":
        return ["[]",i+1]
    if op == "true" or op == "false":
        return [op,i+1]
    result = []
    result.append("{{ \"node\": \"{}\", ".format(op))
    if hastype:
        result.append("\"type\": \"{}\", ".format(parts[1]))
    result.append("\"subtrees\": [")
    if op == "List":
        result = ["["]
    k = i+1
    for j in range(arity):
        [arg,k] = convert_aux(toks,k)
        if j == 0:
            result.append(arg)
        else:
            result.append(", {}".format(arg))
    result.append(" ]")
    if op != "List":
        result.append(" }")
    return ["".join(result),k]

# def convert_pn2json(pn_str):
#     """Convert polish notation output to JSON"""
#     print("Converting PN to JSON: {}".format(pn_str))
#     toks = pn_str.split()
#     [result,_] = convert_aux(toks,0)
#     return result

def convert_pn2json(pn_str):
    """Convert polish notation output to sexp as JSON"""
    print("Converting PN to SEXP as JSON: {}".format(pn_str))
    toks = pn_str.split()
    sexp = ""
    expected_args = []  # stack containing number of expected args at each level
    for tok in toks:
        [op,arity] = get_pn_op_arity(tok)
        parts = op.split(':')
        op = parts[0]
        if arity > 0:
            # if get_op(op) == 'Some':  # ignore 'Some'
            #     continue
            sexp += '[ "' + op + '", '
            expected_args.append(arity)
        else:
            # test = get_op(op)
            # if test == 'None' or test == 'true' or test == 'false' or test == 'Nil':
            #     sexp += op + " "
            # else:
            # sexp += "( " + op + " ) "
            sexp += '"' + op + '" '
            while expected_args:
                exp = expected_args.pop()
                exp -= 1
                if exp > 0:
                    sexp += ", "
                    expected_args.append(exp)
                    break
                else:  # exp==0
                    sexp += "] "
    # Double-check that the S-expression is balanced
    lparen_count = len([tok for tok in sexp.split() if tok=='['])
    rparen_count = len([tok for tok in sexp.split() if tok==']'])
    if lparen_count != rparen_count:
        print("Warning: Mis-matched S-expression!")
    return sexp

def convert_pn2pn(pn_str):
    """Convert polish notation output to sexp as JSON"""
    print("Converting PN to PN as JSON: {}".format(pn_str))
    toks = pn_str.split()
    json = "["
    isfirst = True
    for tok in toks:
        if not isfirst:
            json += ", "
        isfirst = False
        json += '"' + tok + '"'
    json += " ] "
    return json

def convert_pn2sexp(pn_str):
    """Convert polish notation output to sexp"""
    print("Converting PN to SEXP: {}".format(pn_str))
    toks = pn_str.split()
    sexp = ""
    expected_args = []  # stack containing number of expected args at each level
    for tok in toks:
        [op,arity] = get_pn_op_arity(tok)
        if arity > 0:
            if get_op(op) == 'Some':  # ignore 'Some'
                continue
            sexp += "( " + op + " "
            expected_args.append(arity)
        else:
            test = get_op(op)
            if test == 'None' or test == 'true' or test == 'false' or test == 'Nil':
                sexp += op + " "
            else:
                sexp += "( " + op + " ) "
            while expected_args:
                exp = expected_args.pop()
                exp -= 1
                if exp > 0:
                    expected_args.append(exp)
                    break
                else:  # exp==0
                    sexp += ") "                    
    # Double-check that the S-expression is balanced
    lparen_count = len([tok for tok in sexp.split() if tok=='('])
    rparen_count = len([tok for tok in sexp.split() if tok==')'])
    if lparen_count != rparen_count:
        print("Warning: Mis-matched S-expression!")
    return sexp


