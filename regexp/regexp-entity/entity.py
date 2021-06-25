import re
import json

string_prefix = '_REgrammar/String_'
string_regexp = re.compile(r'("[^"]+")')

char_prefix = '_REgrammar/Char_'
char_regexp = re.compile(r"('.')")

def replace_all(prefix,regexp,text):
    substs = {}
    new_text = text
    idx = 0
    for m in re.finditer(regexp, text):
        match = m.group(1)
        placeholder = "{}{:03}".format(prefix,idx)
        idx += 1
        substs[placeholder] = match
        new_text = new_text.replace(match, placeholder)
    return new_text, substs

def process(sentence):
    text = sentence['text']
    new_text, substs = replace_all(string_prefix, string_regexp, text)
    final_text, substs2 = replace_all(char_prefix, char_regexp, new_text)
    final_subst = { **substs, **substs2 }

    result = { 
        'id': sentence['id'],
        'orig-text': text, 
        'new-text': final_text,
        'substitutions': final_subst
    }

    print("\nResult:")
    print(json.dumps(result, indent=2))
    return result

def process_all(sentences):
    results = list(process(sentence) for sentence in sentences)
    return {
        'sentences': results
    }
