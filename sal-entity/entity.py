import spacy
import re
import json

nlp = spacy.load("en_core_web_sm")

entity_prefix = 'ID__'
number_prefix = 'NUMBER__'

# Add boolean etc
# Generator doesn't output anything like "when ID1 is ID2" or "when ID1 equals ID2"
# or "ID1 is equal to ID2" in transition conditions

ignore_words = [
    'it', 'true', 'TRUE', 'false', 'FALSE', 'input', 'output', 'transition',
    'definition', 'type', 'case', 'boolean', 'BOOLEAN', 'integer', 'INTEGER', 'integers',
    'field', 'record', 'sum', 'product', 'value', 'state', 'set'
]
strip_prefixes = [
    'the', 'The', 'a', 'A', 'an', 'An', 'of', 'that', 'type',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
]

def preprocess(text):
    return text.replace(".",". ").replace("-","_")

def is_entity(str):
    return (str.isupper() or re.match(r'^\w+_\w*$', str)) and str not in strip_prefixes and str not in ignore_words

def additional_entities(doc):
    return (tok.text for tok in doc if is_entity(tok.text))

def strip_prefix(chunk):
    stripped = chunk
    # Keep stripping prefixes as long as there may be more.
    # There could be more than one prefix, e.g. "the type ..."
    while True:
        for s in strip_prefixes:
            if stripped.startswith(s + ' '):
                stripped = stripped[len(s)+1:]
                continue
        break
    return re.sub(r'^\d+ ', "", stripped)  # Remove numeric prefixes

def generate_entity_name(chunk):
    return chunk.replace(" ", "_")

def replace_entity(chunk, placeholder_name, text):
    # These regexes are to make sure we are only replacing standalone words,
    # not substrings inside of other words
    res = re.sub(r'(^| ){}($|,|\.| )'.format(chunk), r'\1{}\2'.format(placeholder_name), text) 
    return res

def process_numbers(text):
    number_substs = {}
    new_text = text
    idx = 1
    doc = nlp(text)
    for tok in doc:
        #if tok.like_num:
        if re.match(r'^0|[1-9][0-9]*$', tok.text):
            placeholder_name = number_prefix + str(idx)
            idx += 1
            number_substs[placeholder_name] = tok.text
            new_text = replace_entity(tok.text, placeholder_name, new_text)
    return new_text, number_substs

def process(sentence):
    text = sentence['text']
    print("\nProcessing sentence: {}".format(text))
    cleaned_text = preprocess(text)
    doc = nlp(cleaned_text)
    print("\nRaw noun chunks:")
    for ch in doc.noun_chunks:
        print(ch)
    print("\nProcessed noun chunks:")
    chunks = set()
    for chunk in doc.noun_chunks:
        stripped_chunk = strip_prefix(chunk.text)
        if stripped_chunk not in ignore_words:
            chunks.add(stripped_chunk)
    chunks.update(additional_entities(doc))
    substs = {}
    new_text = cleaned_text
    idx = 1
    for chunk in chunks:
        if isinstance(chunk, str):
            print(chunk)
            placeholder_name = "PLACEHOLDER_" + str(idx)
            idx += 1
            entity_name = generate_entity_name(chunk)
            substs[placeholder_name] = entity_name
            new_text = replace_entity(chunk, placeholder_name, new_text)
        else:
            print("Warning: Non-string chunk: {}".format(chunk))

    # Reorder the entity IDs so that they follow word ordering
    idx = 1
    new_substs = {}
    while True:
        match = re.search(r'PLACEHOLDER_\d+', new_text)  # Find placeholders
        if (match):
            placeholder = match.group(0)
            new_placeholder = entity_prefix + str(idx)
            idx += 1
            entity_name = substs[placeholder]
            new_substs[new_placeholder] = entity_name
            new_text = replace_entity(placeholder, new_placeholder, new_text)
        else:
            break

    final_text, num_substs = process_numbers(new_text)
    final_substs = { **new_substs, **num_substs }

    result = { 
        'id': sentence['id'],
        'orig-text': text, 
        'new-text': final_text,
        'substitutions': final_substs
    }

    print("\nResult:")
    print(json.dumps(result, indent=2))
    return result

def process_all(sentences):
    results = list(process(sentence) for sentence in sentences)
    return {
        'sentences': results
    }
