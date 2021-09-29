import re

substitutions = {
    "&lt;":     "<",
    "&gt;":	    ">",
    "&amp;":    "&",
}

# remove all content in parantheses
def remove_parentheses(line: str):
    return re.sub('\([^)]*\)', '', line)

# remove all remaining html tags
def remove_tags(line: str):
    return re.sub('<.*?>', '', line)

# substitute some escaped html characters with their plain-text equivalents
def subsitute_syms(line : str):
    for orig, replacement in substitutions.items():
        line = line.replace(orig, replacement)
    return line

def clean_input(line :str):
    line = remove_tags(line)
    line = remove_parentheses(line)
    line = subsitute_syms(line)
    return line



