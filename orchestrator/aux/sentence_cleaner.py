# hook to define any cleaning operations on entity-processed sentences
# (This can for example be used to remove content in parentheses,
# substitute html tags, etc.)
def clean_input(sentences):
    return sentences