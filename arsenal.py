#!/usr/local/bin/python3
"""
Using Arsenal to turn natural language descriptions of regular expressions in to actionable  regular expressions
Optionally, displaying the intermediate data formats - entities and CSTs
"""
import argparse
import sys
import re
import json
import requests
from functools import reduce

#TODO allow these to get set on the command line
HOST = "localhost"
PORT_NL = "8070"
PORT_EP = "8060"
PORT_REF = "8090"
#when not run with docker UI as the proxy (localhost:8080 with some other path info
EP_URL_TEMPLATE = "http://{}:{}/process_all"
NL_URL_TEMPLATE = "http://{}:{}/generateir"
REF_URL_TEMPLATE = "http://{}:{}/"

#
# Regular expressions used to clean up the input text
#
#pull out the first word of the line
FIRST_WORD_RE = re.compile('^(\S)+(.*)')
#all upper case letters
ALL_UPPER_RE = re.compile('^[A-z][A-z]+$')
#starts with one or more non digit, ends with a digit
ENTITY_RE = re.compile('^(\D+)\d+$')

def lowercaseFirstLetterOfFirstWord(line: str):
    fw_match = FIRST_WORD_RE.match(line)
    if fw_match:
        fw = fw_match.group(1)
        if ( ALL_UPPER_RE.match(fw) or ENTITY_RE.match(fw) ):
            return line
    #otherwise return the line with the first letter lower case (or blank if not)
    lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
    return lowerFirst(line)

def removeParens(line: str):
    return re.sub('[()]', '', line)

def cleanLine(line: str):
    if line:
        #cleanline = lowercaseFirstLetterOfFirstWord(removeParens(line.strip()))
        cleanline = lowercaseFirstLetterOfFirstWord(line.strip())
        #and finally, remove terminating period
        return re.sub('\.$', '', cleanline)

def importSentenceData(args):
    source_file = args.source_file
    # TODO: Could add file existence checking,etc
    sentencesData = {}
    sentenceList = []
    
    with open(source_file, 'r') as data:
    #load sentences in to list, clean up as required (remove parens, lower case first word, etc)
        i=0
        for line in data:
        #load all the non-empty lines in to a list of tuples with index
            c_line = cleanLine(line) #remove parens, lowercase start char and remove ending period
            if c_line:
                i+=1
            #make a dict for this sentences information
                sDict = {}
                sDict['id'] = "S" + str(i)
                sDict['text'] = c_line
                sentenceList.append(sDict)
    #set up dict for JSON format - might be empty
    sentencesData['sentences'] = sentenceList
    return sentencesData

#
# Call to the Arsenal Entity Process API
#
def callEntityProcessor(sentence_data):
    ep_url = EP_URL_TEMPLATE.format(HOST,PORT_EP)
    ep_input = json.dumps(sentence_data)
    response = requests.post(ep_url,data=ep_input,headers={"Content-Type": "application/json"})
    if (response.ok):
        return json.loads(response.text)
    #TODO how to handle failure gracefully?
    return sentence_data

#
# Call to the Arsenal Entity Process API
#
def callNl2CstProcessor(nl2cst_data):
    nl_url = NL_URL_TEMPLATE.format(HOST,PORT_NL)
    nl_input = json.dumps(nl2cst_data)
    response = requests.post(nl_url,data=nl_input)
    if (response.ok):
        return json.loads(response.text)
    #TODO how to handle failure?
    return nl2cst_data


#look at a list of dicts and find one where the 'id' matches
#if found, return the 'subsitutions' value from that same dict (if that key exist)
def getIDSubstitutions(data_id,subsitution_data):
    for this in subsitution_data:
        if (this.get('id') == data_id) and (this.get('substitutions')):
            return this['substitutions']
    return {}

# TODO - use map and function?
# We know a CST is json so only values, dicts and lists
def traverseAndAddSubstitute(val,substDict):
    keys = substDict.keys()
    if isinstance(val, dict):
        return {k: traverseAndAddSubstitute(v,substDict) for k, v in val.items()}
    elif isinstance(val, list):
        return [traverseAndAddSubstitute(elem,substDict) for elem in val]
    elif isinstance(val,str) and (val in keys):
        #not a list but a dict - but this works because it replaces a leaf
        newnode = { "placeholder": val, "text": substDict[val] }
        return newnode
    else:
       return val # no container, just values (str, int, float)

#This modifies the cst_data, ep_subsitutions
def embedEntityDefinitions(cst_data, ep_data):
    substitution_data= ep_data['sentences']
    cst_sentences = cst_data['sentences']
    for data in cst_sentences:
        data_id = data['id']
        data_cst_list = data['cst']
        substDict = getIDSubstitutions(data_id,substitution_data)
        if (substDict and data_cst_list):
            #modifies the data_cst in place
            newlist = traverseAndAddSubstitute(data_cst_list,substDict)
            #modify the sentence data to include the substitutions
            data['cst'] = newlist
    return cst_data

#
# Call Arsenal's reformulation utiltity 
# and then return only the string text from the reformulations list
#
def reformulateRules(cst_data):
    ref_url = REF_URL_TEMPLATE.format(HOST,PORT_REF)
    ref_input = json.dumps(cst_data)
    response = requests.post(ref_url,data=ref_input)
    if (response.ok):
        ref_data = json.loads(response.text)
        # sys.stdout.write("What we get back from reformulate:\n")
        # sys.stdout.write(json.dumps(ref_data,indent=2) + '\n')
        #this is a list of lists
        ref_lists = ["\n".join(d.get('reformulations')) for d in ref_data.get('sentences')]
        return "\n".join(ref_lists)
    return "Was not able to reformulate result"


###################REGEXP generation logic
DUBQUOTE_RE = re.compile('^"(.*)"$')
SINGQUOTE_RE = re.compile("^'(.*)'$")

def disquote(str):
    q_match = DUBQUOTE_RE.match(str)
    if q_match:
        return q_match.group(1)
    q_match = SINGQUOTE_RE.match(str)
    if q_match:
        return q_match.group(1)
    return str

terminalDict = {
  'empty': '',
  'word': '\\w',
  'any': '.',
  'digit': '\\d',
  'space': '\\s',
  'notword': '\\W',
  'notdigit': '\\D',
  'notspace': '\\S'
}

#
# When handling an entity substitution, the placeholder is subbed back in like this:
# 'subtrees': [
#    {'node': 
#       {'placeholder': 'Char_1', 'text': "'a'"}, 'type': 'entity<kchar>', 'subtrees': []}
#       {'placeholder': 'Char_2', 'text': "'b'"}, 'type': 'entity<kchar>', 'subtrees': []}
#
def createTerminal(cstnode):
    node_val = cstnode.get('node')
    if not node_val:
        return "Unknown node"
    node_val = node_val.lower()
    if node_val in terminalDict:
        return terminalDict.get(node_val)
    # From here we'll pull out any specific values
    subtrees = cstnode.get('subtrees')
    if subtrees and len(subtrees) > 0:
        subtree0_node = subtrees[0].get('node')
        subtree0_val = subtree0_node.get('text')
        if node_val == 'specific':
            return disquote(subtree0_val)
        elif node_val == 'characterrange' and len(subtrees) > 1:
            subtree1_node = subtrees[1].get('node')
            subtree1_val = subtree1_node.get('text')
            return '[' + disquote(subtree0_val) + '-' + disquote(subtree1_val) + ']';
    return("Unknown terminal node")

def createConcat(cstnode):
    if not isinstance(cstnode, list):
        return createRegex(cstnode)
    retval = reduce(lambda thisVal, nextVal: createRegex(thisVal) + createRegex(nextVal), cstnode)
    return retval

def createRegex(cstnode):
    node_val = cstnode.get('node')
    subtree0 = None
    subtrees = cstnode.get('subtrees')
    if subtrees and len(subtrees) > 0:
        subtree0 = subtrees[0]
    if not node_val or not subtree0:
        return "Was not able to generate a regular expression from this CST"
    node_val = node_val.lower()
    if node_val == 'terminal':
        return createTerminal(subtree0)
    elif node_val == 'startofline':
        return '(^' + createRegex(subtree0) + ')'
    elif node_val == 'endofline':
        return '($' + createRegex(subtree0) + ')'
    elif node_val == 'plus':
        return '(' + createRegex(subtree0) + '+)'
    elif node_val == 'star':
        return '(' + createRegex(subtree0) + '*)'
    elif node_val == 'or':
        subtree1 = subtrees[1]
        return '(' + createRegex(subtree0) + '|' + createRegex(subtree1) + ')'
    elif node_val == 'concat':
        #subtree0 should be a list of tree nodes
        return '(' + createConcat(subtree0) + ')'
    else:
        return "Was not able to identify CST node of {}".format(node_val)

def createRegexes(cst_data):
    sentence_list = cst_data['sentences']
    regexes = []
    for data in sentence_list:
        cst = data.get('cst')
        regexp = createRegex(cst)
        regexes.append(regexp)
    return regexes

####################


def parseArguments():
    # Create our Argument parser and set its description
    parser = argparse.ArgumentParser(
        description="Script that runs a file of sentences through nl2cst and produces a set of regular expressions in JSON with an option for human reformulation ",
    )

    # Add the arguments:
    #   - source_file: the source file we want to convert
    #   - dest_file: the destination where the output should go
    parser.add_argument('source_file',help='The location of the source ')
    parser.add_argument('-d','--dest_file',help='Location of dest file (default: stdout',default=None)
    parser.add_argument('-e','--entities',help='show the entity structure genenerated while processing ',action="store_true")
    parser.add_argument('-c','--cst',help='show the CSTs generated while processing',action="store_true")
    parser.add_argument('-r','--reformulate',help='reformat output for human validation ',action="store_true")

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArguments()

    sys.stdout.write("Arguments parsed\n")
    
    out_filename = args.dest_file

    # If the destination file wasn't passed, then use stdout
    if out_filename is None:
        outhandle = sys.stdout
    else:
        outhandle = open(out_filename,'w')
    
    # import and clean up the input text
    sentence_data = importSentenceData(args) 

    # call the API for the entity processor for 
    entity_processed_data = callEntityProcessor(sentence_data)

    sys.stdout.write("Entity processor done\n")

    if args.entities:
        outhandle.write("ENTITIES:\n")
        outhandle.write(json.dumps(entity_processed_data,indent=2) + '\n')

    # call the API for the language processor
    rules_data = callNl2CstProcessor(entity_processed_data)

    sys.stdout.write("Seq2seq done\n")

    # adds the definitions to the rules_data
    embedEntityDefinitions(rules_data, entity_processed_data)

    sys.stdout.write("Tree reconstruction done\n")

    if args.cst:
        outhandle.write("CSTs:\n")
        rulesText = json.dumps(rules_data,indent=2)
        outhandle.write(rulesText + '\n')        

    # if requested, generate human-readable output sentences
    if args.reformulate:
        rulesText = reformulateRules(rules_data)
        sys.stdout.write("Reformulation done\n")
        outhandle.write(rulesText + '\n')

    if outhandle is not sys.stdout:
        outhandle.close()
