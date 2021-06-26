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
# def reformulateRules(cst_data):
#     ref_url = REF_URL_TEMPLATE.format(HOST,PORT_REF)
#     ref_input = json.dumps(cst_data)
#     response = requests.post(ref_url,data=ref_input)
#     if (response.ok):
#         ref_data = json.loads(response.text)
#         #this is a list of lists
#         ref_lists = ["\n".join(d.get('reformulations')) for d in ref_data.get('sentences')]
#         return "\n".join(ref_lists)
#     return "Was not able to reformulate result"


def callReformulate(ep_data, cst_data):
    # Combine EP and CST data into the reformulate request
    ep_sents = ep_data['sentences']
    cst_sents = cst_data['sentences']
    if len(ep_sents) != len(cst_sents):
        raise Exception("Different number of sentences from entity processor vs nl2cst!")
    result_arr = []
    for i in range(len(ep_sents)):
        ep_obj = ep_sents[i]
        cst_obj = cst_sents[i]
        id = ep_obj['id']
        if id != cst_obj['id']:
            raise Exception("Inconsistent sentence IDs from entity processor vs nl2cst!")
        #print(f"Processing sentence {id}: {cst_obj}")
        if 'error' in cst_obj.keys():
            print(f"Error processing sentence {id}: {cst_obj['error']}")
            continue
        result_arr.append({ 
            "orig-text": ep_obj['orig-text'],
            "id": ep_obj['id'], 
            "cst": cst_obj['cst'], 
            "substitutions": ep_obj['substitutions']
        })
    
    ref_url = REF_URL_TEMPLATE.format(HOST,PORT_REF)
    ref_input = json.dumps({ 
        "sentences": result_arr,
        "options" : {
            "prefix": "namespace"
        }
    })
    response = requests.post(ref_url,data=ref_input)
    if (response.ok):
        ref_data = json.loads(response.text)
        return ref_data
    else:
        raise Exception("Was not able to process CSTs!")

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

    out_filename = args.dest_file

    # If the destination file wasn't passed, then use stdout
    if out_filename is None:
        outhandle = sys.stdout
    else:
        outhandle = open(out_filename,'w')
    
    # import and clean up the input text
    sentence_data = importSentenceData(args) 

    # call the API for the entity processor for 
    print('Processing entities...')
    entity_processed_data = callEntityProcessor(sentence_data)

    if args.entities:
        outhandle.write("ENTITIES:\n")
        outhandle.write(json.dumps(entity_processed_data,indent=2) + '\n')

    # call the API for the language processor
    print('Inferring raw CSTs...')
    raw_csts = callNl2CstProcessor(entity_processed_data)

    # call reformulate with both ep results and csts
    print('Creating final CSTs...')
    final_csts = callReformulate(entity_processed_data, raw_csts)
    fcsts = json.dumps(final_csts, indent=2)
    # print(f'Final CSTs: {fcsts}')

        
    # # adds the definitions to the rules_data
    # embedEntityDefinitions(rules_data, entity_processed_data)

    # if args.cst:
    #     outhandle.write("CSTs:\n")
    #     rulesText = json.dumps(rules_data,indent=2)
    #     outhandle.write(rulesText + '\n')        

    #print out the generated results
    outhandle.write("Output:\n")
    outhandle.write(fcsts)
    outhandle.write('\n')

    # # if requested, generate human-readable output sentences
    # if args.reformulate:
    #     rulesText = reformulateRules(rules_data)
    #     outhandle.write(rulesText + '\n')

    if outhandle is not sys.stdout:
        outhandle.close()
