import json
from flask import Flask
from flask import request
import os
import sys
import requests
import hashlib
import re
from aux.entity_rewriter import rewrite_entities
from aux.sentence_cleaner import clean_input

def get_env(argname, default=None):
    is_bool = type(default) == bool
    try:
        os_val = os.environ[argname]
        if is_bool:
            return os_val.lower().strip() == "true"
        return os_val
    except KeyError:
        if default is None:
            sys.exit("{} needs to be specified as an environment variable, exiting".format(argname))
        return default

    
HOST = get_env("HOST", "localhost")
PORT = int(get_env("PORT", 8080))

HOST_EP = get_env("HOST_EP", "entity")
PORT_EP = int(get_env("PORT_EP", 8080))

HOST_NL = get_env("HOST_NL", "nl2cst")
PORT_NL = int(get_env("PORT_NL", 8080))

HOST_REF = get_env("HOST_REF", "reformulate")
PORT_REF = int(get_env("PORT_REF", 8080))

GRAMMAR_PREFIX = get_env("GRAMMAR_PREFIX")

EP_URL_TEMPLATE = "http://{}:{}/process_all"
NL_URL_TEMPLATE = "http://{}:{}/generateir"
REF_URL_TEMPLATE = "http://{}:{}/"

def SentenceDataFromList(sentences):
    sentencesData = {}
    sentenceList = []

    # load sentences in to list, remove parens if needed
    for i, line in enumerate(sentences):
        text = line
        if text.startswith('#') or not text:  # comment or empty line, skip
            continue
        # make a dict for this sentences information
        sDict = {}
        sDict['id'] = "S" + str(i)
        sDict['text'] = text
        sentenceList.append(sDict)
    # set up dict for JSON format - might be empty
    sentencesData['sentences'] = sentenceList
    return sentencesData


# Call to the Arsenal Entity Process API
def callEntityProcessor(sentence_data, noop_ep):
    if noop_ep:
        for sent in sentence_data["sentences"]:
            sent["new-text"] = sent["text"]
            sent["orig-text"] = sent.pop("text")
            sent["substitutions"] = {}
        return sentence_data["sentences"]

    ep_url = EP_URL_TEMPLATE.format(HOST_EP,PORT_EP)
    
    ep_input = json.dumps(sentence_data)
    response = requests.post(ep_url,data=ep_input,headers={"Content-Type": "application/json"})
    if (response.ok):
        entities = json.loads(response.text)
        entities = rewrite_entities(entities)
        return entities['sentences']
    else:
        print(response.text)
        response.raise_for_status()

# clean sentences received from EP (using application-specific cleaning steps such as
# removing content in parentheses, substituting html tags, etc.)
def clean_sentences(ep_result):
    ep_sentences = [s["new-text"] for s in ep_result]
    cleaned_sentences = clean_input(ep_sentences)
    for i, s in enumerate(ep_result):
        assert ep_sentences[i] == s["new-text"]
        s["new-text"] = cleaned_sentences [i]
        s["precleaned-text"] = ep_sentences[i]

    return ep_result

# Preprocess sentences for nl2cst (add ids as hashes of post-EP sentence, i.e. new-text)
def createNl2CstInput(ep_result):
    uids = set()
    new_sents = []
    #for sent in ep_result['sentences']:
    for sent in ep_result:
        text = sent['new-text']
        uid = hashlib.blake2b(text.encode('utf-8'), digest_size=12).hexdigest()
        sent['cst-id'] = uid
        if not uid in uids:
            uids.add(uid)
            new_sents.append({ 'id': uid, 'new-text': text})
    return {'sentences': new_sents}, {'sentences': ep_result}

# Call to the Arsenal NL2CST API
def callNl2CstProcessor(nl2cst_data):
    nl_url = NL_URL_TEMPLATE.format(HOST_NL,PORT_NL)
    nl_input = json.dumps(nl2cst_data)
    response = requests.post(nl_url,data=nl_input)
    if (response.ok):
        return json.loads(response.text)
    #TODO how to handle failure?
    print(f"Error: {response}")
    raise Exception("Nl2CstProcessor didn't send ok response!")

# Call Arsenal's reformulation utility
def callReformulate(ep_data, cst_data, namespace):
    # Combine EP and CST data into the reformulate request
    ep_sents = ep_data['sentences']
    cst_sents = cst_data['sentences']
    cst_map = dict([(s['id'],s) for s in cst_sents])  # group csts by 'id'
    result_arr = []
    for i in range(len(ep_sents)):
        ep_obj = ep_sents[i]
        cst_id = ep_obj['cst-id']
        cst_obj = cst_map[cst_id]
        id = ep_obj['id']
        if 'error' in cst_obj.keys():
            print(f"Error processing sentence {id}: {cst_obj['error']}")
            continue
        result_arr.append({
            "orig-text": ep_obj['orig-text'],
            "ep-text": ep_obj['precleaned-text'],
            "cleaned-text": ep_obj['new-text'],
            "id": ep_obj['id'],
            "cst": cst_obj['cst'],
            "substitutions": ep_obj['substitutions']
        })

    ref_url = REF_URL_TEMPLATE.format(HOST_REF,PORT_REF)
    ref_input = json.dumps({
        "sentences": result_arr,
        "options" : {
            "prefix": namespace
        }
    })

    response = requests.post(ref_url,data=ref_input)

    if (response.ok):
        ref_data = json.loads(response.content.decode("utf-8"))

        # move sentence scores over from the raw cst to the final cst
        # (todo: in the future the reformulator should be extended so that
        # it can receive scores in the raw csts and include them in the final
        # output. But this requires API changes in arsenal base that are not
        # backward-compatible
        for i, final_cst in enumerate(ref_data['result']):
            cst_id = ep_sents[i]['cst-id']
            raw_cst = cst_map[cst_id]

            if "scores" in raw_cst:
                min_softmax_score = raw_cst["scores"][0]["min_softmax_score"]
                min_logit_score = raw_cst["scores"][0]["min_logit_score"]
                final_cst["softmax_score"] = min_softmax_score
                final_cst["logit_score"] = min_logit_score
                if "softmax_scores" in raw_cst["scores"][0]: 
                    softmax_scores = raw_cst["scores"][0]["softmax_scores"]
                    final_cst["softmax_scores"] = softmax_scores
                if "logit_scores" in raw_cst["scores"][0]:
                    logit_scores = raw_cst["scores"][0]["logit_scores"]
                    final_cst["logit_scores"] = logit_scores

            final_cst["raw_cst"] = raw_cst["cst"][0]

        return ref_data
    else:
        raise Exception("Was not able to process CSTs!")


def evaluate_entity_matches(ep_result, raw_csts, final_csts):

    ep_entities = []

    # don't use the structured ep output to determine entities, but extract them from
    # the (ep-processed) cleaned sentence string to make sure that we don't consider
    # any extraneous entitites that nl2cst never sees

    # ep_entities = [list(s["substitutions"].keys()) for s in ep_result]
    for r in ep_result:
        cleaned = r["new-text"]

        ep_ents = []
        for e in cleaned.split():
            m = re.match(f"({GRAMMAR_PREFIX}.*?_\d+)", e)
            if m is not None:
                ep_ents.append(m.group(1))

        ep_entities.append(ep_ents)

    cst_entities = {}  # map from cst id to raw cst entities

    for cst_res in raw_csts["sentences"]:        
        cst = cst_res['cst'][0] # only look at the first generated cst (usually we don't generate more anyway)

        ents = []

        for line in cst:
            parts = line.split("#")
            if parts[1].lower().startswith("entity"):
                ents.append("_" + parts[0])
        
        cst_id = cst_res['id']
        cst_entities[cst_id] = ents

    for i, r in enumerate(final_csts):
        cst_id = ep_result[i]['cst-id']
        ep_ents = ep_entities[i]
        cst_ents = cst_entities[cst_id]
        missing_ents = [e for e in ep_ents if e not in cst_ents]
        extra_ents = [e for e in cst_ents if e not in ep_ents]
        r["missing_entities"] = missing_ents
        r["extra_entities"] = extra_ents

app = Flask(__name__)

@app.route('/')
def hello():
    return "Arsenal orchestrator service is up and running..."

@app.route('/run', methods=['POST'])
def process():
    req = request.get_json(force=True)

    args = req["args"]

    noop_ep = False if "noop_ep" not in args else args["noop_ep"]
    include_full_scores = False if "include_full_scores" not in args else args["include_full_scores"]
    namespace = "http://www.sri.com/arsenal#" if "namespace" not in args else args["namespace"]
    include_raw = False if "include_raw" not in args else args["include_raw"]
    include_scores = True if "include_scores" not in args else args["include_scores"]


    sentence_data = SentenceDataFromList(req["sentence_data"])
    ep_result = callEntityProcessor(sentence_data, noop_ep)
    ep_result = clean_sentences(ep_result)
    
    nl2cst_input, processed_ep_result = createNl2CstInput(ep_result)
    nl2cst_input["include_full_scores"] = include_full_scores
    nl2cst_input["include_scores"] = include_scores
    print(f'Inferring raw CSTs for {len(nl2cst_input["sentences"])} unique inputs out of {len(processed_ep_result["sentences"])} original sentences...')
    raw_csts = callNl2CstProcessor(nl2cst_input)
    print('Creating final CSTs...')
    
    final_csts = callReformulate(processed_ep_result, raw_csts, namespace)["result"]

    evaluate_entity_matches(ep_result, raw_csts, final_csts)

    # extract raw csts from result (mostly s.t. they can be easily displayed in the UI)
    # only keep them in the final results if explicitly requested in the args to 
    # not clutter up the results too much
    raw_csts = []
    for cst in final_csts:
        if "error" in cst:
            entry = dict((k, cst["json"][k]) for k in ["id", "cleaned-text", "cst"])
            entry["raw_cst"] = entry.pop("cst")
            
        else:
            entry = dict((k, cst[k]) for k in ["id", "cleaned-text", "raw_cst"])
        raw_csts.append(entry)

        if not include_raw:
            cst.pop("raw_cst")
    
    # remove internal hash-based ids (only used to avoid repeated processing of identical inputs)
    for e in ep_result:
        e.pop("cst-id")

    return {"entities": ep_result, "raw_csts": raw_csts, "final_csts": final_csts}

#allow a command line argument to set the port (precedence over environment variable)
if __name__ == "__main__":
    setport=PORT
    if len(sys.argv) > 1:
        arg_port = int(sys.argv[1],0)
        if arg_port > 0:
            setport=arg_port
    app.run(host=HOST, port=setport)
