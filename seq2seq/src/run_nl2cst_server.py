from flask import Flask
from flask import request
import os, sys
from ars2seq2seq.http.cst_generator import load_from_setup

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

MODEL_ROOT = get_env('MODEL_ROOT')
NORMALIZE_SAL_ENTITIES = get_env('NORMALIZE_SAL_ENTITIES', default=False)
INCLUDE_NORMED_FORM  = get_env('INCLUDE_NORMED_FORM', default=False)
REORDER_NUMBERED_PLACEHOLDERS  = get_env('REORDER_NUMBERED_PLACEHOLDERS', default=True)
CONVERT_TO_JSON = get_env('CONVERT_TO_JSON', default=True)
NL2CST_HOST = get_env('NL2CST_HOST', '127.0.0.1')
NL2CST_PORT = get_env('NL2CST_PORT', 8080)

app = Flask(__name__)
cst_gen = load_from_setup(MODEL_ROOT, normalize_sal_entities=NORMALIZE_SAL_ENTITIES,
                          include_normed_forms=INCLUDE_NORMED_FORM,
                          reorder_numbered_placeholders=REORDER_NUMBERED_PLACEHOLDERS,
                          convert_to_json=CONVERT_TO_JSON
                          )

@app.route('/hello')
def hello():
    return "Howdy"

@app.route('/generateir', methods=['POST'])
def process():
    as_dict = request.get_json(force=True)
    # Check if in old or new format
    if 'msg' in as_dict:
        # Old format
        sentence_dicts = as_dict['msg']
        print("Processing input={}".format(as_dict))
        retval = cst_gen.process_sentences(sentence_dicts)
    elif 'sentences' in as_dict:
        # New format
        sentence_dicts = as_dict['sentences']
        print("Processing input={}".format(as_dict))
        retval = cst_gen.process_sentences(sentence_dicts)
    return str(retval)


#allow a command line argument to set the port (precedence over environment variable)
if __name__ == "__main__":
    setport=NL2CST_PORT
    if len(sys.argv) > 1:
        arg_port = int(sys.argv[1],0)
        if arg_port > 0:
            setport=arg_port
    app.run(host=NL2CST_HOST, port=setport)
