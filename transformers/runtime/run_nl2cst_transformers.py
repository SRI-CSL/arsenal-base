import json
import math

from flask import Flask
from flask import request
import os, sys

from transformers import EncoderDecoderModel, BertTokenizerFast

from arsenal_tokenizer import PreTrainedArsenalTokenizer

from clean_input import clean_input


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

MODEL_ROOT = get_env('MODEL_ROOT') # the root of the training output
NL2CST_HOST = get_env('NL2CST_HOST', '127.0.0.1')
NL2CST_PORT = get_env('NL2CST_PORT', 8080)
NUM_BEAMS = int(get_env("NUM_BEAMS"))
NUM_OUTPUTS = int(get_env("NUM_OUTPUTS"))
TYPE_FORCING = int(get_env("TYPE_FORCING"))
BATCH_SIZE = int(get_env("BATCH_SIZE"))
CLEAN_INPUT = int(get_env("CLEAN_INPUT"))

app = Flask(__name__)

dataset_properties = json.load(open(os.path.join(MODEL_ROOT, "dataset_properties.json")))
target_vocab = dataset_properties["target_vocab"]
special_tokens = dataset_properties["special_tokens"]
max_input_length = dataset_properties["encoder_max_len"]

bert2arsenal = EncoderDecoderModel.from_pretrained(MODEL_ROOT)

tokenizer_path = os.path.join(MODEL_ROOT, "source_tokenizer")

# Try to use saved source tokenizer from file to prevent any downloads.
# Our older trained models didn't save the source tokenizer to disk, so use
# the download method as a fallback to remain compatible with older models.
if os.path.exists(tokenizer_path):
    source_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
else:
    print(f"no existing source tokenizer found, downloading...")
    source_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    source_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
target_tokenizer = PreTrainedArsenalTokenizer(target_vocab=target_vocab)
type_forcing_vocab =  target_tokenizer.id2vocab if TYPE_FORCING else None


@app.route('/')
def hello():
    return "NL2CST service is up and running..."

@app.route('/generateir', methods=['POST'])
def process():
    as_dict = request.get_json(force=True)

    # this returns a list of dicts of the form
    # {"id":"S1","sentence":"The parameter of the ENT_1 is a string."}
    #
    if 'msg' in as_dict:
        # Old format
        sentence_dicts = as_dict['msg']
        nl_inputs = [x["sentence"] for x in sentence_dicts]
    elif 'sentences' in as_dict:
        # New format
        sentence_dicts = as_dict['sentences']
        nl_inputs = [x["new-text"] for x in sentence_dicts]

    if CLEAN_INPUT:
        nl_inputs = [clean_input(line) for line in nl_inputs]

    num_batches = math.ceil(len(nl_inputs) / BATCH_SIZE)
    results = []
    for i in range(num_batches):

        batch_start = i * BATCH_SIZE
        batch_end = min((i + 1) * BATCH_SIZE, len(nl_inputs))
        batch = nl_inputs[batch_start:batch_end]

        input_tokens = source_tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt",
                              max_length=max_input_length)
        output_tokens = bert2arsenal.generate(
            input_ids=input_tokens.input_ids,
            attention_mask=input_tokens.attention_mask,
            decoder_start_token_id=target_tokenizer.cls_token_id,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_OUTPUTS,
            type_forcing_vocab=type_forcing_vocab,
            no_repeat_ngram_size=0  # default was 3, but this punishes desired translations -> figure out if/what setting we want to use here
        )

        # iterate over instances in batch to prepare outputs
        for j in range(batch_start, batch_end):
            try:
                id = sentence_dicts[j]["id"]
                nl = nl_inputs[j]

                # if beam search returns multiple outputs per input, those are stacked along the same dimension as the
                # outputs of different sentences, so we'll need to iterate over the output tensor in intervals of NUM_OUTPUTS
                csts = []
                for k in range(j-batch_start, j-batch_start+NUM_OUTPUTS):
                    csts.append(target_tokenizer.runtime_decode(output_tokens[k].tolist()))
                results.append({"id": id, "nl": nl, "cst": csts})
            except Exception as e:
                print(f"Exception processing sentence: {repr(e)}")
                results.append({"error": repr(e)})

    results = {"sentences": results}
    results = json.dumps(results, indent=3)
    return str(results)


#allow a command line argument to set the port (precedence over environment variable)
if __name__ == "__main__":
    setport=NL2CST_PORT
    if len(sys.argv) > 1:
        arg_port = int(sys.argv[1],0)
        if arg_port > 0:
            setport=arg_port
    app.run(host=NL2CST_HOST, port=setport)
