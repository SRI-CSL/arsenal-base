import os
import torch
import sys
import traceback
import json
from json import JSONDecodeError
from ars2seq2seq.models.multi_attn_seq2seq import MultiHeadSeq2Seq
#from ars2seq2seq.experiments.e190130_robustness import MultiHeadSeq2Seq
from ars2seq2seq.util.vocab import load_lang
from ars2seq2seq.util.txt2dict import gen_toks2dict
from nvidia_utils import *
from ars2seq2seq.util.polish import convert_pn2sexp
from ars2seq2seq.util.polish import convert_pn2json
from ars2seq2seq.util.vocab import normalize_string
from ars2seq2seq.util.entities import normalize_sal_entities, reorder_numbered_placeholders, reinsert_from_lookup

"""

190426 TODO: Currently we are using datasets that do not quote their targets, while others do quote their targets to generate
JSON.  Because of this, in some cases we will generate paired double-quotes (e.g., '""') on tokens.  Until we can
normalize the training to be uniform, we will have to do a string replace hack to account for this.
"""

NUM_HIDDEN = 128
NUM_LAYERS = 2
NORMALIZE_ENTITIES=False

def convert_sexp2json(sexp_str):
    """ Hack to accommodate sexpression to JSOn conversion"""
    print("Converting SEXP to JSON: {}".format(sexp_str))
    json_str = sexp_str
    toks = []
    for tok in json_str.split():
        if tok not in set(['(',')']):
            parts = tok.split(':',1)
            if isinstance(parts,list) and len(parts)==2:
                [node,nodetype] = parts
                if node != 'List':
                    if node != 'Nil':
                        toks.append("{{ \"node\": \"{}\", \"type\": \"{}\" }}".format(node,nodetype))
                    else:
                        toks.append("[ ]")
            else:
                if tok != 'List':
                    if tok != 'Nil':
                        toks.append("\"{}\"".format(tok))
                    else:
                        toks.append("[ ]")
        else:
            toks.append(tok.replace('(','[').replace(')', ']'))
    json_str = " ".join(toks)
    json_str = json_str.replace('" "', '", "')
    json_str = json_str.replace('" [', '", [')
    json_str = json_str.replace('" {', '", {')
    json_str = json_str.replace('} [', '}, [')
    json_str = json_str.replace('} {', '}, {')
    json_str = json_str.replace('} "', '}, "')
    json_str = json_str.replace('] [', '], [')
    json_str = json_str.replace('] "', '], "')
    json_str = json_str.replace('] {', '], {')
    json_str = json_str.replace('"None"','null').replace('"true"','true').replace('"false"','false')
    return json_str

class CSTGenerator:
    def __init__(self, model_root,
                 num_hidden=NUM_HIDDEN,
                 input_lang_name="eng",
                 output_lang_name="cst",
                 max_length=1000,
                 num_layers=NUM_LAYERS,
                 num_attn=2,
                 match_parens=True,
                 init_type='',
                 dropout=0.1,
                 device=None,
                 include_normed_forms=False,
                 normalize_sal_entities=NORMALIZE_ENTITIES,  # Norm entity mentions into ID__$DIGIT form
                 reorder_numbered_placeholders=True,
                 convert_to_json=True,
                 verbose=False):
        self.model_root = model_root
        self.input_lang_name, self.output_lang_name = input_lang_name, output_lang_name
        self.include_normed_forms = include_normed_forms
        self.normalize_sal_entities = normalize_sal_entities
        self.reorder_numbered_placeholders = reorder_numbered_placeholders
        if "regexp/eng-sexp" in model_root:
            print("Automatically activating sexp to json conversion, triggered on regexp/eng-sexp model path")
            self.convert_to_json = True
        else:
            if self.output_lang_name == "sexp" or self.output_lang_name == "pn":
                self.convert_to_json = convert_to_json
            else:
                self.convert_to_json = False
        self.verbose = verbose
        if device is None:
            device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print("CST Generator starting on device={}".format(device))
            print("Convert to JSON={}".format(self.convert_to_json))
        input_vocab_fname = "{}.vocab".format(self.input_lang_name)
        output_vocab_fname = "{}.vocab".format(self.output_lang_name)
        vocab_dir = model_root
        input_lang = load_lang(self.input_lang_name, os.path.join(vocab_dir, input_vocab_fname))
        output_lang = load_lang(self.output_lang_name, os.path.join(vocab_dir, output_vocab_fname))
        self.seq2seq = MultiHeadSeq2Seq(num_hidden, input_lang, output_lang,
                               device=device,
                               verbose=verbose,
                                max_length=max_length,
                                num_layers=num_layers,
                                num_attn=num_attn,
                                match_parens=match_parens,
                                init_type=init_type,
                                dropout=dropout)
        self.seq2seq.load_checkpoint(override_checkpoint_dir=model_root)
        if self.verbose:
            print("Norm SAL entities={}".format(self.normalize_sal_entities))
            print("Reorder numbered placeholders={}".format(self.reorder_numbered_placeholders))
            print("Include normed forms={}".format(self.include_normed_forms))

    #
    # TODO: Print out warnings for unseen vocab items
    #
    def process_sentences(self, sentence_dicts):
        accum_cst = []
        for sentence_dict in sentence_dicts:
            try:
                sid = sentence_dict['id']
                if 'sentence' in sentence_dict:
                    # Old format
                    raw_sentence = sentence_dict['sentence']
                else:
                    # New format
                    raw_sentence = sentence_dict['new-text']
                if self.verbose:
                    print("sid={}, raw={}".format(sid, raw_sentence))
                # For each sentence, identify the IDs first, normalize them, and then replace them
                sentence = normalize_string(raw_sentence)
                if self.normalize_sal_entities:
                    sentence, e2, rlookup = normalize_sal_entities(sentence, "")
                else:
                    rlookup = {}
                if self.reorder_numbered_placeholders:
                    sentence, _, rlookup2 = reorder_numbered_placeholders(sentence, "")
                    rlookup.update(rlookup2)
                if self.verbose:
                    print("Placeholder checks done")
                gen_cst_toks, _ = self.seq2seq.generate(sentence)
                if self.verbose:
                    print("... generated")
                # Hack to fix empty tokens introduced before '!=' (inequality) functions
                gen_cst_toks = [t for t in gen_cst_toks if len(t) > 0]
                if self.normalize_sal_entities or self.reorder_numbered_placeholders:
                    print("RLookup={}".format(rlookup))
                    gen_cst_toks = reinsert_from_lookup(gen_cst_toks, rlookup)
                if self.output_lang_name.lower() == "json" or self.convert_to_json:
                    gen_str_form = " ".join(gen_cst_toks).replace("<EOS>", "").replace("<SOS>", "")
                else:
                    # Apply JSON-ification only if the target is not JSON
                    gen_str_form = gen_toks2dict(gen_cst_toks)
                if self.convert_to_json:
                    if self.output_lang_name=='sexp':
                        gen_str_form = convert_sexp2json(gen_str_form)
                    elif self.output_lang_name=='pn':
                        #debugging - making this call prints out before and after
                        print(convert_pn2sexp(gen_str_form))
                        gen_str_form = convert_pn2json(gen_str_form)
                gen_cst_dict = json.loads(gen_str_form)
                if self.verbose:
                    print("... remapping done")
                if self.include_normed_forms:
                    accum_cst.append({"id": sid, "nl": raw_sentence, "cst": gen_cst_dict,
                                      "normed_form" : sentence })
                else:
                    accum_cst.append({"id": sid, "nl": raw_sentence, "cst": gen_cst_dict})
            except JSONDecodeError:
                print("Invalid JSON! Value={}".format(gen_str_form))
                if self.include_normed_forms:
                    accum_cst.append({ "id" : sid, "error" : "Invalid JSON", "normed_form" : sentence ,
                                       "nl": raw_sentence,
                                       "cst_attempted": gen_str_form})
                else:
                    accum_cst.append({ "id" : sid, "error" : "Invalid JSON, value={}".format(gen_str_form)})
        #     except:
        #         if self.include_normed_forms:
        #             accum_cst.append({ "id" : sid, "error" : "Unhandled exception", "normed_form" : sentence })
        #         else:
        #             accum_cst.append({ "id" : sid, "error" : "Unhandled exception, e={}".format(sys.exc_info()) })
        root_elt = {"sentences" : accum_cst }
        return json.dumps(root_elt, indent=3)

    def process_txt_sentences(self, txt_sentences, strip_eos=True):
        ret_tgt = []
        for sentence_txt in txt_sentences:
            sentence = normalize_string(sentence_txt)
            if self.normalize_sal_entities:
                sentence, e2, rlookup = normalize_sal_entities(sentence, "")
            else:
                rlookup = None
            if self.reorder_numbered_placeholders:
                sentence, _, rlookup2 = reorder_numbered_placeholders(sentence, "")
            if rlookup is None:
                rlookup = rlookup2
            else:
                rlookup.update(rlookup2)
            gen_tgt_toks, _ = self.seq2seq.generate(sentence)
            gen_tgt_toks = [t for t in gen_tgt_toks if len(t) > 0]  # Fix to remove double empties
            if strip_eos:
                if gen_tgt_toks[-1] == '<EOS>':
                    gen_tgt_toks = gen_tgt_toks[0:-1]
            if self.normalize_sal_entities or self.reorder_numbered_placeholders:
                gen_tgt_toks = reinsert_from_lookup(gen_tgt_toks, rlookup)
            ret_str = " ".join(gen_tgt_toks)
            if self.convert_to_json:
                if self.output_lang_name == 'sexp':
                    ret_str = convert_sexp2json(ret_str)
                elif self.output_lang_name == 'pn':
                    ret_str = convert_pn2json(ret_str)
#                    ret_str = convert_sexp2json(convert_pn2sexp(ret_str))
            ret_tgt.append(ret_str)
        return ret_tgt


def load_from_setup(model_root, device="cpu", verbose=True,
                    normalize_sal_entities=False,
                    include_normed_forms=False,
                    reorder_numbered_placeholders=True,
                    convert_to_json=True):
    with open(os.path.join(model_root, "setup.json"), "r") as f:
        args = json.load(f)
        return CSTGenerator(model_root,
                                   num_hidden=args['hidden_size'],
                                   input_lang_name = args['input_lang'],
                                   output_lang_name = args['output_lang'],
                                   max_length=args['max_length'],
                                   num_layers=args['num_layers'],
                                   num_attn=args['num_attn'],
                                   #set default values for model params that may not exist in legacy models
                                   match_parens=args.get('match_parens',True),
                                   init_type=args.get('init_type',''),
                                   dropout=args.get('dropout',0.1),
                                   verbose=verbose,
                                   normalize_sal_entities=normalize_sal_entities,
                                   reorder_numbered_placeholders=reorder_numbered_placeholders,
                                   include_normed_forms=include_normed_forms,
                                   convert_to_json=convert_to_json,
                            device=device)
