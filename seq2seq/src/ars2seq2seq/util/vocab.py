

"""
Simple utilities for working with vocabularies (word->idx)
"""

import os
import re
import unicodedata
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

PAD_token = 0  # Added new PAD token for padded sequences used in batched seq2seq
UNK_token = 1
SOS_token = 2
EOS_token = 3
LPAREN_token = 4
RPAREN_token = 5
LSQUARE_token = 6
RSQUARE_token = 7
LCURLY_token = 8
RCURLY_token = 9

def process_sentence(sentence):
    """
    Customized tokenizer for dealing with negative values and punctuations
    :param sentence:
    :return:
    """
    return [tok for tok in re.split('([\s:;,*\.\-\(\)\[\]\{\}])',sentence) if tok and tok!=' ']
#    return [tok for tok in re.split('(\W)', sentence) if tok and tok!=' ']
#     init_toks = sentence.split(' ')
#     ret_toks = []
#     for tok in init_toks:
#         if tok.startswith("-"):
#             ret_toks.append('-')  # negated values
#             ret_toks.append(tok[1:])
#         elif tok.endswith(",") and len(tok.strip()) > 1:
#             ret_toks.append(tok[:-1])
#             ret_toks.append(',')
#         elif tok.endswith(":") and len(tok.strip()) > 1:
#             ret_toks.append(tok[:-1])
#             ret_toks.append(':')
#         elif tok.endswith(".") and len(tok.strip()) > 1:
#             ret_toks.append(tok[:-1])
#             ret_toks.append('.')
#         else:
#             ret_toks.append(tok)
#     return ret_toks


class Lang:
    def __init__(self, name, load_path=None):
        self.name = name
        self.word2index = {"PAD": PAD_token, "UNK": UNK_token, "SOS": SOS_token, "EOS": EOS_token, 
                           "[": LSQUARE_token, "]": RSQUARE_token,
                           "(": LPAREN_token, ")": RPAREN_token, 
                           "{": LCURLY_token, "}": RCURLY_token}
        self.index2word = {}
        for word, idx in self.word2index.items():
            self.index2word[idx] = word
        self.n_words = len(self.word2index)  # Count special tokens
        self.BASELINE_N = len(self.word2index)
        self.load_path = load_path
        if load_path is not None and os.path.isfile(load_path):
            print("Loading existing vocab resource from {}".format(load_path))
            self.word2index, self.index2word = load_vocab(load_path)
            # If reconstituting, get the maximal index + 1 as the new n_words
            self.n_words = max([x for x in self.index2word.keys()]) + 1
            print("Loaded, total words={}, n_words={}".format(len(self.word2index), self.n_words))

    def is_opening_bracket(self, token):
        if self.name=='json':
            return token==LSQUARE_token
        elif self.name=='sexp':
            return token==LPAREN_token
        else:
            return False

    def is_closing_bracket(self, token):
        if self.name=='json':
            return token==RSQUARE_token
        elif self.name=='sexp':
            return token==RPAREN_token
        else:
            return False
            
    def fit_sentence(self, sentence):
        sentence_toks = process_sentence(sentence)
        for word in sentence_toks:
            self.fit_word(word)

    def fit_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            #self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        #else:
            #self.word2count[word] += 1

    def save(self, override_fpath=None):
        save_path = self.load_path
        if override_fpath is not None:
            save_path = override_fpath
        print("Saving Lang {} out to {}, #={}".format(self.name,
                                                      len(self.word2index),
                                                      save_path))
        save_vocab(self.word2index, save_path)

    def __len__(self):
        return len(self.word2index)

    def __str__(self):
        return "{}, #={}".format(self.name, len(self))


def load_lang(name, load_path):
    return Lang(name, load_path=load_path)


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalize_string(s, lowercase=False):
    if lowercase:
        s = unicodeToAscii(s.lower().strip())  # Lowercase
    else:
        s = unicodeToAscii(s.strip())
    s = re.sub(r"([.,!?])", r" \1", s)  # trim punctuation

    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # We want to retain control characters
    return s


def _vocab_fname(lang_name, root_dir):
    return os.path.join(root_dir, "{}.vocab".format(lang_name))


def read_paired(lang1_name, lang2_name, root_dir,
                reverse=False,
                filter_fn=None,
                load_if_present=True):
    print("Reading paired datafile, lang1={}, lang2={}, root_dir={}".format(lang1_name,
                                                                            lang2_name,
                                                                            root_dir))
    fname = os.path.join(root_dir, "{}-{}.txt".format(lang1_name, lang2_name))

    # Read the file and split into lines
    lines = open(fname, encoding='utf-8').\
        read().strip().split('\n')

    print("Normalizing, total lines={}".format(len(lines)))
    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    with ThreadPool(12) as p:
        def inner_fn(l):
            return [normalize_string(s) for s in l.split('\t')]
        pairs = list(tqdm(p.imap_unordered(inner_fn, lines)))

    if filter_fn:
        # Inefficient, but need to guarantee that inner_fn won't return
        # None deliberately.
        pairs = [pair for pair in pairs if filter_fn(pair)]

    lang1 = Lang(lang1_name, load_path=_vocab_fname(lang1_name, root_dir))
    lang2 = Lang(lang2_name, load_path=_vocab_fname(lang2_name, root_dir))

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = lang2
        output_lang = lang1
    else:
        input_lang = lang1
        output_lang = lang2

    if len(input_lang.word2index) == 3 and len(output_lang.word2index) == 3:
        print("Generating vocab for languages")
        for pair in pairs:
            input_lang.fit_sentence(pair[0])
            output_lang.fit_sentence(pair[1])
        print("Saving vocab out")
        input_lang.save()
        output_lang.save()

    return input_lang, output_lang, pairs


def length_filter_pair(p, MAX_LENGTH=1000):
    p0_toks = process_sentence(p[0])
    p1_toks = process_sentence(p[1])
    return len(p0_toks) < MAX_LENGTH and \
        len(p1_toks) < MAX_LENGTH


#
# Serialization
#
def save_vocab(vocab, tgtfpath):
    print("Saving vocab size={} out to {}".format(len(vocab), tgtfpath))
    with open(tgtfpath, "w") as f:
        ordered = sorted(vocab.items(), key=lambda x: x[1])
        for word, idx in ordered:
            f.write("{}\t{}\n".format(word.replace("\t", " "), idx))


def load_vocab(tgtpath):
    vocab = {"EOS": EOS_token, "SOS": SOS_token, "UNK": UNK_token}
    with open(tgtpath, "r") as f:
        for line in f:
            tuples = line.split("\t")
            if len(tuples) == 2:
                vocab[tuples[0]] = int(tuples[1])
            else:
                print("Warning, invalid line={}".format(line))
    reverse_vocab = {v:k for k,v in vocab.items()}
    return vocab, reverse_vocab


def sanitycheck(save_path="./sanity_check.vocab.txt"):
    vocab = {}
    vocab["a"] = 3
    vocab["b"] = 2
    vocab["c"] = 1
    save_vocab(vocab, save_path)
    loaded_vocab = load_vocab(save_path)
    sane = loaded_vocab["a"] == 3 and loaded_vocab["b"] == 2 and loaded_vocab["c"] == 1
    if sane:
        print("Sane")
    else:
        print("Not sane")
    os.remove(save_path)



if __name__ == "__main__":
    sanitycheck()
