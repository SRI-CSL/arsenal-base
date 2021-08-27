import os
from typing import Optional, Tuple

from tokenizers import Tokenizer, pre_tokenizers, processors
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

PRETRAINED_TOKENIZER_FILE = "arsenal_target_tokenizer.json"

class FastArsenalTokenizer(BaseTokenizer):

    def __init__(
        self,
        target_vocab,
    ):
        special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        }

        vocab = {}
        vocab[special_tokens["pad_token"]] = 0

        tkn_idx = 1
        unused_ctr = 0

        # not sure whether that's relevant, but fill 1..99  and 105...999
        # with unused tokens to keep BERT's tokenizer style
        # as a result, one can easily identify special tokens:
        # 0 is padding
        # 1xx are other special tokens
        # any four-digit tokens are actual payload
        fill_tokens = False

        if(fill_tokens):
            while(tkn_idx < 100):
                vocab[f"[unused{unused_ctr}]"] = tkn_idx
                tkn_idx += 1
                unused_ctr += 1

        for token in ["unk_token", "cls_token", "sep_token", "mask_token"]:
            vocab[special_tokens[token]] = tkn_idx
            tkn_idx += 1

        if(fill_tokens):
            while(tkn_idx < 1000):
                vocab[f"[unused{unused_ctr}]"] = tkn_idx
                tkn_idx += 1
                unused_ctr += 1

        for word in target_vocab:
            vocab[word] = tkn_idx
            tkn_idx += 1

        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=special_tokens["unk_token"]))
        tokenizer.add_special_tokens(list(special_tokens.values()))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        sep_token_id = tokenizer.token_to_id(special_tokens["sep_token"])
        cls_token_id = tokenizer.token_to_id(special_tokens["cls_token"])

        tokenizer.post_processor = processors.BertProcessing(
            (special_tokens["sep_token"], sep_token_id), (special_tokens["cls_token"], cls_token_id)
        )

        parameters = special_tokens
        parameters["model"] = "WordLevel"

        super().__init__(tokenizer, parameters)

        # need to save the tokenizer s.t.
        tokenizer.save(PRETRAINED_TOKENIZER_FILE)

# wrapper class to make tokenizer available for huggingface transformers
class PreTrainedArsenalTokenizer(PreTrainedTokenizerFast):
    def __init__(self, target_vocab):
        FastArsenalTokenizer(target_vocab) # this doesn't need to be done everytime, but it is so fast that it doesn't matter
        super().__init__(tokenizer_file=PRETRAINED_TOKENIZER_FILE)
        self.add_special_tokens({"pad_token": "[PAD]"})
        self.add_special_tokens({"mask_token": "[MASK]"})
        self.add_special_tokens({"sep_token": "[SEP]"})
        self.add_special_tokens({"unk_token": "[UNK]"})
        self.add_special_tokens({"cls_token": "[CLS]"})

        self.id2vocab = {}
        for word, id in self.backend_tokenizer.get_vocab().items():
            self.id2vocab[id] = word

    # dummy save method to prevent "not implemented" errors during training
    # (Our tokenizer is simply word-based. There's nothing to be learned, and thus no need to save any progress.)
    def save_pretrained(self, output_dir):
        pass

    def decode(self, tokens):
        decoded = []
        for token in tokens:
            if token != 0 and token != -100:
                decoded.append(self.id2vocab[token])

        return " ".join(decoded)


