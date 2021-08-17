import json
import os

from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer


data_dir = "./data/"

special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"] # the BERT standard special tokens

# # do we need to identify special tokens in the decoder? The target language basically consists only of special tokens
# # add arsenal-specific special tokens
# with open(os.path.join(data_dir, "special_tokens.txt"), "r") as f:
#     for line in f:
#         special_tokens.append(line[:-1])

with open(os.path.join(data_dir, "dataset"), "r") as f:
    instances = f.read().splitlines()

dataset = {}
target_vocab = []

# all words containing '_' are treated as special tokens and added to the tokenizer

for inst in instances:
    [source, target] = inst.split("\t")

    if source not in dataset:
        dataset[source] = [target]
    else:
        dataset[source].append(target)
    target_vocab.extend(target.split(" "))

target_vocab = list(set(target_vocab))

targets = []
for i in dataset.values():
    for j in i:
        targets.append(j)

targets = list(set(targets))

# tokenizer expects input files, so save targets to disk
targets_file = os.path.join(data_dir, "targets")
with open(targets_file, "w") as f:
    for target in targets:
        f.write(target + "\n")


# target_vocab = list(set(target_vocab))

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(special_tokens=special_tokens)

tokenizer.train(trainer, [targets_file])

print(tokenizer)

# print(tokenizer.get_vocab_size())
# print(tokenizer.get_vocab())
# print(target_vocab)
# print(tokenizer.get_vocab()["[CLS]"])
# for st in special_tokens

for k, v in tokenizer.get_vocab().items():
    if "[" in k:
        print(f"{k} - {v}")

vocab = {}
offset = len(special_tokens)
for i, token in enumerate(target_vocab):
    vocab[token] = i + offset

vocab_file =  open(os.path.join(data_dir, "vocab_file"), "w")
json.dump(vocab, vocab_file)
# tokenizer.vocab = vocab

print(f"size of target vocab: {len(target_vocab)}")

# tokenizer.model.vocab = vocab

# print(tokenizer.get_vocab_size())
print(tokenizer.model)

tokenizer.save(os.path.join(data_dir, "tokenizer_test"))


# wlt = Tokenizer(WordLevel(os.path.join(data_dir, "vocab_file")))