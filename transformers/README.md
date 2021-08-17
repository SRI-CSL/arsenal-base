# A Seq2Seq model from English to Arsenal Effigy Language
This builds a translation model to translate sequences of natual English language to sequences of "Arsenal's Effigy Language" (i.e., a sequential representation of ASTs.) The main idea is to use two different transformer-based language model to model source and target language, respectively. And then learn a encoder-decoder-model to translate from the source language to the target language. The source language model is represented through an off-the-shelf BERT English language model, while the target model is a custom learned LM for Arsenal's Effigy Language.

The steps are:

## Preoprocessing
- The original generated data is in data/dataset.tar.gz (compressed to avoid file size limitations in repo), can be extracted with ```tar -xzvf dataset.tar.gz```. (No need to do this manually, the following step will extract the archive automatically if there's no extracted file.)
- Based on generated data, the (train/val/test) datasets are created via ```build_dataset.py```. This creates tokenized sets using
  - BertTokenizer to tokenize the source (english) sentences
  - ArsenalTokenizer (see below) to tokenize target (formal) sentences

### BertTokenizer
- This is nearly an off-the-shelf BERT tokenizer pretrained on English. The only change is that Arsenal's Entity placeholder variable names are added as special tokens (s.t. they are treated as individual tokens and not treated according to English language tokenization rules).

### ArsenalTokenizer
- The arsenal tokenizer creates WordLevel tokens, one for each word in Arsenal's grammar (or to be more precise: one for each unique word found in the target part of the data set). Additionally, it contains the usual special BERT tokens for *unknown* (though that should never occur), *mask*, etc.
- This is implemented using hugginface's tokenizer library (which is distinct from the transformers library). The actual tokenizer is a subclass of `tokenizer.BaseTokenizer` and generates a json file with all relevant tokenizer information.
- These tokenizers cannot be used directly in the transformer library (though that might change soon - this is under heavy development). To make this available, `FastArsenalTokenizer` is a wrapper class that makes above tokenizer accessible in the transformers library. To instantiate this, it needs to be initialized with a list of the target vocab like this `FastArsenalTokenizer(target_vocab)`. This will then automatically create a tokenizer as described above, and use the generated json file to to create the transformers-compatible tokenizer. This process can probably be improved, but it works...

## Target Model
- `target_model.py` uses the target part of the dataset, tokenizes it with the arsenal-specific tokenizer, and then trains a masked language model from that. The resulting language model should represent the "Arsenal language".

## Seq2Seq Translation Model
- `bert2arsenal.py` builds on an `EncoderDecodelModel` from huggingface's transformers library to build a translation model that translates from English natural language to Arsenal's formal Effigy Language. The key elements of this model are
 - A pretrained BERT-Model for English used as the encoder. This is just a downloaded model without any modification (except adding special tokens to the tokenizer, as described above).
 - A custom-trained language model for Effigy, as described in the previous section.
- The training is done through huggingface's `Seq2SeqTrainer`. Significant aspects of the current training configuration are:
 - Uses beam search, currently set to 4 beams
 - Uses early stopping (but this hasn't triggered in current experiments so far, not sure how many epochs would be needed to reach that point)
 - Penalizes longer output sequences. Penalty weight is set to 2.0 (where 1.0 corresponds to no penalty, values > 1 introduce an exponential penalty for long sequences, while values < 1 penalize short sequences).
 - The model doesn't allow for more than two occurrences of any n-gram in the target model. This is to prevent target models that keep repeating themselves. We need to check what a meaningful configuration is for the Arsenal Effigy language.
 - There are many more parameters that can be tweaked, lots of room for experiments and optimization...

# Putting it all together
The script `run.sh` executes all of the above tasks in a pipeline. This requires a `dataset` file in `./data` (can also be a compressed file `./data/dataset.tar.gz` - this would be automatically unpacked) and produces per default
- a target LM in `./arsenal-model`
- a translation model in `../arsenal/large-files/models/transformers/[date]/translation/`
The paths (and a few other parameters such as # of epochs) can be changed in the `run.sh` script, many more parameters could be tweaked directly in the python files (todo: make relevant parameters available as arguments).

Tensorboard logs can be found in the `./logs` subfolders of the respective target folders mentioned above.

Loading a pretrained model can be done like this:
```python
bert2arsenal = EncoderDecoderModel.from_pretrained(path)
```
where path points to a checkpoint from the training process, i.e., something like `./results/translation/{current-time}/checkpoint-XXXXXX`

__*Note:*__ There seems to be a bug with saving the vocab size of the model. Additionally added tokens are not considered when saving the model's state. When directly trying to load a model from a checkpoint, this will cause a RuntimeError:
```
size mismatch for encoder.embeddings.word_embeddings.weight: copying a param with shape torch.Size([30678, 768]) from checkpoint, the shape in current model is torch.Size([30522, 768]).
```
The difference between these two dimensions should correspond exactly to the number of added special tokens (156 in this example). To fix this, the `config.json` in the checkpoint dir needs to be edited: There's a property `vocab_size` that needs to be corrected, b/c it misses the additionally added tokens. In the example above, it is specified as `30522` and needs to be changed to `30678` (original size + # added special tokens).
