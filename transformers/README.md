# A Transformers-based Seq2Seq model from English to Arsenal Effigy Language
This builds a translation model to translate sequences of natual English language to sequences of "Arsenal's Target Language" (i.e., a sequential representation of CSTs.) The main idea is to use two different transformer-based language model to model source and target language, respectively. And then learn a encoder-decoder-model to translate from the source language to the target language. The source language model is represented through an off-the-shelf BERT English language model, while the target model is a domain-specific custom learned LM for Arsenal's Target Language. The main steps of the transformers pipeline are (described in more detail below):
1. Build the data sets (essenentially tokenize the data)
2. Train the target LM
3. Based on a standard pretrained NL model for the source language and the target model trained in the previous step, train a translation model to translate from NL to CSTs.
4. Use the trained translation model to generate translations from the NL part of the validation set. These generated translations can then be compared against the CST part of the validation set to evaluate the model's performance.

## Components
There are two major components to the transformers-model:
1. A framework to prepare datasets, train models, and evaluate results, as described below, the corresponding code is in ```./src/```
2. A runtime environment to integrate a trained model into a dockerized Arsenal system, the corresponding code is in ```./runtime/```

## Requirements
It is recommended to set up a virtual environment to run this pipeline. For example, with miniconda this can be done like this:
``` shell
conda create -y -n arsenal-transformers python==3.8
```

(the ```-y``` skips interactive prompts and ```-n``` specifies the name for the new environment, can be chosen arbitrarily). This scripts have been tested with python 3.8, other versions might work, too.  
After the environment has been created, it needs to be activated, e.g., (when above command has been used):
```shell
conda activate arsenal-transformers
```

Then, additional requirements can be installed with
```shell
pip install -r requirements.txt
```

## Optional Typeforcing
The transformers model can optionally use typeforcing to ensure that generated translations are type-correct according to the grammar. However, experiments so far showed that the trained transformers model learn the grammar well enough so that type-forcing does not bring any additional improvements.  

To enable typeforcing in the train/evaluation framework, a patched version of huggingface's transformers library is required. The easiest way to get the patched version is to run the script
```typeforcing_patch.sh``` in ```./typeforcing_patch/``` and then install the resulting library into the conda environment with
```
pip install -e typeforcing_patch/transformers-4.4.2/
```
(this will replace the default transformers library installed via the ```pip install -r requirements.txt``` step above).

The dockerized runtime will take care of patching the library automatically.

# The Process in more detail
## Preoprocessing
- The original generated data is stored in text files with tab-separated pairs of English sentence and corresponding CSTs, one pair per line.
- Based on the generated data, the (train/val) datasets are created via ```build_dataset.py```. This creates tokenized sets using
  - BertTokenizer to tokenize the source (english) sentences
  - ArsenalTokenizer (see below) to tokenize target "sentences" (i.e., CSTs in polish notation)
Per default, it is assumed that the input files in the input data directory are called ```eng-pn.train.txt``` resp. ```eng-pn.val.txt```, but both the directory and the file names can me changed (see. ```args.py``` for details).

### BertTokenizer
- This is nearly an off-the-shelf BERT tokenizer pretrained on English. The only change is that Arsenal's Entity placeholder variable names are added as special tokens (s.t. they are treated as individual tokens and not treated according to English language tokenization rules).

### ArsenalTokenizer
- The arsenal tokenizer creates WordLevel tokens, one for each word in Arsenal's grammar (or to be more precise: one for each unique word found in the target part of the data set). Additionally, it contains the usual special BERT tokens for *unknown* (though that should never occur), *mask*, etc.
- This is implemented using hugginface's tokenizer library (which is distinct from the transformers library). The actual tokenizer is a subclass of `tokenizer.BaseTokenizer` and generates a json file with all relevant tokenizer information.
- These tokenizers cannot be used directly in the transformer library (though that might change soon - this is under heavy development). To make this available, `FastArsenalTokenizer` is a wrapper class that makes above tokenizer accessible in the transformers library. To instantiate this, it needs to be initialized with a list of the target vocab like this `FastArsenalTokenizer(target_vocab)`. This will then automatically create a tokenizer as described above, and use the generated json file to to create the transformers-compatible tokenizer. This process can probably be improved, but it works...

## Target Model
- `target_model.py` uses the target part of the dataset, tokenizes it with the arsenal-specific tokenizer, and then trains a masked language model from that. The resulting language model should represent the formal "Arsenal Target language" (i.e., CSTs in polish notation).

## Seq2Seq Translation Model
- `bert2arsenal.py` builds on an `EncoderDecodelModel` from huggingface's transformers library to build a translation model that translates from English natural language to Arsenal's formal target Language. The key elements of this model are
 - A pretrained BERT-Model for English used as the encoder. This is just a downloaded model without any modification (except adding special tokens to the tokenizer, as described above).
 - A custom-trained language model for the respective Arsenal domain, as described in the previous section.
- The training is done through huggingface's `Seq2SeqTrainer`.

## Evaluating the Trained Model
- `generate_predictions.py` takes the NL sentences from the validation set and uses the latest checkpoint of the trained translation model from the previous step to generate translations to polish-notation CSTs for the NL sentences. The generated translations are stored in the root folder of the respective run (i.e., ```model_root/[run_id]/predictions_[run_id]_[checkpoint_id].txt```). These generated translation can then subsequently be evaluated against the ground truth specified in the validation file. Cf. ```eval.py``` for some examples.  
Generating the predictions is a (gpu) resource intensive task, thus the generation step is included in the overall pipeline described below. Evaluating the generated predictions on the other hand is a fairly cheap task and might require several tweaks to produce the desired metrics, graphs, etc. Thus, the actual eval script is (currently) not integrated into the main pipeline.


# Putting it all together

- The script `run.py` executes all of the above tasks in a pipeline. All parameters have default values so that the script can run without any parameter specifications if the expected directory structure (i.e., dataset directory and model root) exists. However, all relevant parameters can be set manually, cf. ```args.py``` for an overview of all parameters together with their default values and brief explanations (or simply run ```python run.py -h``` to get this information).  

- All of the above steps can also be run individually by executing the corresponding python files. For consistency, all scripts rely on the same set of arguments specified in ```args.py``` even though each individual step will only require a subset of these arguments.

- The default directory structure for the created models is:
  ```shell
  - [model_root]/
    - [run_id]/ # automatically created as MM-DD-YYYY if not specified
      - target_model/ # stores everything related to the learned LM of Arsenal's "target language"
      - translation_model/ # stores everything related to the learned translation model
  ```

- Tensorboard logs can be found in the `./logs` subfolders of the respective model folders mentioned above.

- To use a trained model in the dockerized Arsenal setup, the docker configuration needs to point to a checkpoint of a trained translation model, i.e., a path like ```[model_root]/[run_id]/translation/model/checkpoint_XXX```.
