-------------------------------------------------------
README
-------------------------------------------------------

This implements the recurrent encoder-decoder transduction model for converting 
English into the desired target form.  This module implements the following,

* HTTP service
* Model training
* Commandline evaluation 

-------------------------------------------------------
INSTALLATION
-------------------------------------------------------

- Setup Python 3.6

- Install Pytorch first.  Unfortunately the installation is entirely dependent
  upon OS, CUDA version, and Python version.  The invocations for configuring
  and installing Pytorch are at,

    https://pytorch.org/get-started/locally

- Install dependencies via:
   pip install -r dependencies.txt

-------------------------------------------------------
HTTP SERVICE
-------------------------------------------------------

- To start server on 8000,

  python run_nl2cst_server.py

- To test, star the server, then cd to scripts/ and execute

  wget_test.sh

-------------------------------------------------------
TRAINING
-------------------------------------------------------

To train a seq2seq model, use the train_seq2seq_single.py file via

   python train_seq2seq_single.py $DATA_ROOT, $NAME
   
Where $DATA_ROOT is the path to the data directory containing
the paired training and validation files.

$NAME is the string name describing this experiment, and is used
as a prefix for filenames used with this experiment.

The data itself in a text file, with each source and target instance
per line.  The source and target strings are tab delimited.

The training and validation files should be formatted as,
   $SRC-$TGT.train.txt
   $SRC-$TGT.val.txt
   
where $SRC and $TGT are the source and target language names.  
By default these are 'eng' and 'cst'.

The script also offers a debug option, which is set by adding -debug.  This allows the learning
to target an optional set of (presumably smaller) training files.  These should be under the
-data_root, and are specified as,

   $SRC-$TGT.train.debug.txt
   $SRC-$TGT.val.debug.txt
   
Debug runs also use a limited number of iterations.

Using '-h' with train_seq2seq_single.py will bring up a help
message describing additional arguments that can be used for training.

Models are saved into the $OUTPUT_DIR/..../checkpoints directory.
The filepath to the directory itself will be emitted during save out.

-------------------------------------------------------
EVALUATION
-------------------------------------------------------

To evaluate a trained model, 

   python eval_seq2seq.py $MODEL_ROOT $INPUT_FILE
   
Where
 
   $MODEL_ROOT is a path to a model checkpoint directory produced during training.
   $INPUT_FILE is a text file of input sentences, one per line
   
This will emit target sentences, one per line, corresponding to 
those in $INPUT_FILE.

