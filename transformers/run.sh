#!/usr/bin/env bash

now=$(date +"%m-%d-%Y")
hostname=$(hostname)
# defines which gpus to use
export CUDA_VISIBLE_DEVICES=3

skip_databuild=false
# building absolute path bc/ tmux's pipe-pane doesn't work with relative path
root_dir=$(pwd)"/../../large_files/"
data_subdir="datasets/"
results_subdir="models/transformers/"$now"/" 

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -skipdatabuild)
            skip_databuild=true
            ;;
        -rootdir)
            root_dir=$VALUE
            ;;
        -datasubdir)
            data_subdir=$VALUE
            ;;
        -resultssubdir)
            results_subdir=$VALUE
            ;;
    esac
    shift
done

data_dir=$root_dir$data_subdir                      # location of processed data set
results_dir=$root_dir$results_subdir                # location of the results
target_lang_model=$results_dir"arsenal_model"       # the target LM model
translation_model=$results_dir"translation_model"   # the translation model, based on above
mkdir $results_dir

logfile=$results_dir$hostname"_"$now".log"
echo "logging to "$logfile
# logs output (if this is executed within tmux)
tmux pipe-pane "cat >> $logfile"

echo "data dir: "$data_dir
echo "results dir:"$results_dir

 if [[ $skip_databuild == false ]]; then
  echo "************** "$(date)": building dataset **************"
 python ./build_dataset.py \
      -data_dir=$data_dir \
      -out_dir=$data_dir \
      -max_source_len=75
 fi
 
 echo "************** "$(date)": training target model **************"
 python ./target_model.py \
      -data_dir=$data_dir \
      -output_dir=$target_lang_model \
      -epochs=1 \
      -batch_size=4 \
      -hidden_size=768 \
      -intermediate_size=256 \
      -num_hidden_layers=4 \
      -num_attention_heads=4
echo "************** "$(date)": training translation model **************"
python ./bert2arsenal.py \
     -epochs=1 \
     -data_dir=$data_dir \
     -target_model=$target_lang_model \
     -output_dir=$translation_model \
     -batch_size=4 \
#     -fp16=false 

echo "************** "$(date)": generate predictions from trained model **************"
python ./generate_predictions.py \
     -data_dir=$data_dir \
     -model_dir=$translation_model \
