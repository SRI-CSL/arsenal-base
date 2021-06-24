#!/bin/bash
TOP=$1 # top-level type
shift
echo "Checking if we have an up-to-date version number"
./record_grammar.sh || exit
GRAMMAR=`cat version`
DATE=`date "+%Y-%m-%dT%H%M%:::z"`
TOPNAME=`echo "$TOP" | sed -e "s=/=-=g"`
DATASET="${GRAMMAR}_${TOPNAME}_${DATE}"
echo "Recording grammar.json"
./_build/default/src/grammar2json.exe $TOP > "grammars/grammar-$DATASET.json" || exit
echo "Generating labelled pairs"
./_build/default/src/generate.exe $TOP -polish -types $@ > dataset.txt || exit
echo "Compressing the dataset in file $DATASET.tar.gz"
tar -czvf "training_sets/$DATASET.tar.gz" dataset.txt
echo "Removing tmp file"
rm dataset.txt
