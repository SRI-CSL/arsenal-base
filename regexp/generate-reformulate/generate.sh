#!/bin/bash

SIZE=500000
if [ "$#" -eq 1 ]; then
    SIZE=$1
fi

TOP="REgrammar.re" # top-level type
shift
echo "Checking if we have an up-to-date version number"
./record_grammar.sh || exit
GRAMMAR=`cat version`
DATE=`date "+%Y-%m-%dT%H%M%z"`
TOPNAME=`echo "$TOP" | sed -e "s=/=-=g"`
DATASET="${GRAMMAR}_${TOPNAME}_${DATE}"
echo "Recording grammar.json"
./grammar2json.exe $TOP > "grammars/grammar-$DATASET.json" || exit
echo "Generating labelled pairs"


./generate.exe -separator="/" $TOP -polish -types -no-duplicates -n $SIZE > dataset.txt || exit

training_percentage=70
size=$(wc -l < dataset.txt)
train_size=$(($size * $training_percentage / 100))
test_size=$(($size * (100-$training_percentage) / 100))

# splits the dataset into two files eng-pn.aa and eng-pn.ab
echo "splitting data set into $train_size train and $test_size test samples"
split -l $train_size dataset.txt eng-pn.
mv eng-pn.aa eng-pn.train.txt
mv eng-pn.ab eng-pn.val.txt

./generate.exe $TOP -polish -types $@ > dataset.txt || exit
echo "Compressing the dataset in file $DATASET.tar.gz"
tar -czvf "training_sets/$DATASET.tar.gz" eng-pn.train.txt eng-pn.val.txt
echo "Removing tmp files"
rm dataset.txt
rm eng-pn.train.txt
rm eng-pn.val.txt
