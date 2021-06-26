#!/bin/bash
HOST=$1 # URL to arsenal-base git repo where the learning took place 
MODEL=`echo "$2" | sed 's=^\([^/]*\)/\(.*\)$=\1='`
mkdir "$MODEL"
URL="$HOST/seq2seq/output/single/$2/checkpoints"
scp "$URL/decoder" "$MODEL"
scp "$URL/encoder" "$MODEL"
scp "$URL/eng.vocab" "$MODEL"
scp "$URL/pn.vocab" "$MODEL"
scp "$URL/notes.txt" "$MODEL"
scp "$URL/setup.json" "$MODEL"

