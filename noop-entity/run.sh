#!/bin/bash
FILENAME=$1
HOST=$2
printf "\nProcessing $FILENAME on $HOST\n"
FILE_DATA=`cat $FILENAME` 
curl -X POST -H "Content-type: application/json" $HOST/process -d @- <<CURL_DATA 
{ "text": "$FILE_DATA", "id": "S0" } 
CURL_DATA
