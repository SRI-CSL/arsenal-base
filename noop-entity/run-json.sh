#!/bin/bash
FILENAME=$1
HOST=$2
printf "\nProcessing $FILENAME on $HOST\n"
FILE_DATA=`cat $FILENAME` 
curl -X POST -H "Content-type: application/json" $HOST/process_all -d @- <<CURL_DATA 
$FILE_DATA
CURL_DATA
