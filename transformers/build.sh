#!/bin/bash
docker build . -f Dockerfile -t arsenal-nl2cst
if [ $? -ne 0 ]; then exit 1; fi 
docker tag arsenal-nl2cst sricsl/arsenal-nl2cst:`cat VERSION`
docker tag arsenal-nl2cst sricsl/arsenal-nl2cst:latest
