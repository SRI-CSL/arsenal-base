#!/bin/bash
docker build . -f docker/Dockerfile -t arsenal-ui
if [ $? -ne 0 ]; then exit 1; fi 
docker tag arsenal-ui sricsl/arsenal-ui:`cat VERSION`
docker tag arsenal-ui sricsl/arsenal-ui:latest
