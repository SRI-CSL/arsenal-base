#!/bin/bash
docker build . -f Dockerfile -t arsenal-orchestrator
if [ $? -ne 0 ]; then exit 1; fi 
docker tag arsenal-orchestrator sricsl/arsenal-orchestrator:`cat VERSION`
docker tag arsenal-orchestrator sricsl/arsenal-orchestrator:latest
