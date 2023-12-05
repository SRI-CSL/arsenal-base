#!/bin/bash
pushd ../../
docker build --no-cache . -f reformulate/builder/Dockerfile -t arsenal-reformulate-builder
if [ $? -ne 0 ]; then exit 1; fi 
popd
docker tag arsenal-reformulate-builder sricsl/arsenal-reformulate-builder:`cat VERSION`
docker tag arsenal-reformulate-builder sricsl/arsenal-reformulate-builder:latest
