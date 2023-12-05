#!/bin/bash
pushd ../..
docker build . --no-cache -f reformulate/reformulator/Dockerfile -t arsenal-reformulate
if [ $? -ne 0 ]; then exit 1; fi 

popd
docker tag arsenal-reformulate sricsl/arsenal-reformulate:`cat VERSION`
docker tag arsenal-reformulate sricsl/arsenal-reformulate:latest
