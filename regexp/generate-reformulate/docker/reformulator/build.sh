#!/bin/bash
pushd ../../..
docker build . --no-cache -f generate-reformulate/docker/reformulator/Dockerfile -t regex-reformulate
popd
docker tag regex-reformulate sricsl/regex-reformulate:`cat ../../version`
docker tag regex-reformulate sricsl/regex-reformulate:latest
