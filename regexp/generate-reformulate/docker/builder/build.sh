#!/bin/bash
pushd ../../../../
docker build --no-cache . -f regexp/generate-reformulate/docker/builder/Dockerfile -t regex-builder
popd
docker tag regex-builder sricsl/regex-builder:`cat VERSION`
docker tag regex-builder sricsl/regex-builder:latest
