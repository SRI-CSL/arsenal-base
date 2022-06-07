#!/bin/bash
pushd ../../..
docker build . --no-cache -f generate-reformulate/docker/reformulator/Dockerfile -t effigy-reformulate
popd
docker tag effigy-reformulate sricsl/effigy-reformulate:`cat ../../version`
docker tag effigy-reformulate sricsl/effigy-reformulate:latest
