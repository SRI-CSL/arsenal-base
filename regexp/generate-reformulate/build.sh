DOCKERNAME=regexp-reformulate
VERSION=`cat VERSION`
pushd ../../
docker build . -t ${DOCKERNAME} -f regexp/generate-reformulate/Dockerfile $@
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:${VERSION}
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:latest
popd
