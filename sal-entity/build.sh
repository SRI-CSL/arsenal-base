DOCKERNAME=generic-entity
VERSION=`cat VERSION`
docker build . -t ${DOCKERNAME}
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:${VERSION}
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:latest
