DOCKERNAME=noop-entity
VERSION=`cat VERSION`

if [ -z $1 ] || [ $1 != "--skip-lein" ]
then
    lein ring uberjar
fi

docker build . -t ${DOCKERNAME}
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:${VERSION}
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:latest

