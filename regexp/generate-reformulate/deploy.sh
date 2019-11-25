DOCKERNAME=regexp-reformulate
VERSION=`cat VERSION`

docker push arsenal-docker.cse.sri.com/${DOCKERNAME}:${VERSION}
docker push arsenal-docker.cse.sri.com/${DOCKERNAME}:latest