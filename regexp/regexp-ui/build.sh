DOCKERNAME=regexp-ui
VERSION=`cat VERSION`

if [ -z $1 ] || [ $1 != "--skip-yarn" ]
then
    yarn run build
fi

docker build -f docker/Dockerfile . -t $DOCKERNAME
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:${VERSION}
docker tag ${DOCKERNAME} arsenal-docker.cse.sri.com/${DOCKERNAME}:latest
