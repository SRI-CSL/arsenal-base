#!/bin/bash

echo "Computing version number"
GRAMMAR=`md5 -q _build/default/src/generate.exe`
GRAMMARDATE=`date -r _build/default/src/generate.exe  "+%Y-%m-%dT%H%M%z"`
VERSION="${GRAMMARDATE}_${GRAMMAR:0:7}"
echo "Version is $VERSION"
echo "Checking if git work tree is clean"
status=$( git status -s src | grep "M " )
if [ -z "$status" ]
then
    HASH=$( git log -n 1 --format=format:"%h" )
else
    echo "git work tree unclean; please commit before generating dataset."
    exit 1
fi
echo "Checking if version was already recorded"
if grep -q $VERSION version
then
    echo "Version already present";
else
    # echo "Getting last version";
    # LASTVERSION=`cat version`
    # echo "Checking last version $LASTVERSION is present";
    # grep -q $LASTVERSION versions || exit 1     
    echo "Writing into version";
    echo -n "$VERSION" > version;
    if grep -q $VERSION versions
    then 
        echo "Version already present";
    else
        echo "Recording version and git commit to versions file"
        echo "$VERSION $HASH" >> versions;
    fi
fi
