#!/bin/bash

# JUST FOR TESTING PURPOSES - NO NEED TO RUN THIS SCRIPT
# this script downloads specified transformers version and 
# adds custom patch to enable type forcing
# (corresponding actions are already included in Dockerfile)
tr_version="4.4.2"
wget -O "v"$tr_version".tar.gz" "https://github.com/huggingface/transformers/archive/v"$tr_version".tar.gz"
tar -xf "v"$tr_version".tar.gz"
cp *.py "transformers-"$tr_version"/src/transformers/"