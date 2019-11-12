#!/usr/bin/env bash

pushd data
./extract.sh
popd
cat pretrained_data/* > glove.6B.100d.txt
