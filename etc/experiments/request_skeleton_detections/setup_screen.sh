#!/bin/bash

EXP_FOLDER=$1
SEQUENCE=$2
CURRENT_FOLDER=`pwd`
REQUEST_FOLDER="/home/felippe/dev/skeleton-joints-3d-reconstruction"

sed -e "s/{sequence}/${SEQUENCE}_haggling1/g" -e "s/{exp}/${EXP_FOLDER}/g" request_options.json >&1 | tee /tmp/request_options.json

cd $REQUEST_FOLDER
screen -S "${SEQUENCE}" -d -m bash -c 'python3 -m bin.request.skeleton_detection --config-file /tmp/request_options.json; exec bash'
cd $CURRENT_FOLDER
