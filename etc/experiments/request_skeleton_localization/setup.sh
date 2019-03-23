#!/bin/bash

EXP_FOLDER=$1
DATASET=$2
CURRENT_FOLDER=`pwd`
REQUEST_FOLDER="/home/felippe/dev/skeleton-joints-3d-reconstruction"

kubectl delete -f frame_transformation.yaml
sed -e "s/{dataset}/${DATASET}_haggling1/g" frame_transformation.yaml | kubectl apply -f -

kubectl delete -f $EXP_FOLDER/skeletons_grouper.yaml
kubectl apply -f $EXP_FOLDER/skeletons_grouper.yaml

sleep 5

sed -e "s/{sequence_folder}/${DATASET}_haggling1/g" $EXP_FOLDER/request_options.json >&1 | tee /tmp/request_options.json
cd $REQUEST_FOLDER
screen -S "exp" -d -m bash -c 'python3 -m bin.request.skeleton_localization --config-file /tmp/request_options.json; exec bash'
cd $CURRENT_FOLDER
