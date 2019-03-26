#!/bin/bash
EXP="exp0"

kubectl delete -f ${EXP}/detector.yaml
kubectl apply -f ${EXP}/detector.yaml

./setup_screen.sh ${EXP} 160224
sleep 5
./setup_screen.sh ${EXP} 160226
sleep 5
./setup_screen.sh ${EXP} 160422
sleep 5
./setup_screen.sh ${EXP} 161202