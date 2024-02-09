#!/bin/bash

CONSTRAINTS="./constraints/full"
DATASET="${HOME}/datasets/"
EXPDIR="${HOME}/experiments/ccn_plus_strong_coherency/"
KINETICS="${HOME}/projects/pretrained_models/kinetics-pt/"
###########################

CLIP="10."
OPTIM="SGD"
CENTRALITY="rev-custom2"   
CCN_CUSTOM_ORDER="7,40,38,23,39,27,11,33,28,5,22,21,24,20,31,16,26,15,9,6,18,34,19,30,25,3,12,2,10,36,8,35,4,32,1,37,29,14,13,17,0"
CCN_NUM_CLASSES="41"
BATCH_SIZE=4


model="RCGRU"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py $DATASET $EXPDIR $KINETICS --MODE=train \
    --ARCH=resnet50 --MODEL_TYPE=$model --DATASET=road \
    --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 \
    --SEQ_LEN=8 --TEST_SEQ_LEN=8 \
    --BATCH_SIZE=$BATCH_SIZE --LR=0.0041 --OPTIM=$OPTIM --CLIP=$CLIP \
    --CCN_CONSTRAINTS=$CONSTRAINTS \
    --CCN_CENTRALITY=$CENTRALITY \
    --CCN_CUSTOM_ORDER=$CCN_CUSTOM_ORDER \
    --CCN_NUM_CLASSES=$CCN_NUM_CLASSES 
