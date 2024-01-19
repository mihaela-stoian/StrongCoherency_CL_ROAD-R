#!/bin/bash



source $HOME/.bashrc
source activate slowfast_road 
export LD_LIBRARY_PATH=~/software/miniconda3/envs/slowfast_road/lib:$LD_LIBRARY_PATH

ROOT="$HOME"
DATASET="${ROOT}/datasets/"
KINETICS="${ROOT}/projects/pretrained_models/kinetics-pt/"
EXPDIR="${ROOT}/projects/ccn_plus/"



OPTIM="SGD"
CCN_NUM_CLASSES="41"
BATCH_SIZE=4

CONSTRAINTS=$1
CLIP=$2
CENTRALITY=$3
CCN_CUSTOM_ORDER=$4
VAL_SUBSETS=$5
EXP_NAME=$6
MODEL=$7

echo "Testing with parameters:"
echo "Constraints: $CONSTRAINTS"
echo "Clip: $CLIP"
echo "Centrality: $CENTRALITY ($CCN_CUSTOM_ORDER)"
echo "Validation Subset: $VAL_SUBSETS"
echo "Experiment Name: $EXP_NAME"
echo "Model Type: $MODEL"

NCCL_DEBUG="INFO" python main.py $DATASET $EXPDIR $KINETICS --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=$MODEL --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=$VAL_SUBSETS --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=$BATCH_SIZE --LR=0.0041 --CCN_CONSTRAINTS=$CONSTRAINTS --CCN_CENTRALITY=$CENTRALITY --CCN_CUSTOM_ORDER=$CCN_CUSTOM_ORDER --CCN_NUM_CLASSES=$CCN_NUM_CLASSES --OPTIM=$OPTIM --CLIP=$CLIP --EXP_NAME=$EXP_NAME --IOU_THRESH=0.5
