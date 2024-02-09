#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=RCLSTM-train_3D_Retinanet
#SBATCH --mail-type=ALL
#SBATCH --partition=long
#SBATCH --gres=gpu:8

source $HOME/.bashrc
source activate slowfast_road 
export LD_LIBRARY_PATH=~/software/miniconda3/envs/slowfast_road/lib:$LD_LIBRARY_PATH

ROOT="$HOME"
DATASET="${ROOT}/datasets/"
EXPDIR="${ROOT}/projects/ccn_plus_strong_coherency/"
KINETICS="${ROOT}/projects/pretrained_models/kinetics-pt/"

CONSTRAINTS="./constraints/full"

CLIP="10."
OPTIM="SGD"
CCN_NUM_CLASSES="41"
BATCH_SIZE=8
model="RCLSTM"

#CENTRALITY="rev-custom2"  # rev_freq
#CCN_CUSTOM_ORDER="7,40,38,23,39,27,11,33,28,5,22,21,24,20,31,16,26,15,9,6,18,34,19,30,25,3,12,2,10,36,8,35,4,32,1,37,29,14,13,17,0"

CENTRALITY="rev-custom"
CCN_CUSTOM_ORDER="7,40,23,5,27,28,39,18,19,21,38,22,16,11,20,33,9,30,3,24,4,15,34,31,25,26,13,17,29,37,14,36,12,35,6,0,10,32,2,1,8"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG="INFO" python main.py $DATASET $EXPDIR $KINETICS --MODE=train --ARCH=resnet50 --MODEL_TYPE=$model --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=$BATCH_SIZE --LR=0.0041 --CCN_CONSTRAINTS=$CONSTRAINTS --CCN_CENTRALITY=$CENTRALITY --CCN_CUSTOM_ORDER=$CCN_CUSTOM_ORDER --CCN_NUM_CLASSES=$CCN_NUM_CLASSES --OPTIM=$OPTIM --CLIP=$CLIP
