#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=test_3D_Retinanet
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=long
#SBATCH --gres=gpu:8

source ~/.bashrc
source activate ccn

ROOT="$HOME"
DATASET="${ROOT}/road-dataset/"
KINETICS="${ROOT}/3D-RetinaNet/kinetics-pt"

#CONSTRAINTS="${ROOT}/3D-RetinaNet/constraints/full"
CONSTRAINTS=''

#CLIP="10."
CLIP="100000."

OPTIM="SGD"
CENTRALITY="None"
CCN_NUM_CLASSES="41"
BATCH_SIZE=4

EXP_NAME="resnet50I3D512-Pkinetics-b8s8x1x1-roadt1-h3x3x3-Nonec100000.000000-03-22-11-46-22x"

# CHECK ALL PARAMETERS!!!!!!!!

cd ..
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG="INFO" python main.py $DATASET $DATASET $KINETICS --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=$BATCH_SIZE --LR=0.0041 --CCN_CONSTRAINTS=$CONSTRAINTS --CCN_CENTRALITY=$CENTRALITY --CCN_NUM_CLASSES=$CCN_NUM_CLASSES --OPTIM=$OPTIM --CLIP=$CLIP --EXP_NAME=$EXP_NAME
