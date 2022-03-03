#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=train_3D_Retinanet
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=long
#SBATCH --gres=gpu:4

source ~/.bashrc
source activate ccn

ROOT="$HOME"
DATASET="${ROOT}/road-dataset/"
KINETICS="${ROOT}/3D-RetinaNet/kinetics-pt"
CONSTRAINTS="${ROOT}/3D-RetinaNet/constraints/full"
#CONSTRAINTS=''

# CHECK ALL PARAMETERS!!!!!!!!

cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $DATASET $DATASET $KINETICS --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=1 --LR=0.0041 --CCN_CONSTRAINTS=$CONSTRAINTS
