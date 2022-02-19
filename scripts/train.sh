#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=train_3D_Retinanet
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=long
#SBATCH --gres=gpu:4

source ~/.bashrc
source activate ccn

DATASET='/jmain02/home/J2AD009/ttl04/aat50-ttl04/road-dataset/'
KINETICS='/jmain02/home/J2AD009/ttl04/aat50-ttl04/3D-RetinaNet/kinetics-pt'
CONSTRAINTS='/jmain02/home/J2AD009/ttl04/aat50-ttl04/3D-RetinaNet/constraints/full'

cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $DATASET $DATASET $KINETICS --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --CCN_CONSTRAINTS=$CONSTRAINTS
