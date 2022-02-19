#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=test_3D_Retinanet
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=long
#SBATCH --gres=gpu:4

source ~/.bashrc
source activate ccn

DATASET='/jmain02/home/J2AD009/ttl04/aat50-ttl04/road-dataset/'
KINETICS='/jmain02/home/J2AD009/ttl04/aat50-ttl04/3D-RetinaNet/kinetics-pt'

cd ..
python main.py $DATASET $DATASET $KINETICS --MODE=gen_dets --EVAL_EPOCHS=31 --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_1 --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 
