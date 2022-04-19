#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=test_rev_custom2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=small
#SBATCH --gres=gpu:1

CONSTRAINTS="${HOME}/3D-RetinaNet/constraints/full"
CLIP="10."
CENTRALITY="rev-custom2"
CCN_CUSTOM_ORDER="7,40,38,23,39,27,11,33,28,5,22,21,24,20,31,16,26,15,9,6,18,34,19,30,25,3,12,2,10,36,8,35,4,32,1,37,29,14,13,17,0"

#VAL_SUBSETS="val_1"
VAL_SUBSETS="test"

EXP_NAME="resnet50I3D512-Pkinetics-b8s8x1x1-roadt1-h3x3x3-rev-custom2c10.000000-04-02-11-09-28x"

sh test.sh $CONSTRAINTS $CLIP $CENTRALITY $CCN_CUSTOM_ORDER $VAL_SUBSETS $EXP_NAME
