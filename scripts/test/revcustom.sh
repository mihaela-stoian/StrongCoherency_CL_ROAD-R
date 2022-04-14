#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=test_rev_custom
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=long
#SBATCH --gres=gpu:1

CONSTRAINTS="${HOME}/3D-RetinaNet/constraints/full"
CLIP="10."
CENTRALITY="rev-custom"
CCN_CUSTOM_ORDER="7,40,23,5,27,28,39,18,19,21,38,22,16,11,20,33,9,30,3,24,4,15,34,31,25,26,13,17,29,37,14,36,12,35,6,0,10,32,2,1,8"

#VAL_SUBSETS="val_1"
VAL_SUBSETS="test"

EXP_NAME="resnet50I3D512-Pkinetics-b8s8x1x1-roadt1-h3x3x3-rev-customc10.000000-03-28-14-50-06x"

sh test.sh $CONSTRAINTS $CLIP $CENTRALITY $CCN_CUSTOM_ORDER $VAL_SUBSETS $EXP_NAME
