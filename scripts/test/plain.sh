#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=test_plain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=small
#SBATCH --gres=gpu:1

CONSTRAINTS=""
CLIP="100000."
CENTRALITY="None"
CCN_CUSTOM_ORDER=""

#VAL_SUBSETS="val_1"
VAL_SUBSETS="test"

EXP_NAME="resnet50I3D512-Pkinetics-b8s8x1x1-roadt1-h3x3x3-Nonec100000.000000-03-22-11-46-22x"

sh test.sh "$CONSTRAINTS" $CLIP $CENTRALITY "$CCN_CUSTOM_ORDER" $VAL_SUBSETS $EXP_NAME
