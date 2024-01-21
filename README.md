# Strongly Coherent Constraint Layers for ROAD-R
This repository contains baseline code for the first and second tasks of the [ROAD-R Challenge](https://sites.google.com/view/road-r/).
The code is built on top of [3D-RetinaNet for ROAD](https://github.com/gurkirt/road-dataset).

The task requires that the models' predictions are compliant with the 243 requirements provided in `constraints/requirements.txt`.

## Table of Contents
- <a href='#dep'>Dependencies and data preparation</a>
- <a href='#training'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#prostprocessing'>Post-processing</a>



## Dependencies and data preparation
This repository requires [CLOVERD](https://github.com/mihaela-stoian/CLOVERD).
Alternatively, the code for building constraint layers [strongly coherent](https://github.com/mihaela-stoian/StrongCoherencyCCN) with requirements can be obtained by:
```
git clone git@github.com:mihaela-stoian/StrongCoherencyCCN.git ccn
```

For the dataset preparation and packages required to train the models, please see the [Requirements](https://github.com/gurkirt/3D-RetinaNet#requirements) section from 3D-RetinaNet for ROAD.  

To download the pretrained weights, please see the end of the [Performance](https://github.com/gurkirt/3D-RetinaNet#performance) section from 3D-RetinaNet for ROAD.

## Training

To train the model, provide the following positional arguments:
 - `DATA_ROOT`: path to a directory in which `road` can be found, containing `road_test_v1.0.json`, `road_trainval_v1.0.json`, and directories `rgb-images` and `videos`.
 - `SAVE_ROOT`: path to a directory in which the experiments (e.g. checkpoints, training logs) will be saved.
 - `MODEL_PATH`: path to the directory containing the weights for the chosen backbone (e.g. `resnet50RCGRU.pth`).

Example train command (to be run from the root of this repository):

```
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
BATCH_SIZE=8


model="RCGRU"
python main.py $DATASET $EXPDIR $KINETICS --MODE=train \
    --ARCH=resnet50 --MODEL_TYPE=$model --DATASET=road \
    --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 \
    --SEQ_LEN=8 --TEST_SEQ_LEN=8 \
    --BATCH_SIZE=$BATCH_SIZE --LR=0.0041 --OPTIM=$OPTIM --CLIP=$CLIP \
    --CCN_CONSTRAINTS=$CONSTRAINTS \
    --CCN_CENTRALITY=$CENTRALITY \
    --CCN_CUSTOM_ORDER=$CCN_CUSTOM_ORDER \
    --CCN_NUM_CLASSES=$CCN_NUM_CLASSES 
```

## Testing 
Below is an example command to test a model.

```
CUDA_VISIBLE_DEVICES=1 python main.py $DATASET $EXPDIR $KINETICS \
    --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=$MODEL --DATASET=road \
    --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=8 \
    --BATCH_SIZE=$BATCH_SIZE --LR=0.0041 --OPTIM=$OPTIM --CLIP=$CLIP \
    --CCN_CONSTRAINTS=$CONSTRAINTS \
    --CCN_CENTRALITY=$CENTRALITY \
    --CCN_CUSTOM_ORDER=$CCN_CUSTOM_ORDER \
    --CCN_NUM_CLASSES=$CCN_NUM_CLASSES \
    --EXP_NAME=$EXP_NAME --IOU_THRESH=0.5
```

This command will generate a file containing the detected boxes.