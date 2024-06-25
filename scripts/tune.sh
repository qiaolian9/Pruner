#!/bin/bash
if [ $# != 6 ]  && [ $# != 7 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/tune.sh [network] [trails] [cost_model] [gpu_node] [DEVICE_ID] [use_psa_model] [cost_model_pretrain_ckpt](optional) "
  echo "for example: bash scripts/tune.sh  networks trails cost_model gpu_node use_psa_model whether_use_pretrain "
  echo "=============================================================================================================="
  exit 1
fi

NUM_FOLDS=1
expr $5 + 1 &> /dev/null
if [ $? != 0 ]; then
    echo "DEVICE_ID=$5 is not an integer"
    exit 1
fi

DEVICE_ID=$5
USE_PSA_MODEL=$6

if [ $# != 7 ]; then
    COSTMODELCKPT="wopretrain"
    substring="wopretrain"
else
    substring=$7
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
NETWORK=$1
TRAILS=$2
COSTMODEL=$3
NODE=$4
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
WORKLOAD_OUTPUT=${PROJECT_DIR}/res/$NETWORK-node$NODE
if [ ! -d $WORKLOAD_OUTPUT ]; then
    echo ${WORKLOAD_OUTPUT}
    mkdir $WORKLOAD_OUTPUT
fi
TRAIN_OUTPUT=${WORKLOAD_OUTPUT}/${TRAILS}_${COSTMODEL}_${substring}_${USE_PSA_MODEL}

if [ -d $TRAIN_OUTPUT ]; then
    echo $WORKLOAD_OUTPUT
    rm -rf $TRAIN_OUTPUT
fi
mkdir $TRAIN_OUTPUT
cd $TRAIN_OUTPUT || exit

cp ../../../tune_network.py ./
cp ../../../dump_network_info.py ./
cp ../../../common.py ./
cp ../../../search.py ./
cp -r ../../../model_workload ./

env > env.log
echo $NETWORK
echo $TRAILS
echo $COSTMODEL
echo $DEVICE_ID
echo $substring

for num in $(seq 1 ${NUM_FOLDS})
do
    mkdir $num
    echo "FOLD-$num"
    echo "workload-$NETWORK"
    echo "trails-$TRAILS"
    echo "costmodel-$COSTMODEL"
    echo "PSA-$USE_PSA_MODEL"
    echo "deviceID-$DEVICE_ID"
    echo "ckpt-$substring"
    if [ "$substring" = "wopretrain" ] ; then
        CUDA_VISIBLE_DEVICES=$DEVICE_ID python tune_network.py \
                    --network $NETWORK \
                    --n-trials $TRAILS \
                    --cost-model $COSTMODEL \
                    --psa $USE_PSA_MODEL
    else
        # finetune
        COSTMODELCKPT="./ckpt/a100/500k/$substring.pkl"
        echo "$COSTMODELCKPT"
        CUDA_VISIBLE_DEVICES=$DEVICE_ID python tune_network.py \
                    --network $NETWORK \
                    --n-trials $TRAILS \
                    --cost-model $COSTMODEL \
                    --load-model $COSTMODELCKPT \
                    --psa $USE_PSA_MODEL
    
    fi
    mv *.tsv ./$num
    mv *.json ./$num
done