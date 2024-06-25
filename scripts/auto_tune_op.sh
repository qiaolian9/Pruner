!/bin/bash

if [ $# != 4 ]  && [ $# != 3 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/tune_op.sh  [cost_model] [gpu_node] [DEVICE_ID] [use_psa_model] [USE_PRETRAIN]"
  echo "for example: bash scripts/auto_tune_op.sh  cost_model gpu_node device_id a100_40 pretrian_model"
  echo "=============================================================================================================="
  exit 1
fi


expr $3 + 1 &> /dev/null
if [ $? != 0 ]; then
    echo "DEVICE_ID=$3 is not an integer"
    exit 1
fi
DEVICE_ID=$3
USE_PSA_MODEL=$4

if [ $# = 4 ]; then
    COSTMODELCKPT="wopretrain"
    substring="wopretrain"
else
    substring=$5
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
COSTMODEL=$1
NODE=$2

declare -A my_dict;

# device 1
list_matmul=("128_128_128" "1024_1024_1024" "512_512_512"  "256_256_256" "128_128_128"  "128_1000_4096"  "65536_1024_2" "512_1024_1024" "1_1000_2048" "32_1024_1024" "64_1000_4096" "64_756_86" "740_1024_500" "512_4096_1024" "1024_4096_1024" "2048_4096_1024"  "4096_4096_1024" "8192_4096_1024" "16384_4096_1024");
list_conv2dS1=("128_128_28_28_128_3_3"  "32_128_256_256_3_3_3" "32_256_128_128_128_3_3" "32_64_64_64_32_4_4" "1_512_8_8_256_4_4" "3_32_114_114_3_3_3" "16_110_71_71_50_5_5" "128_42_83_83_42_1_1" "128_42_83_83_96_1_1" "128_64_64_64_32_1_1" "128_336_21_21_336_1_1" "128_168_42_42_336_1_1");
list_conv2dS1_nchw=("128_42_64_64_42_1_1" "128_42_83_83_96_3_3" "128_96_32_32_32_2_2" "128_64_56_56_256_1_1" "128_84_42_42_168_1_1" "128_84_83_83_168_1_1" "128_128_28_28_512_1_1" "128_168_21_21_1008_1_1" "128_168_42_42_168_1_1" "128_256_16_16_256_1_1" "128_256_28_28_512_1_1" "128_256_56_56_64_1_1" "128_336_21_21_336_3_3" "128_512_14_14_1024_1_1" "128_512_28_28_128_1_1" "32_128_32_32_64_1_1" "64_64_32_32_32_3_3" "64_128_16_16_64_3_3");
list_conv2dS2=("1_2048_14_14_1024_1_1" "32_2048_14_14_1024_1_1" "64_2048_14_14_1024_1_1" "128_2048_14_14_1024_1_1" "256_2048_14_14_1024_1_1" "512_2048_14_14_1024_1_1" "1_128_256_256_3_7_7" "16_128_58_58_128_3_3" "128_256_32_32_256_3_3" "64_512_16_16_256_3_3" "32_64_128_128_16_4_4" "3_32_128_128_16_4_4" "3_512_30_30_256_5_5" "16_128_64_64_32_3_3" "16_128_128_128_8_3_3" "32_64_28_28_16_2_2" "32_128_16_16_64_3_3" "1_128_64_64_64_3_3");
list_conv2dS2_nchw=("128_64_112_112_3_7_7" "128_96_32_32_32_2_2" "128_128_28_28_128_3_3" "128_256_14_14_256_3_3" "128_512_7_7_512_3_3" "128_512_28_28_256_1_1" "128_1024_14_14_512_1_1" "1_32_64_64_32_3_3" "1_128_128_128_32_3_3" "3_64_28_28_32_3_3" "32_256_14_14_128_3_3" "64_128_56_56_64_3_3");
list_fused_conv2dS2=("64_64_32_32_32_3_3" "128_128_28_28_128_3_3" "128_256_14_14_256_3_3" "128_512_7_7_512_3_3" "128_1024_14_14_512_1_1" "128_512_16_16_256_3_3" "128_64_112_112_3_7_7" "128_96_32_32_32_2_2" "128_512_28_28_256_1_1" "64_128_56_56_64_3_3" "1_32_64_64_32_3_3" "1_128_128_128_32_3_3" "3_64_28_28_32_3_3" "32_256_14_14_128_3_3");
list_fused_conv2dS1=("64_128_16_16_64_3_3" "128_168_42_42_168_1_1" "1024_14_14_256_1_1" "128_256_16_16_256_1_1" "128_512_14_14_1024_1_1" "128_168_21_21_1008_1_1" "128_42_64_64_42_1_1" "32_128_32_32_64_1_1" "128_42_83_83_96_1_1" "128_96_32_32_32_2_2" "128_84_83_83_168_1_1" "128_336_21_21_336_1_1" "128_256_56_56_64_1_1" "128_64_56_56_256_1_1" "128_512_28_28_128_1_1" "128_128_28_28_512_1_1" "128_84_42_42_168_1_1" "128_256_28_28_512_1_1");
list_dwConv2dS1=("32_256_14_14_3_3" "128_64_32_32_3_3" "3_128_114_114_3_3" "1_32_56_56_3_3" "128_42_83_83_5_5" "128_84_21_21_3_3" "32_256_128_128_3_3" "32_512_64_64_4_4" "64_128_114_114_3_3" "64_256_56_56_3_3" "1_1024_64_64_4_4");
list_dwConv2dS2=("128_16_32_32_3_3" "128_128_64_64_4_4"); 
list_avgPoolS1=("64_32_83_83_3_3" "32_512_16_16_2_2");
list_avgPoolS2=("128_168_83_83_4_4" "128_617_21_21_3_3");
list_maxPoolS2=("64_1024_53_53_5_5");
list_reduc1=("65536_1024" "1024_8192");
list_reduc2=("128_512_1024");
list_reduc3=("128_4032_11_11" "32_1000_128_128");

my_dict["matmul"]="${list_matmul[@]}"
my_dict["conv_expr_S1D1P0"]="${list_conv2dS1[@]}"
my_dict["conv_expr_S2D1P0"]="${list_conv2dS2[@]}"
my_dict["conv_expr_S1D1P0_NCHW"]="${list_conv2dS1_nchw[@]}"
my_dict["conv_expr_S2D1P0_NCHW"]="${list_conv2dS2_nchw[@]}"
my_dict["fused_conv_expr_S1D1P0"]="${list_fused_conv2dS1[@]}"
my_dict["fused_conv_expr_S2D1P0"]="${list_fused_conv2dS2[@]}"
my_dict["depthwiseconv_expr_S1D1P1"]="${list_dwConv2dS1[@]}"
my_dict["depthwiseconv_expr_S2D1P0"]="${list_dwConv2dS2[@]}"
my_dict["avgpool2d_expr_S1P0"]="${list_avgPoolS1[@]}"
my_dict["avgpool2d_expr_S2P0"]="${list_avgPoolS2[@]}"
my_dict["maxpool2d_expr_S2P1"]="${list_maxPoolS2[@]}"
my_dict["reduce_expr1"]="${list_reduc1[@]}"
my_dict["reduce_expr2"]="${list_reduc2[@]}"
my_dict["reduce_expr3"]="${list_reduc3[@]}"

echo "All keys:"
for key in "${!my_dict[@]}"; do
  echo $key
done

for OP in "${!my_dict[@]}"; do
  echo $OP
  for value in ${my_dict[${OP}]}; do
    for SHAPE in "${value[@]}"; do
      cd /staff/Anonymous/pruner/scripts
      echo $OP $SHAPE

      PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
      WORKLOAD_OUTPUT=${PROJECT_DIR}/res/singleop/${COSTMODEL}_${substring}_psa/${OP}-node${NODE}/
      if [ ! -d $WORKLOAD_OUTPUT ]; then
          echo $WORKLOAD_OUTPUT
          mkdir $WORKLOAD_OUTPUT
      fi
      TRAIN_OUTPUT=${WORKLOAD_OUTPUT}/${SHAPE}

      if [ -d $TRAIN_OUTPUT ]; then
          echo $WORKLOAD_OUTPUT
          rm -rf $TRAIN_OUTPUT
      fi
      mkdir $TRAIN_OUTPUT
      cd $TRAIN_OUTPUT || exit

      cp /staff/Anonymous/pruner/scripts/singleop.py ./
      cp -r /staff/Anonymous/pruner/scripts/test_config ./

      env > env.log
      echo $OP
      echo $SHAPE
      echo $DEVICE_ID
      echo $COSTMODEL
      if [ "$COSTMODELCKPT" = "wopretrain" ] ; then
          python singleop.py \
              --op $OP \
              --shape $SHAPE \
              --cost-model $COSTMODEL \
              --psa_model_type $USE_PSA_MODEL
      else
          COSTMODELCKPT="/staff/Anonymous/pruner/scripts/ckpt/$substring.pkl"
          python singleop.py \
              --op $OP \
              --shape $SHAPE \
              --cost-model $COSTMODEL \
              --load-model $COSTMODELCKPT
      fi
    done
  done
done


