#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tf_hub

TRAINING_DATA_DIR=$1
[ -z "$TRAINING_DATA_DIR" ]  &&  echo "usage:  /path/to/training/data"  &&  exit 1

if [ ! -d $TRAINING_DATA_DIR ] ; then 
    echo "does not exist: $TRAINING_DATA_DIR"
    exit 2
fi

TS="$(date '+%Y%m%d%H%M')"
SM_DIR=saved_model_$TS
mkdir -p $SM_DIR pretrained_model
mkdir -p out_$TS/intermediate out_$TS/bottleneck  out_$TS/summaries out_$TS/graphs 

# these might help
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=16

# default ARCH_BASIS is InceptionV3
#STEPS=1600

# To see possible precomputed embeddings, see https://tfhub.dev/s?module-type=image-feature-vector
#inception
#STEPS=5000
#ARCH_BASIS="--tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/3"
#ARCH_BASIS="--tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/3"
#STEPS=8000
#ARCH_BASIS="--tfhub_module https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3"
#STEPS=2000
ARCH_BASIS="--tfhub_module https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/3"
#STEPS=2000
#ARCH_BASIS="--tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/3"
#STEPS=6000
#ARCH_BASIS="--tfhub_module h https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/3"
#STEPS=5000
#ARCH_BASIS="--tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/3"
#STEPS=3000
#ARCH_BASIS="--tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/3"
STEPS=7000

LEARNING_RATE=0.001
#LEARNING_RATE=0.001
echo "===================Config: STEPS=$STEPS ARCH_BASIS=$ARCH_BASIS"
# numactl --cpunodebind=0 --membind=0  
./tf_hub_image_classifier.py --image_dir=./training_data $ARCH  --how_many_training_steps $STEPS --output_graph=./retrained_graph.pb \
  --output_labels=out_$TS/retrained_labels.txt --bottleneck_dir=out_$TS/bottleneck \
  --intermediate_output_graphs_dir=out_$TS/graphs --summaries_dir=out_$TS/summaries \
  --model_dir=out_$TS/model --saved_model_dir=$SM_DIR  --image_dir=$TRAINING_DATA_DIR  \
  --testing_percentage=10 --learning_rate=$LEARNING_RATE --validation_percentage=10 \
  --validation_batch_size=-1  --print_misclassified_test_images

echo "See SavedModel in $SM_DIR"
