# This script runs one model end-to-end

#MODEL NAMES
# googlenet  inception_v3  mnasnet1_0  mobilenet_v2  resnet18  resnet50  resnext50_32x4d  vgg16

# Stage 1.1 & 1.2
export MODELNAME=resnet18
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
GETSTATSYAML=classification_stats/get_stats_pq.yaml

export PYTHON=~/anaconda3/envs/env1/bin/python
export DATASET=/data/datasets/imagenet2017/ILSVRC/Data/CLS-LOC

CUDA_VISIBLE_DEVICES=0 $PYTHON tools/bit_search/compression/compress_classifier.py --pretrained -j 4 -b 512 --arch $MODELNAME  --data $DATASET --evaluate --quantize-eval --qe-lapq --qe-config-file $GETSTATSYAML

# Stage 1.3
$PYTHON tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME

# Step 1.4.1: Run Autosplit algorithm.
$PYTHON tools/bit_search/auto_split.py --arch $MODELNAME


# BREAK AT THIS POINT AND RUN THE REST OF THE SCRIPT AFTER MANUALLY SETTING THE TIMESTAMP FOLDER PATH
# Step 1.4.2: Generate Latency Stats
export TIMESTAMPFOLDER=resnet50_20201002-202338
$PYTHON tools/bit_search/model_latency_2_gen.py -n $TIMESTAMPFOLDER

# Step 1.4.3: Collect Latency Stats
# Need to wait until previous script is finished.
# Collect stats generated in the previous step.
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER


# Step 2.1: Set yaml configurations for all bit configurations
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME

# Step 2.2: Run various bit configurations on distiller LAPQ
export DATASET='~/datasets/ImageNet2017/ILSVRC/Data/CLS-LOC/'
$PYTHON  tools/run_quantization/run_distiller.py -n $TIMESTAMPFOLDER --arch $MODELNAME $DATASET

# Step 2.3 Collect accuracy stats from the previous run
$PYTHON tools/run_quantization/get_accuracy.py -n $TIMESTAMPFOLDER