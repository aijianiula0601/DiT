#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

#-------------------------
#解压所有文件
#-------------------------
#bash unzip_images.sh


data_path='/mnt/cephfs/hjh/common_dataset/images/imagenet/debug_dataset'
results_dir='/mnt/cephfs/hjh/train_record/images/DiT/training_debug'
cd ../../

#-------------------------
# 训练
#-------------------------
your_random_port=65038
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --nproc_per_node=2 --master_port=${your_random_port} train.py \
  --data-path ${data_path} \
  --results-dir ${results_dir} \
  --model 'DiT-XL/2' \
  --num-workers 4 \
  --log-every 100 \
  --ckpt-every 5000 \
  --num-classes 2 \
  --global-batch-size 8 \
  --process-name 'debug'