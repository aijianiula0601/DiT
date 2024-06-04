#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../

echo "----:$(pwd)"

data_path='/mnt/cephfs/hjh/common_dataset/images/imagenet/my_test'
results_dir='/mnt/cephfs/hjh/train_record/images/DiT/training'

your_random_port=65036

CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --nproc_per_node=2 --master_port=${your_random_port} train.py \
  --data-path ${data_path} \
  --results-dir ${results_dir} \
  --model 'DiT-XL/2' \
  --num-workers 8 \
  --log-every 100 \
  --ckpt-every 5000
