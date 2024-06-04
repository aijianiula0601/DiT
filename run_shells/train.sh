#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

#解压所有文件
bash unzip_images.sh

cd ../

echo "----:$(pwd)"

data_path='/mnt/cephfs/hjh/common_dataset/images/imagenet/ILSVRC2012_img_train'
results_dir='/mnt/cephfs/hjh/train_record/images/DiT/training'

your_random_port=65036

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 --master_port=${your_random_port} train.py \
  --data-path ${data_path} \
  --results-dir ${results_dir} \
  --model 'DiT-XL/2' \
  --num-workers 8 \
  --log-every 100 \
  --ckpt-every 5000 \
  --num-classes 1000 \
  --global-batch-size 32
