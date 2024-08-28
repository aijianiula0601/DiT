#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

#-------------------------
#解压所有文件
#-------------------------
#bash unzip_images.sh


data_path='/mnt/cephfs/hjh/common_dataset/images/imagenet/ILSVRC2012_img_train'
# data_path = '/mnt/cephfs/hjh/common_dataset/images/imagenet/debug_dataset'
cd ../../

#-------------------------
# 训练
#-------------------------
your_random_port=65036
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 --master_port=${your_random_port} check_images.py ${data_path}
