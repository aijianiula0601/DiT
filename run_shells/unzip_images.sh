#!/bin/bash

set -ex

reading_dir='/mnt/cephfs/hjh/common_dataset/images/imagenet/ILSVRC2012_img_train'

for f_name in ${reading_dir}/*.tar; do
  f_base_name=$(basename $f_name .tar)
  tar_base_dir="${reading_dir}/${f_base_name}"
  mkdir -p ${tar_base_dir}
  echo "doing unzip ${tar_base_dir}.tar"
  tar -xxf ${tar_base_dir}.tar -C ${tar_base_dir}
  echo "${tar_base_dir}.tar done!"
  rm -rf ${tar_base_dir}.tar
done
