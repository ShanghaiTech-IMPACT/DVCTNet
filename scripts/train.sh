#!/bin/bash

conda activate DVCTNet
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(realpath "${script_dir}/../")
export PYTHONPATH="$PYTHONPATH:${project_dir}/mmdet_custom"

python ${project_dir}/tools/train.py \
    ${project_dir}/configs/models/dvctnet_dinov2_base_fpn_50_epoch.py

# # for multi-gpu training
# ${project_dir}/tools/dist_train.sh \
#     ../configs/models/dvctnet_dinov2_base_fpn_50_epoch.py <GPU_NUM>