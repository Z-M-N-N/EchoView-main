#!/bin/bash

if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="/home/mzhao/ECHO_VIEW/EchoView-main"
    cd "${EchoView_dir}/SPLIT_SFT_BUILDER" || exit 1

else
    CONDA_BASE="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/miniconda3"
    EchoView_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/EchoView-main"
    cd "${EchoView_dir}/SPLIT_SFT_BUILDER" || exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Qwen_back


python split_data.py "../../Data/Echo-View/dicom_videos_group" \
    --test-size 0.2 \
    --output "../../Data/Echo-View/Train_Test_JSON" 