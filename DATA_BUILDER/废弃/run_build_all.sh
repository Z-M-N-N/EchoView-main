#!/bin/bash

if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="/home/mzhao/ECHO_VIEW/EchoView-main"
    cd "${EchoView_dir}/DATA_BUILDER" || exit 1

else
    CONDA_BASE="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/miniconda3"
    EchoView_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/EchoView-main"
    cd "${EchoView_dir}/DATA_BUILDER" || exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Qwen_back


data_dir="../../Data/dicom_videos_group/Dataset"
python process_vcr.py --data_dir "$data_dir" --mode "vcr" --mode_dir_name "Train_Test_JSON"

python process_pvd_mcd_cmd.py --data_dir "$data_dir" --mode "pvd" --mode_dir_name "Train_Test_JSON"
python process_pvd_mcd_cmd.py --data_dir "$data_dir" --mode "mcd" --mode_dir_name "Train_Test_JSON"
python process_pvd_mcd_cmd.py --data_dir "$data_dir" --mode "cmd" --mode_dir_name "Train_Test_JSON"


echo "ALL DONE"
