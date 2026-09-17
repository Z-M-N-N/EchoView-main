#!/bin/bash

if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="/home/mzhao/ECHO_VIEW/EchoView-main/DATA_BUILDER"
    data_dir="/home/mzhao/ECHO_VIEW/Data/dicom_videos_group/Dataset"
    cache_file="/home/mzhao/ECHO_VIEW/Data/dicom_videos_group/subfolders_cache.json"

    cd "${EchoView_dir}" || exit 1

else
    CONDA_BASE="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/miniconda3"
    EchoView_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/EchoView-main/DATA_BUILDER"
    data_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/Data_back/dicom_videos_group/Dataset"
    cache_file="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/Data_back/dicom_videos_group/subfolders_cache.json"
    cd "${EchoView_dir}" || exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Qwen_back


# python process_vcr.py --data_dir "$data_dir" --mode "vcr" --mode_dir_name "Train_Test_JSON" --cache_file "$cache_file"

# python process_pvd_avd_cmd.py --data_dir "$data_dir" --mode "pvd" --mode_dir_name "Train_Test_JSON" --cache_file "$cache_file"
python process_pvd_avd_cmd.py --data_dir "$data_dir" --mode "avd" --mode_dir_name "Train_Test_JSON" --cache_file "$cache_file"
python process_pvd_avd_cmd.py --data_dir "$data_dir" --mode "cmd" --mode_dir_name "Train_Test_JSON" --cache_file "$cache_file"


echo "ALL DONE"
