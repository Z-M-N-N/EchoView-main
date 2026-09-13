import re
from pathlib import Path

# 基础路径

ROOT_PATH = Path("")

# 定义数据集配置的基类（使用dataclass或普通函数）
def make_config(annotation_path, data_path=""):
    """创建数据集配置"""
    return {
        "annotation_path": annotation_path,
        "data_path": data_path
    }



def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names,data_json):
    # 定义各个数据集的路径映射（避免重复定义train/test）
    train_path=data_json.replace("test","train")
    test_path=data_json.replace("train","test")
    print("使用的训练文件:",train_path)
    print("使用的测试文件:",test_path)
    DATASET_PATHS = {
        # 数据集名称: (训练集路径, 测试集路径)
        "ECHO_HEART": (
            train_path,
            test_path,
        )
    }

    # 批量生成配置
    data_dict = {}

    for name, (train_path, test_path) in DATASET_PATHS.items():
        # 训练集
        data_dict[f"{name}_train"] = make_config(str(ROOT_PATH / train_path))
        # 测试集
        data_dict[f"{name}_test"] = make_config(str(ROOT_PATH / test_path))



    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
