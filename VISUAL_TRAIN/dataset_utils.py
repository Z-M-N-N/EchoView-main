
import json

import torch
from torch.utils.data import Dataset
from myqwen_vl_utils import process_vision_info

# 导入多任务模型
# ============ 数据加载 ============
# def load_jsonl(file_path):
#     """加载JSONL格式文件（每行一个JSON对象）"""
#     data = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:  # 跳过空行
#                 try:
#                     data.append(json.loads(line))
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing line: {e}")
#                     continue
#     return data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_videos_multilabels(json_file):
    """加载多任务标签数据"""
    json_list = load_json(json_file)
    videos = []
    binary_labels = []
    regression_labels = []
    
    for j in json_list:
        try:
            # 根据您的数据格式调整
            video_path = j["video"]
            if isinstance(video_path, list):
                video_path = video_path[0]
            
            videos.append(video_path)
            binary_labels.append(j["binary_labels"])
            regression_labels.append([float(x) for x in j["regression_labels"]])
            
        except Exception as e:
            print(f"Error processing video: {e}")
            continue
    
    print(f"Loaded {len(videos)} videos")
    if len(videos) > 0:
        print(f"Binary labels: {len(binary_labels[0])} tasks")
        print(f"Regression labels: {len(regression_labels[0])} tasks")
    return videos, binary_labels, regression_labels

# ============ Dataset ============
class VideoMultiTaskDataset(Dataset):
    def __init__(self, video_paths, binary_labels, regression_labels, processor, num_frames=8):
        self.video_paths = video_paths
        self.binary_labels = binary_labels
        self.regression_labels = regression_labels
        self.processor = processor
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        binary_label = torch.tensor(self.binary_labels[idx], dtype=torch.float32)
        regression_label = torch.tensor(self.regression_labels[idx], dtype=torch.float32)
        messages = [
                        {"role": "user", "content": [
                            {"type": "video", "video": video_path,
                                "nframes": self.num_frames,                    # 每秒1帧
                                "min_pixels": 224 * 224,
                                "max_pixels": 224 * 224
                              },
                            # {"type": "text", "text": prompt}
                        ]}
                    ]

        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=text, 
            images=images, 
            videos=videos, 
            padding=True, 
            return_tensors="pt", 
            **video_kwargs
        )

        
        pixel_values_videos = inputs["pixel_values_videos"].squeeze(0)
        video_grid_thw = inputs["video_grid_thw"].squeeze(0)
        target_T = 8  # nframes / temporal_patch_size = 16 / 2
        patches_per_frame = 256  # 16 * 16
        
        current_T = video_grid_thw[0].item()
        
        if current_T != target_T:
            # 记录
            # with open('t_fixed.txt', 'a') as f:
            #     f.write(f"Sample {idx}: T={current_T} -> {target_T}, path={video_path}\n")
            
            if current_T < target_T:
                # 帧数不足，复制最后一帧填充
                frames_to_add = target_T - current_T
                last_frame = pixel_values_videos[-patches_per_frame:, :]
                padding = last_frame.repeat(frames_to_add, 1)
                pixel_values_videos = torch.cat([pixel_values_videos, padding], dim=0)
            else:
                # 帧数过多，截取
                pixel_values_videos = pixel_values_videos[:target_T * patches_per_frame, :]
            
            # 修正 grid_thw
            video_grid_thw = torch.tensor([target_T, 16, 16], dtype=video_grid_thw.dtype)
        return {
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "binary_labels": binary_label,
        "regression_labels": regression_label
        }
        return {
            "pixel_values_videos": inputs["pixel_values_videos"].squeeze(0),
            "video_grid_thw": inputs["video_grid_thw"].squeeze(0),
            "binary_labels": binary_label,
            "regression_labels": regression_label
        }