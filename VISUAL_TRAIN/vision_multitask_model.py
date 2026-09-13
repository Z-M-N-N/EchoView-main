import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionMultiTask(nn.Module):
    def __init__(self, visual, num_binary_tasks=28, reg_tasks=7, 
                 unfreeze_layers=0, unfreeze_merger=False):
        super().__init__()
        self.visual = visual
        self.num_binary_tasks = num_binary_tasks
        self.reg_tasks = reg_tasks
        
        # 冻结视觉编码器
        for param in self.visual.parameters():
            param.requires_grad = False
        if unfreeze_layers != 0:
            self._unfreeze_layers(unfreeze_layers, unfreeze_merger)
        
        # 获取hidden_size
        self.hidden_size = visual.merger.mlp[-1].out_features
        
        # ===== 极简头部：只有1层线性层 =====
        # 27个二分类任务 -> 27个logits
        self.classifier = nn.Linear(self.hidden_size, num_binary_tasks)
        
        # 7个回归任务 -> 7个值
        self.reg_head = nn.Linear(self.hidden_size, reg_tasks)
        
        # 初始化
        self._init_weights()
        self._print_trainable_params()
    
    def _init_weights(self):
        """轻量初始化"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.xavier_uniform_(self.reg_head.weight)
        nn.init.constant_(self.reg_head.bias, 0.0)
    
    def _unfreeze_layers(self, unfreeze_layers, unfreeze_merger=False):
        """解冻视觉编码器最后N层"""
        if hasattr(self.visual, 'blocks'):
            total_blocks = len(self.visual.blocks)
            
            if unfreeze_layers == -1:
                for param in self.visual.parameters():
                    param.requires_grad = True
                print(f"✓ Unfreezing all {total_blocks} layers")
            else:
                unfreeze_start = max(0, total_blocks - unfreeze_layers)
                for i in range(unfreeze_start, total_blocks):
                    for param in self.visual.blocks[i].parameters():
                        param.requires_grad = True
                print(f"✓ Unfreezing last {unfreeze_layers} layers (blocks {unfreeze_start}-{total_blocks-1})")
            
            if unfreeze_merger and hasattr(self.visual, 'merger'):
                for param in self.visual.merger.parameters():
                    param.requires_grad = True
                print("✓ Also unfreezing merger layer")
    
    def _print_trainable_params(self):
        trainable_vision = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        total_vision = sum(p.numel() for p in self.visual.parameters())
        head_params = sum(p.numel() for p in self.classifier.parameters()) + \
                      sum(p.numel() for p in self.reg_head.parameters())
        
        print(f"📊 Vision encoder: {trainable_vision:,}/{total_vision:,} ({trainable_vision/total_vision*100:.2f}%)")
        print(f"📊 Classifier head: {sum(p.numel() for p in self.classifier.parameters()):,} params")
        print(f"📊 Regression head: {sum(p.numel() for p in self.reg_head.parameters()):,} params")
        print(f"✅ Total trainable: {trainable_vision + head_params:,}")
        print(f"🎯 Head params占比: {head_params/(trainable_vision + head_params)*100:.2f}%")
    
    def forward(self, pixel_values, grid_thw=None):
        """
        Args:
            pixel_values: [B, C, H, W] or [B, T, C, H, W]
            grid_thw: Qwen的网格信息
        Returns:
            cls_logits: [B, 27]
            reg_values: [B, 5]
        """
        # 提取视觉特征
        if grid_thw is not None:
            vision_features = self.visual(pixel_values, grid_thw=grid_thw)
        else:
            vision_features = self.visual(pixel_values)
        
        # print(vision_features.dim())
        # 聚合时序/空间维度
        if vision_features.dim() == 3:
            vision_features = vision_features.mean(dim=1)
        elif vision_features.dim() == 2 and vision_features.size(0) != pixel_values.size(0):
            batch_size = pixel_values.size(0)
            if vision_features.size(0) % batch_size == 0:
                seq_len = vision_features.size(0) // batch_size
                vision_features = vision_features.view(batch_size, seq_len, -1).mean(dim=1)
        

        if vision_features.dtype != torch.float32:
            vision_features = vision_features.to(torch.float32)
        # 轻量级预测
        cls_logits = self.classifier(vision_features)  # [B, 27]
        reg_values = self.reg_head(vision_features)    # [B, 5]
        
        return cls_logits, reg_values