import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LoRA 线性层：挂到视觉编码器的注意力/MLP 线性层上做低秩适配
# 保存时可用 get_merged_state_dict() 把 LoRA 合并回原权重，
# 导出格式与原始 visual.* 完全一致（兼容 predict / merge 脚本）
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, base_weight, base_bias,
                 r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # 保留原权重副本，参数名与 nn.Linear 对齐（weight/bias）
        self.weight = nn.Parameter(base_weight.detach().clone())
        if base_bias is not None:
            self.bias = nn.Parameter(base_bias.detach().clone())
        else:
            self.register_parameter('bias', None)
        # LoRA：A 用 kaiming 初始化，B 全零（初始等价于原层，不破坏预训练特征）
        # ⚠️ dtype 跟随基础权重（如 bf16），避免与输入精度不匹配
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=self.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=self.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        lora = self.dropout(x)
        # 前向时与输入精度对齐，兼容模型整体被转成 fp16/bf16/fp32 的情况
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        lora = lora @ a.t()
        lora = lora @ b.t()
        return out + lora * self.scaling

    def merged_weight(self):
        """返回合并 LoRA 后的权重（与 forward 的 base + lora*scaling 完全等价）"""
        return self.weight.float() + (self.lora_B.float() @ self.lora_A.float()) * self.scaling


# ============================================================
# 可学习时序注意力池化：把 [B, T, H] 加权聚合为 [B, H]
# 相比直接 mean，能让模型自己学习每一帧的重要性（对心超等强时序任务更友好）
# ============================================================
class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.scale = hidden_size ** -0.5

    def forward(self, x):  # x: [B, T, H]
        attn = torch.einsum('bth,h->bt', x, self.query) * self.scale
        attn = torch.softmax(attn, dim=1)
        return (x * attn.unsqueeze(-1)).sum(dim=1)


class VisionMultiTask(nn.Module):
    def __init__(self, visual, num_binary_tasks=28, reg_tasks=7,
                 unfreeze_layers=0, unfreeze_merger=False,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.0,
                 lora_target='attn', temporal_pool='mean'):
        super().__init__()
        self.visual = visual
        self.num_binary_tasks = num_binary_tasks
        self.reg_tasks = reg_tasks
        self.use_lora = use_lora
        self.temporal_pool = temporal_pool

        # 冻结视觉编码器
        for param in self.visual.parameters():
            param.requires_grad = False

        # 注入 LoRA（只替换 blocks 内的注意力/MLP 线性层，LoRA 参数保持可训练）
        if use_lora:
            self._inject_lora(lora_r, lora_alpha, lora_dropout, lora_target)
            if unfreeze_layers != 0:
                print(f"⚠️ 使用 LoRA 时同时解冻最后 {unfreeze_layers} 层视觉基础权重，"
                      f"两层适配叠加，请确认这是有意的")
            self._unfreeze_layers(unfreeze_layers, unfreeze_merger)
        else:
            if unfreeze_layers != 0:
                self._unfreeze_layers(unfreeze_layers, unfreeze_merger)

        # 获取hidden_size
        self.hidden_size = visual.merger.mlp[-1].out_features

        # ===== 极简头部：只有1层线性层 =====
        self.classifier = nn.Linear(self.hidden_size, num_binary_tasks)
        self.reg_head = nn.Linear(self.hidden_size, reg_tasks)

        # 可学习时序池化（仅 temporal_pool='attention' 时创建，保持旧 checkpoint 兼容）
        if temporal_pool == 'attention':
            self._temporal_pool = TemporalAttentionPool(self.hidden_size)
        elif temporal_pool != 'mean':
            raise ValueError(f"temporal_pool 仅支持 'mean' 或 'attention'，收到: {temporal_pool}")

        self._init_weights()
        self._print_trainable_params()

    def _inject_lora(self, r, alpha, dropout, target):
        """把视觉编码器 blocks 内的目标线性层替换为 LoRALinear"""
        if target == 'attn':
            leaf_set = {'qkv', 'proj'}
        elif target == 'mlp':
            leaf_set = {'gate_proj', 'up_proj', 'down_proj'}
        elif target == 'attn+mlp':
            leaf_set = {'qkv', 'proj', 'gate_proj', 'up_proj', 'down_proj'}
        else:
            raise ValueError(f"lora_target 仅支持 'attn'/'mlp'/'attn+mlp'，收到: {target}")

        to_replace = []
        for name, module in self.visual.named_modules():
            parts = name.split('.')
            if 'blocks' not in parts or not isinstance(module, nn.Linear):
                continue
            if parts[-1] in leaf_set:
                to_replace.append((name, module))

        if not to_replace:
            print(f"⚠️ 未找到任何可注入 LoRA 的层（target={target}），请检查视觉编码器结构")
        replaced = 0
        for full_name, lin in to_replace:
            parent_name, leaf = full_name.rsplit('.', 1)
            parent = self.visual.get_submodule(parent_name)
            new_lin = LoRALinear(lin.in_features, lin.out_features,
                                 lin.weight, lin.bias, r=r, alpha=alpha, dropout=dropout)
            # LoRA 模式下基础权重保持冻结，仅训练 lora_A/lora_B
            new_lin.weight.requires_grad = False
            if new_lin.bias is not None:
                new_lin.bias.requires_grad = False
            setattr(parent, leaf, new_lin)
            replaced += 1
        print(f"✅ 已注入 LoRA：{replaced} 个线性层（target={target}, r={r}, alpha={alpha}, dropout={dropout}）")

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
        if hasattr(self, '_temporal_pool'):
            head_params += sum(p.numel() for p in self._temporal_pool.parameters())

        print(f"📊 Vision encoder: {trainable_vision:,}/{total_vision:,} ({trainable_vision/total_vision*100:.2f}%)")
        print(f"📊 Classifier head: {sum(p.numel() for p in self.classifier.parameters()):,} params")
        print(f"📊 Regression head: {sum(p.numel() for p in self.reg_head.parameters()):,} params")
        print(f"✅ Total trainable: {trainable_vision + head_params:,}")
        print(f"🎯 Head params占比: {head_params/(trainable_vision + head_params)*100:.2f}%")

    def get_merged_state_dict(self):
        """
        导出可直接用于 predict/merge 的 state_dict：
        - LoRA 已合并回 visual.* 权重（权重格式与原始模型完全一致）
        - 不含 lora_A/lora_B 键
        """
        sd = self.state_dict()
        if not self.use_lora:
            return sd
        remove = []
        # 每层只处理一次（只遍历 lora_B 键，lora_A 是其对应配对），
        # 避免同一层被合并两次导致 scaling 重复作用
        for name in list(sd.keys()):
            if not name.endswith('.lora_B'):
                continue
            prefix = name[:name.rfind('.')]
            a_name = prefix + '.lora_A'
            b_name = prefix + '.lora_B'
            w_name = prefix + '.weight'
            if a_name not in sd or w_name not in sd:
                continue
            scaling = self.get_submodule(prefix).scaling
            merged = sd[w_name].float() + (sd[b_name].float() @ sd[a_name].float()) * scaling
            sd[w_name] = merged.to(sd[w_name].dtype)
            remove.extend([a_name, b_name])
        for k in set(remove):
            sd.pop(k, None)
        return sd

    def forward(self, pixel_values, grid_thw=None):
        """
        Args:
            pixel_values: [B, C, H, W] or [B, T, C, H, W]
            grid_thw: Qwen的网格信息
        Returns:
            cls_logits: [B, num_binary_tasks]
            reg_values: [B, reg_tasks]
        """
        if grid_thw is not None:
            vision_features = self.visual(pixel_values, grid_thw=grid_thw)
        else:
            vision_features = self.visual(pixel_values)

        # 聚合时序维度
        if vision_features.dim() == 3:
            if self.temporal_pool == 'attention':
                vision_features = self._temporal_pool(vision_features)
            else:
                vision_features = vision_features.mean(dim=1)
        elif vision_features.dim() == 2 and vision_features.size(0) != pixel_values.size(0):
            batch_size = pixel_values.size(0)
            if vision_features.size(0) % batch_size == 0:
                seq_len = vision_features.size(0) // batch_size
                vision_features = vision_features.view(batch_size, seq_len, -1)
                if self.temporal_pool == 'attention':
                    vision_features = self._temporal_pool(vision_features)
                else:
                    vision_features = vision_features.mean(dim=1)

        if vision_features.dtype != torch.float32:
            vision_features = vision_features.to(torch.float32)

        cls_logits = self.classifier(vision_features)  # [B, 28]
        reg_values = self.reg_head(vision_features)    # [B, 7]

        return cls_logits, reg_values
