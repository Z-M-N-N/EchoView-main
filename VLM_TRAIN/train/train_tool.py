from transformers import TrainerCallback
class LLMTrainingLossEarlyStopping(TrainerCallback):
    def __init__(self, patience=1000, threshold=0.01, min_steps=500):
        """
        Args:
            patience: 连续多少步损失未改善就停止
            threshold: 改善阈值（绝对下降值）
            min_steps: 最少训练步数，避免过早停止
        """
        self.patience = patience
        self.threshold = threshold
        self.min_steps = min_steps
        self.best_loss = float('inf')
        self.counter = 0
        self.step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or 'loss' not in logs:
            return
            
        self.step += 1
        
        # 前 min_steps 步不检查（大模型训练前期损失波动大）
        if self.step < self.min_steps:
            return
            
        current_loss = logs['loss']
        
        # 检查是否显著改善
        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.counter = 0
            # 可选：打印日志
            if self.step % 100 == 0:
                print(f"✅ Step {self.step}: Loss improved to {current_loss:.4f}")
        else:
            self.counter += 1
            if self.step % 100 == 0:
                print(f"⚠️  Step {self.step}: Loss {current_loss:.4f} not improved ({self.counter}/{self.patience})")
        
        # 触发早停
        if self.counter >= self.patience:
            print(f"🛑 Step {self.step}: Training loss hasn't improved for {self.patience} steps, stopping!")
            control.should_training_stop = True
            control.should_save = True  # 保存当前模型