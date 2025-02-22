from pathlib import Path

class ModelConfig:
    def __init__(self):
        # 模型基本配置
        self.base_model_name = 'hfl/chinese-macbert-large'  # 在线模型名称
        self.local_model_path = Path(r'E:\CyberLife\models\chinese-macbert-large')  # 本地模型路径，如果设置则优先使用本地模型
        self.use_local_model = True  # 是否使用本地模型
        
        # LoRA配置
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = ["query", "value", "key"]
        
        # 训练配置
        self.train_batch_size = 32 
        self.num_train_epochs = 3
        self.learning_rate = 2e-5
        self.warmup_ratio = 0.1
        self.random_seed = 42
        self.mask_probability = 0.3
        self.min_cluster_size = 5
