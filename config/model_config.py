from pathlib import Path

class ModelConfig:
    def __init__(self):
        # 模型基本配置
        self.bert = {
            'name': 'hfl/chinese-macbert-large',  # 在线模型名称
            'local_path': Path(r'models\chinese-macbert-large'),  # 本地模型路径
            'use_local': True,  # 是否使用本地模型
        }
        
        # Spacy模型配置
        self.spacy = {
            'name': 'zh_core_web_sm',  # spacy中文模型
            'local_path': Path(r'models\spacy\zh_core_web_sm'),  # 本地模型路径
            'use_local': True,  # 是否使用本地模型
        }
        
        # LoRA配置
        self.lora = {
            'r': 8,              # LoRA秩
            'alpha': 32,         # LoRA alpha参数
            'dropout': 0.1,      # Dropout率
            'bias': 'none',      # 是否使用偏置项
            'target_modules': ['query', 'value']  # 目标模块
        }
        
        # 训练配置
        self.training = {
            'batch_size': 32,
            'learning_rate': 2e-4,
            'num_epochs': 3,
            'warmup_steps': 500,
            'max_grad_norm': 1.0,
            'weight_decay': 0.01
        }
