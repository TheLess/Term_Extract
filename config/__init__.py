from pathlib import Path
from .data_config import DataConfig
from .model_config import ModelConfig

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

class Config:
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        
        # 设置输出目录
        self.output_dir = ROOT_DIR / 'outputs'
        self.model_dir = self.output_dir / 'models'
        self.terms_dir = self.output_dir / 'terms'
        self.logs_dir = self.output_dir / 'logs'
        
        # 创建必要的目录
        for dir_path in [self.output_dir, self.model_dir, self.terms_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
