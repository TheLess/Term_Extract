import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.trainer import ModelTrainer
from src.data_processing.data_processor import DataProcessor
from config.model_config import ModelConfig
from config.data_config import DataConfig
import pandas as pd
import logging
import tempfile

# 设置基本的日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_term_extraction():
    # 准备测试数据
    test_data = pd.DataFrame({
        'text_clean': [
            "角色的生命值和攻击力都很高",
            "这个技能可以恢复生命值",
            "生命值上限提高了",
            "攻击力加成效果",
            "魔法攻击力很强"
        ]
    })
    
    # 创建临时目录作为数据目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 初始化配置
        data_config = DataConfig()
        data_config.data_dir = Path(temp_dir)
        
        # 初始化数据处理器
        data_processor = DataProcessor(data_config)
        terms = data_processor.build_term_corpus(test_data['text_clean'].tolist())
        
        # 验证结果
        print("\n提取的术语:")
        for term, freq in sorted(terms.items(), key=lambda x: x[1], reverse=True):
            print(f"{term}: {freq}")
        
        # 确保关键术语被正确提取
        assert "生命值" in terms, "未能提取出'生命值'术语"
        assert "攻击力" in terms, "未能提取出'攻击力'术语"
        
        return terms

if __name__ == "__main__":
    test_term_extraction()
