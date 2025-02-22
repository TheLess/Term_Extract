import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import torch
import logging
import time

from config import Config
from src.model import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建临时目录
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # 创建配置
        cls.config = Config()
        cls.config.model_dir = cls.test_dir / 'models'
        cls.config.logs_dir = cls.test_dir / 'logs'
        cls.config.model_dir.mkdir(exist_ok=True)
        cls.config.logs_dir.mkdir(exist_ok=True)
        
        # 使用实际的本地模型路径进行测试
        cls.config.model.local_model_path = Path('E:/CyberLife/models/bert-base-chinese')
        cls.config.model.use_local_model = True
        
        # 创建示例数据
        cls.sample_data = pd.DataFrame({
            'text_clean': ['攻击力提升', '生命值恢复', '魔法防御提升'] * 10
        })
        
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        try:
            # 确保所有文件句柄都已关闭
            logging.shutdown()
            # 等待一小段时间确保文件被释放
            time.sleep(1)
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            print(f"清理测试环境时出错: {str(e)}")
        
    def setUp(self):
        """每个测试用例的设置"""
        self.trainer = ModelTrainer(self.config)
        
    def test_model_initialization(self):
        """测试模型初始化"""
        # 验证模型和分词器是否正确初始化
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.tokenizer)
        
        # 验证设备设置
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.assertEqual(self.trainer.device, expected_device)
        
        # 验证是否使用了本地模型
        self.assertTrue(self.config.model.use_local_model)
        self.assertTrue(self.config.model.local_model_path.exists())
        
    def test_preprocess_data(self):
        """测试数据预处理"""
        # 创建包含text_clean列的测试数据
        test_data = pd.DataFrame({
            'text_clean': ['攻击力提升', '生命值恢复', '魔法防御提升']
        })
        
        dataset = self.trainer._preprocess_data(test_data)
        
        # 验证数据集创建成功
        self.assertIsNotNone(dataset)
        self.assertIn('input_ids', dataset.features)
        self.assertIn('labels', dataset.features)
        
    def test_term_embeddings(self):
        """测试术语向量计算"""
        terms = ['攻击力', '生命值', '魔法防御']
        embeddings = self.trainer._get_term_embeddings(terms)
        
        # 验证向量维度
        self.assertEqual(embeddings.shape[0], len(terms))
        self.assertEqual(embeddings.shape[1], 768)  # BERT的隐藏层维度
        
    def test_term_clustering(self):
        """测试术语聚类"""
        # 创建示例向量
        embeddings = np.random.rand(10, 768)
        clusters = self.trainer._cluster_terms(embeddings)
        
        # 验证聚类结果
        self.assertEqual(len(clusters), len(embeddings))
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in clusters))
        
    def test_term_refinement(self):
        """测试术语优化"""
        term_dict = {
            '攻击力提升': 5,
            '攻击力增加': 3,
            '生命值恢复': 4,
            'HP恢复': 2,
            '攻击力上升': 1  # 添加更多相似术语以确保合并
        }
        
        refined_terms = self.trainer.refine_terms(term_dict)
        
        # 验证术语合并
        self.assertLess(len(refined_terms), len(term_dict))
        
        # 验证频率合并
        for term, freq in refined_terms.items():
            self.assertGreaterEqual(freq, term_dict.get(term, 0))
        
    def test_model_save_load(self):
        """测试模型保存和加载"""
        # 保存模型
        save_path = self.config.model_dir / 'test_model'
        self.trainer.model.save_pretrained(save_path)
        
        # 验证模型文件已创建
        self.assertTrue(save_path.exists())
        
if __name__ == '__main__':
    unittest.main()
