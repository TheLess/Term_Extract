import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from config import Config
from src.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建临时目录
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # 创建配置
        cls.config = Config()
        cls.config.data.translations_file = cls.test_dir / 'test_translations.xlsx'
        cls.config.data.term_db_file = cls.test_dir / 'test_term_db.json'
        
        # 创建示例数据
        cls.sample_data = pd.DataFrame({
            '简体中文': ['攻击力+10', '生命值恢复', '魔法防御提升', ''],
            'English': ['Attack +10', 'HP Recovery', 'Magic Defense Up', 'Test'],
            '日本語': ['攻撃力+10', 'HP回復', '魔法防御アップ', 'テスト'],
            '한국어': ['공격력 +10', 'HP 회복', '마법 방어 증가', '테스트']
        })
        cls.sample_data.to_excel(cls.config.data.translations_file, index=False)
        
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        shutil.rmtree(cls.test_dir)
        
    def setUp(self):
        """每个测试用例的设置"""
        self.processor = DataProcessor(self.config)
        
    def test_load_data(self):
        """测试数据加载功能"""
        df = self.processor.load_data()
        
        # 验证数据加载成功
        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(self.sample_data))
        
        # 验证清理后的文本列存在
        self.assertIn('text_clean', df.columns)
        
    def test_validate_data(self):
        """测试数据验证功能"""
        # 测试有效数据
        self.assertTrue(self.processor.validate_data(self.sample_data))
        
        # 测试缺少必需列的数据
        invalid_data = self.sample_data.drop('简体中文', axis=1)
        self.assertFalse(self.processor.validate_data(invalid_data))
        
    def test_clean_text(self):
        """测试文本清理功能"""
        # 测试标点符号移除
        text = "攻击力+10！"
        cleaned = self.processor.clean_text(text)
        self.assertNotIn("！", cleaned)
        
        # 测试空值处理
        self.assertEqual(self.processor.clean_text(np.nan), "")
        
    def test_build_term_corpus(self):
        """测试术语库构建功能"""
        texts = ["攻击力+10", "攻击力", "生命值", "生命值恢复"]
        term_dict = self.processor.build_term_corpus(texts)
        
        # 验证术语提取
        self.assertIn("攻击力", term_dict)
        self.assertIn("生命值", term_dict)
        
        # 验证频率计数
        self.assertEqual(term_dict["攻击力"], 2)
        
    def test_term_db_operations(self):
        """测试术语库保存和加载功能"""
        test_terms = {"攻击力": 2, "生命值": 1}
        
        # 保存术语库
        self.processor.save_term_db(test_terms)
        
        # 验证文件已创建
        self.assertTrue(self.config.data.term_db_file.exists())
        
        # 加载并验证内容
        loaded_terms = self.processor.load_term_db()
        self.assertEqual(loaded_terms, test_terms)
        
if __name__ == '__main__':
    unittest.main()
