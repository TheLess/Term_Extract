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
            'CN': ['攻击力+10', '生命值恢复', '魔法防御提升', ''],
            'EN': ['Attack +10', 'HP Recovery', 'Magic Defense Up', 'Test']
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
        invalid_data = self.sample_data.drop('CN', axis=1)
        self.assertFalse(self.processor.validate_data(invalid_data))
        
    def test_is_valid_term(self):
        """测试术语有效性验证功能"""
        # 测试有效术语
        valid_terms = [
            "攻击力",
            "生命值",
            "魔法防御",
            "技能冷却"
        ]
        for term in valid_terms:
            self.assertTrue(
                self.processor._is_valid_term(term),
                f"Term '{term}' should be valid"
            )
            
        # 测试无效术语
        invalid_terms = [
            "A",  # 非中文开头
            "攻",  # 单字
            "攻击力10",  # 包含数字
            "攻击力+",  # 包含特殊字符
            "attack",  # 非中文
            "",  # 空字符串
            "攻击力+10",  # 包含数字和特殊字符
        ]
        for term in invalid_terms:
            self.assertFalse(
                self.processor._is_valid_term(term),
                f"Term '{term}' should be invalid"
            )
            
    def test_build_term_corpus(self):
        """测试术语库构建功能"""
        texts = [
            "攻击力+10",  # 应该提取出"攻击力"
            "生命值-5",   # 应该提取出"生命值"
            "魔法防御提升+20",  # 应该提取出"魔法防御"和"魔法防御提升"
            "技能冷却时间"  # 应该提取出"技能冷却"和"冷却时间"
        ]
        term_dict = self.processor.build_term_corpus(texts)
        
        # 验证基础术语提取
        expected_terms = {
            "攻击力",
            "生命值",
            "魔法防御",
            "魔法防御提升",
            "技能冷却",
            "冷却时间"
        }
        
        # 验证提取的术语都在预期集合中
        for term in term_dict:
            self.assertIn(
                term, 
                expected_terms,
                f"Unexpected term '{term}' was extracted"
            )
            
        # 验证数值修饰语已被正确移除
        self.assertNotIn("攻击力+10", term_dict)
        self.assertNotIn("生命值-5", term_dict)
        self.assertNotIn("提升+20", term_dict)
        
        # 验证频率计数
        # 注意：由于分词和组合的原因，具体频率可能会变化
        # 这里主要验证频率是否为正数
        for term, freq in term_dict.items():
            self.assertGreater(
                freq, 
                0,
                f"Term '{term}' has invalid frequency {freq}"
            )
            
    def test_clean_text(self):
        """测试文本清理功能"""
        # 配置clean_text的行为
        self.processor.config.data.remove_punctuation = True
        self.processor.config.data.remove_numbers = False
        self.processor.config.data.lowercase = False
        
        test_cases = [
            ("攻击力+10！", "攻击力10"),  # 只移除标点
            ("生命值  恢复", "生命值  恢复"),  # 保留空格
            ("魔法防御!!!", "魔法防御"),  # 移除多个标点符号
            ("HP+100%", "HP100"),  # 只移除特殊字符，保留数字
            (np.nan, ""),  # 处理空值
            (123, "123"),  # 保留数字
            (None, ""),  # 处理None
        ]
        
        for input_text, expected in test_cases:
            cleaned = self.processor.clean_text(input_text)
            self.assertEqual(
                cleaned, 
                expected,
                f"Clean text failed for '{input_text}'. Expected '{expected}', got '{cleaned}'"
            )
            
        # 测试不同的配置组合
        test_config_cases = [
            # (remove_punct, remove_nums, lowercase, input, expected)
            (True, True, False, "攻击力+10", "攻击力"),  # 移除标点和数字
            (False, True, False, "攻击力+10", "攻击力+"),  # 只移除数字
            (True, False, False, "HP+100", "HP100"),  # 只移除标点
            (True, False, True, "HP+100", "hp100"),  # 移除标点并转小写
        ]
        
        for remove_punct, remove_nums, lowercase, input_text, expected in test_config_cases:
            self.processor.config.data.remove_punctuation = remove_punct
            self.processor.config.data.remove_numbers = remove_nums
            self.processor.config.data.lowercase = lowercase
            
            cleaned = self.processor.clean_text(input_text)
            self.assertEqual(
                cleaned, 
                expected,
                f"Clean text failed for '{input_text}' with config "
                f"(punct={remove_punct}, nums={remove_nums}, lower={lowercase}). "
                f"Expected '{expected}', got '{cleaned}'"
            )
            
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
