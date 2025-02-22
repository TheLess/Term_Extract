import pandas as pd
import spacy
import jieba
from spacy.lang.zh import Chinese
import re
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional

class DataProcessor:
    def __init__(self, config):
        """初始化数据处理器
        
        Args:
            config: 配置对象，包含数据处理相关的配置
        """
        self.config = config
        self.nlp = Chinese()
        self.nlp.add_pipe('sentencizer')
        jieba.initialize()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """配置日志"""
        # 使用数据目录下的logs子目录
        log_dir = self.config.data_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'data_processor.log'
        
        # 配置日志
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据格式是否正确
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            bool: 验证是否通过
        """
        # 检查必需的列
        required_cols = [self.config.source_lang_col] + self.config.target_lang_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"缺少必需的列: {missing_cols}")
            return False
            
        # 检查空值
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"发现空值:\n{null_counts[null_counts > 0]}")
            
        return True
        
    def clean_text(self, text: str) -> str:
        """清理文本
        
        Args:
            text: 待清理的文本
            
        Returns:
            str: 清理后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 移除标点
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
            
        # 移除数字
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        # 转小写（针对英文）
        if self.config.lowercase:
            text = text.lower()
            
        return text.strip()
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """加载并预处理数据
        
        Returns:
            Optional[pd.DataFrame]: 处理后的数据，如果加载失败则返回None
        """
        try:
            df = pd.read_excel(str(self.config.translations_file))
            
            # 验证数据
            if not self.validate_data(df):
                return None
                
            # 清理文本
            df['text_clean'] = df[self.config.source_lang_col].apply(self.clean_text)
            
            self.logger.info(f"成功加载数据，共 {len(df)} 行")
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return None

    def build_term_corpus(self, texts: List[str]) -> Dict[str, int]:
        """构建术语库
        
        Args:
            texts: 文本列表
            
        Returns:
            Dict[str, int]: 术语及其频率
        """
        term_counter = {}
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            # 使用spacy进行分句
            doc = self.nlp(text)
            for sent in doc.sents:
                # 使用jieba分词并保留较长的词组
                words = jieba.lcut(sent.text)
                
                # 提取n-gram特征
                for n in range(1, 4):  # 提取1-3gram
                    for i in range(len(words)-n+1):
                        term = ''.join(words[i:i+n])
                        if (self.config.min_term_length <= len(term) <= 
                            self.config.max_term_length):
                            term_counter[term] = term_counter.get(term, 0) + 1
                        
        # 按频率过滤
        term_counter = {k: v for k, v in term_counter.items() 
                       if v >= self.config.min_term_freq}
                       
        self.logger.info(f"提取出 {len(term_counter)} 个术语")
        return term_counter

    def save_term_db(self, term_dict: Dict[str, int]) -> None:
        """保存术语库
        
        Args:
            term_dict: 术语字典
        """
        try:
            with open(self.config.term_db_file, 'w', encoding='utf-8') as f:
                json.dump(term_dict, f, ensure_ascii=False, indent=2)
            self.logger.info(f"术语库已保存到 {self.config.term_db_file}")
        except Exception as e:
            self.logger.error(f"保存术语库失败: {str(e)}")

    def load_term_db(self) -> Dict[str, int]:
        """加载术语库
        
        Returns:
            Dict[str, int]: 术语字典
        """
        try:
            with open(self.config.term_db_file, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("术语库文件不存在，返回空字典")
            return {}
