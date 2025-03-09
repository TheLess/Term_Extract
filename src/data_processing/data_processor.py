import pandas as pd
import spacy
import jieba
import re
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional
from collections import Counter

class DataProcessor:
    def __init__(self, config):
        """初始化数据处理器
        
        Args:
            config: 配置对象，包含数据处理相关的配置
        """
        self.config = config
        
        # 首先设置日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 然后初始化模型
        self._initialize_nlp()
        jieba.initialize()
        
    def _setup_logging(self):
        """配置日志"""
        # 使用配置中的日志目录
        log_dir = self.config.logs_dir
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
        
    def _initialize_nlp(self):
        """初始化NLP模型"""
        try:
            if self.config.model.spacy['use_local']:
                model_path = self.config.model.spacy['local_path']
                # 确保model_path是Path对象
                if isinstance(model_path, str):
                    model_path = Path(model_path)
                if not model_path.exists():
                    self.logger.warning(f"本地模型路径不存在: {model_path}")
                    raise OSError(f"找不到本地模型: {model_path}")
                self.nlp = spacy.load(str(model_path))
                self.logger.info(f"从本地加载spacy模型: {model_path}")
            else:
                model_name = self.config.model.spacy['name']
                self.nlp = spacy.load(model_name)
                self.logger.info(f"加载spacy模型: {model_name}")
        except OSError as e:
            self.logger.warning(f"加载模型失败: {str(e)}")
            model_name = self.config.model.spacy['name']
            self.logger.info(f"尝试下载spacy模型: {model_name}")
            try:
                from spacy.cli import download
                download(model_name)
                self.nlp = spacy.load(model_name)
                self.logger.info(f"成功下载并加载模型: {model_name}")
            except Exception as e:
                self.logger.error(f"下载模型失败: {str(e)}")
                raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据格式是否正确
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            bool: 验证是否通过
        """
        # 检查必需的列
        required_cols = [self.config.data.source_lang_col] + self.config.data.target_lang_cols
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
        if pd.isna(text):
            return ""
            
        # 确保文本是字符串类型
        text = str(text)
            
        # 移除标点
        if self.config.data.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
            
        # 移除数字
        if self.config.data.remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        # 转小写（针对英文）
        if self.config.data.lowercase:
            text = text.lower()
            
        return text.strip()
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """加载并预处理数据
        
        Returns:
            Optional[pd.DataFrame]: 处理后的数据，如果加载失败则返回None
        """
        try:
            df = pd.read_excel(str(self.config.data.translations_file))
            
            # 验证数据
            if not self.validate_data(df):
                return None
                
            # 清理文本
            df['text_clean'] = df[self.config.data.source_lang_col].apply(self.clean_text)
            
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
                
            # 更严格的预处理：移除所有数字和加减号
            clean_text = re.sub(r'[-+]?\d+', '', text).strip()
            if not clean_text:  # 如果清理后为空，跳过
                continue
                
            # 使用jieba分词
            words = jieba.lcut(clean_text)
            
            # 提取潜在术语（2-3个词的组合）
            for n in range(2, 4):
                for i in range(len(words)-n+1):
                    term = ''.join(words[i:i+n])
                    # 严格的过滤规则
                    if self._is_valid_term(term):
                        term_counter[term] = term_counter.get(term, 0) + 1
            
            # 单独处理可能的单词术语（如"攻击力"）
            for word in words:
                if self._is_valid_term(word):
                    term_counter[word] = term_counter.get(word, 0) + 1
                    
        # 使用配置中的最小频率
        min_freq = self.config.data.min_term_freq
        term_counter = {k: v for k, v in term_counter.items() 
                       if v >= min_freq}
                       
        self.logger.info(f"提取出 {len(term_counter)} 个术语")
        return term_counter
        
    def _is_valid_term(self, term: str) -> bool:
        """检查是否为有效的游戏术语
        
        Args:
            term: 待检查的术语
            
        Returns:
            bool: 是否为有效术语
        """
        # 长度检查
        if not (self.config.data.min_term_length <= len(term) <= 
                self.config.data.max_term_length):
            return False
            
        # 必须以中文开头
        if not re.match(r'^[\u4e00-\u9fff]', term):
            return False
            
        # 不能是纯数字或包含数字
        if re.search(r'\d', term):
            return False
            
        # 不能只有一个字
        if len(term) <= 1:
            return False
            
        # 不能包含特殊字符
        if re.search(r'[^\u4e00-\u9fff]', term):
            return False
            
        return True

    def save_term_db(self, term_dict: Dict[str, int]) -> None:
        """保存术语库，使用UTF-8编码确保多语言支持
        
        Args:
            term_dict: 术语及其频率的字典
        """
        try:
            output_path = Path(self.config.data.term_db_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8', errors='surrogateescape') as f:
                json.dump(term_dict, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"术语库已保存到 {output_path}")
        except Exception as e:
            self.logger.error(f"保存术语库失败: {str(e)}")
            raise

    def load_term_db(self) -> Dict[str, int]:
        """加载术语库，使用UTF-8编码确保多语言支持
        
        Returns:
            Dict[str, int]: 术语字典
        """
        try:
            with open(self.config.data.term_db_file, encoding='utf-8', errors='surrogateescape') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("术语库文件不存在，返回空字典")
            return {}
        except json.JSONDecodeError:
            self.logger.error("术语库文件格式错误")
            return {}
        except Exception as e:
            self.logger.error(f"加载术语库失败: {str(e)}")
            return {}
