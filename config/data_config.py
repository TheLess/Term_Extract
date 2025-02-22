from pathlib import Path

class DataConfig:
    def __init__(self):
        # 数据文件路径
        self.data_dir = Path('data')
        self.translations_file = self.data_dir / 'translations.xlsx'
        self.term_db_file = self.data_dir / 'term_database.json'
        
        # 数据处理参数
        self.min_term_length = 2  # 最小术语长度
        self.min_term_freq = 2    # 最小术语频率
        self.max_term_length = 10 # 最大术语长度
        
        # 必需的列名
        self.source_lang_col = '简体中文'
        self.target_lang_cols = ['English', '日本語', '한국어']  # 支持的目标语言
        
        # 数据清理参数
        self.remove_punctuation = True  # 是否移除标点
        self.remove_numbers = False     # 是否移除数字
        self.lowercase = False          # 是否转小写（针对英文）
