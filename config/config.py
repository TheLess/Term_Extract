"""统一配置文件

包含所有项目配置参数的单一配置类
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

class Config:
    """统一配置类，包含所有项目参数"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置
        
        Args:
            config_path: 可选的配置文件路径，如果提供则从文件加载配置
        """
        # ===== 基础路径配置 =====
        self.data_dir = ROOT_DIR / 'data'
        self.output_dir = ROOT_DIR / 'outputs'
        self.model_dir = self.output_dir / 'models'
        self.terms_dir = self.output_dir / 'terms'
        self.logs_dir = self.output_dir / 'logs'
        
        # 创建命名空间对象
        self.data = type('DataConfig', (), {})()
        self.model = type('ModelConfig', (), {})()
        
        # ===== 输入输出文件配置 =====
        self.data.translations_file = self.data_dir / 'translations.xlsx'  # 默认输入文件
        self.data.term_db_file = self.terms_dir / 'term_database.json'     # 默认输出文件
        
        # ===== 数据处理参数 =====
        # 术语提取配置
        self.data.min_term_length = 2     # 最小术语长度
        self.data.max_term_length = 10    # 最大术语长度
        self.data.min_term_freq = 1       # 最小术语频率
        
        # 数据列名配置
        self.data.source_lang_col = 'CN'  # 默认源语言列名
        self.data.target_lang_cols = ['EN']  # 默认目标语言列名
        
        # 数据清理配置
        self.data.remove_punctuation = True   # 是否移除标点
        self.data.remove_numbers = False      # 是否移除数字
        self.data.lowercase = False           # 是否转小写
        
        # ===== 模型配置 =====
        # BERT模型配置
        self.model.bert = {
            'name': 'hfl/chinese-macbert-large',  # 在线模型名称
            'local_path': ROOT_DIR / 'models' / 'chinese-macbert-large',  # 本地模型路径
            'use_local': True  # 是否使用本地模型
        }
        
        # Spacy模型配置
        self.model.spacy = {
            'name': 'zh_core_web_sm',  # spacy模型名称
            'local_path': ROOT_DIR / 'models' / 'spacy' / 'zh_core_web_sm',  # 本地路径
            'use_local': True  # 是否使用本地spacy模型
        }
        
        # LoRA配置
        self.model.lora = {
            'r': 8,              # LoRA秩
            'alpha': 32,         # LoRA alpha参数
            'dropout': 0.1,      # Dropout率
            'bias': 'none',      # 是否使用偏置项
            'target_modules': ['query', 'value']  # 目标模块
        }
        
        # 训练配置
        self.model.training = {
            'batch_size': 32,        # 批处理大小
            'learning_rate': 2e-4,   # 学习率
            'num_epochs': 3,         # 训练轮数
            'warmup_steps': 500,     # 预热步数
            'max_grad_norm': 1.0,    # 梯度裁剪
            'weight_decay': 0.01     # 权重衰减
        }
        
        # 术语优化配置
        self.model.term_refinement = {
            'optimization_rounds': 1,  # 术语优化轮次 (默认降低为1轮，避免术语过度损失)
            'preserve_top_terms': True  # 是否保留高频术语
        }
        
        # 保留原来的属性以保持向后兼容
        self.translations_file = self.data.translations_file
        self.term_db_file = self.data.term_db_file
        self.min_term_length = self.data.min_term_length
        self.max_term_length = self.data.max_term_length
        self.min_term_freq = self.data.min_term_freq
        self.source_lang_col = self.data.source_lang_col
        self.target_lang_cols = self.data.target_lang_cols
        self.remove_punctuation = self.data.remove_punctuation
        self.remove_numbers = self.data.remove_numbers
        self.lowercase = self.data.lowercase
        
        self.bert = self.model.bert
        self.spacy = self.model.spacy
        self.lora = self.model.lora
        self.training = self.model.training
        
        self.train_model = True      # 是否训练模型
        
        # ===== 日志配置 =====
        self.log_level = 'INFO'      # 日志级别
        
        # 创建必要的目录
        self._create_directories()
        
        # 如果提供了配置文件路径，则从文件加载
        if config_path:
            self.load_from_file(config_path)
            
    def _create_directories(self):
        """创建必要的目录"""
        for dir_path in [self.output_dir, self.model_dir, self.terms_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        # 特殊处理命名空间属性
        for namespace in ['data', 'model']:
            if namespace in config_dict and hasattr(self, namespace):
                namespace_obj = getattr(self, namespace)
                namespace_dict = {}
                for key in dir(namespace_obj):
                    if not key.startswith('_'):
                        namespace_dict[key] = getattr(namespace_obj, key)
                config_dict[namespace] = namespace_dict
            
        return config_dict
        
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """从字典更新配置"""
        for key, value in config_dict.items():
            if key in ['data', 'model'] and isinstance(value, dict):
                # 特殊处理命名空间属性
                if not hasattr(self, key):
                    setattr(self, key, type(f'{key.capitalize()}Config', (), {})())
                namespace_obj = getattr(self, key)
                for ns_key, ns_value in value.items():
                    setattr(namespace_obj, ns_key, ns_value)
            elif hasattr(self, key):
                if isinstance(value, dict) and isinstance(getattr(self, key), dict):
                    # 合并字典
                    getattr(self, key).update(value)
                else:
                    # 直接替换值
                    setattr(self, key, value)
            else:
                logging.warning(f"未知配置参数: {key}")
                    
    def save_to_json(self, json_path: str) -> None:
        """保存配置到JSON文件
        
        Args:
            json_path: JSON文件路径
        """
        # 将Path对象转换为字符串
        def path_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [path_to_str(item) for item in obj]
            else:
                return obj
                
        # 转换配置
        config_dict = path_to_str(self.to_dict())
        
        # 保存到文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
    def load_from_file(self, config_path: str) -> None:
        """从文件加载配置
        
        Args:
            config_path: 配置文件路径，支持.json和.yaml/.yml格式
        """
        import json
        from pathlib import Path
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        # 根据文件扩展名选择加载方式
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            # 如果需要支持YAML，需要安装pyyaml
            try:
                import yaml
            except ImportError:
                raise ImportError("加载YAML配置需要安装pyyaml: pip install pyyaml")
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
        # 将字符串转换为Path对象
        def str_to_path(obj, keys_to_convert):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and k in keys_to_convert:
                        obj[k] = Path(v)
                    elif isinstance(v, (dict, list)):
                        obj[k] = str_to_path(v, keys_to_convert)
                return obj
            elif isinstance(obj, list):
                return [str_to_path(item, keys_to_convert) for item in obj]
            else:
                return obj
                
        # 需要转换为Path的键
        path_keys = [
            'data_dir', 'output_dir', 'model_dir', 'terms_dir', 'logs_dir',
            'translations_file', 'term_db_file'
        ]
        
        # 转换并更新配置
        config_dict = str_to_path(config_dict, path_keys)
        self.update_from_dict(config_dict)
        
        # 更新嵌套字典中的路径
        if 'data' in config_dict:
            if 'translations_file' in config_dict['data']:
                self.data.translations_file = Path(config_dict['data']['translations_file'])
            if 'term_db_file' in config_dict['data']:
                self.data.term_db_file = Path(config_dict['data']['term_db_file'])
        
        if 'model' in config_dict:
            if 'bert' in config_dict['model'] and 'local_path' in config_dict['model']['bert']:
                self.model.bert['local_path'] = Path(config_dict['model']['bert']['local_path'])
                
            if 'spacy' in config_dict['model'] and 'local_path' in config_dict['model']['spacy']:
                self.model.spacy['local_path'] = Path(config_dict['model']['spacy']['local_path'])
        
        # 向后兼容处理
        for orig_attr, namespace_attr in [
            ('translations_file', ('data', 'translations_file')),
            ('term_db_file', ('data', 'term_db_file')),
            ('bert', ('model', 'bert')),
            ('spacy', ('model', 'spacy')),
            ('lora', ('model', 'lora')),
            ('training', ('model', 'training')),
        ]:
            if orig_attr in config_dict:
                namespace, attr = namespace_attr
                if hasattr(self, namespace):
                    namespace_obj = getattr(self, namespace)
                    if isinstance(config_dict[orig_attr], dict) and isinstance(getattr(namespace_obj, attr, None), dict):
                        getattr(namespace_obj, attr).update(config_dict[orig_attr])
                    else:
                        setattr(namespace_obj, attr, config_dict[orig_attr])
        
        # 重新创建必要的目录
        self._create_directories()
