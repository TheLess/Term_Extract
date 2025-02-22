from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.cluster import KMeans
import logging
from pathlib import Path
from typing import Dict, List, Optional
import datasets
from tqdm import tqdm
import pandas as pd

class ModelTrainer:
    def __init__(self, config):
        """初始化模型训练器
        
        Args:
            config: 配置对象，包含模型训练相关的配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化模型和分词器
        self._init_model_and_tokenizer()
        
    def _setup_logging(self):
        """配置日志"""
        log_file = self.config.logs_dir / 'model_trainer.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已经配置了日志处理器
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
        
    def _init_model_and_tokenizer(self):
        """初始化模型和分词器"""
        try:
            # 确定模型来源
            if self.config.model.use_local_model and self.config.model.local_model_path:
                model_path = self.config.model.local_model_path
                self.logger.info(f"从本地加载模型: {model_path}")
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"本地模型路径不存在: {model_path}")
            else:
                model_path = self.config.model.base_model_name
                self.logger.info(f"从Hugging Face下载模型: {model_path}")
            
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            
            # 配置LoRA
            self.lora_config = LoraConfig(
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                target_modules=self.config.model.lora_target_modules,
                lora_dropout=self.config.model.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"  # 适用于MacBERT
            )
            
            self.model = get_peft_model(model, self.lora_config)
            self.model.to(self.device)
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            raise
            
    def _preprocess_data(self, df) -> Optional[datasets.Dataset]:
        """预处理训练数据
        
        Args:
            df: 包含文本数据的DataFrame
            
        Returns:
            Optional[datasets.Dataset]: 处理后的数据集
        """
        try:
            if 'text_clean' not in df.columns:
                self.logger.error("数据集缺少 'text_clean' 列")
                return None
                
            term_dict = df['text_clean'].value_counts().to_dict()
            
            def tokenize_and_mask(examples):
                texts = examples['text_clean']
                if not isinstance(texts, list):
                    texts = [texts]
                
                all_input_ids = []
                all_labels = []
                
                for text in texts:
                    if pd.isna(text):
                        text = ""
                    tokens = self.tokenizer.tokenize(str(text))
                    masked_tokens = []
                    labels = []
                    
                    for token in tokens:
                        if token in term_dict and np.random.rand() < self.config.model.mask_probability:
                            masked_tokens.append('[MASK]')
                            labels.append(self.tokenizer.convert_tokens_to_ids(token))
                        else:
                            masked_tokens.append(token)
                            labels.append(-100)
                    
                    # 确保序列长度一致
                    input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
                    if len(input_ids) > 512:  # BERT的最大序列长度
                        input_ids = input_ids[:512]
                        labels = labels[:512]
                    
                    all_input_ids.append(input_ids)
                    all_labels.append(labels)
                
                return {
                    'input_ids': all_input_ids,
                    'labels': all_labels
                }
            
            dataset = datasets.Dataset.from_pandas(df)
            processed_dataset = dataset.map(
                tokenize_and_mask,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            return processed_dataset
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            return None
            
    def train(self, dataset) -> bool:
        """训练模型
        
        Args:
            dataset: 训练数据集
            
        Returns:
            bool: 训练是否成功
        """
        try:
            self.logger.info("开始数据预处理...")
            encoded_data = self._preprocess_data(dataset)
            if encoded_data is None:
                return False
                
            # 设置训练参数
            training_args = TrainingArguments(
                output_dir=str(self.config.model_dir),
                num_train_epochs=self.config.model.num_train_epochs,
                per_device_train_batch_size=self.config.model.train_batch_size,
                learning_rate=self.config.model.learning_rate,
                logging_dir=str(self.config.logs_dir),
                logging_steps=10,
                save_strategy="epoch",
                warmup_ratio=self.config.model.warmup_ratio,
                seed=self.config.model.random_seed
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=encoded_data
            )
            
            self.logger.info("开始训练模型...")
            trainer.train()
            
            # 保存模型
            save_path = self.config.model_dir / 'final_model'
            self.model.save_pretrained(save_path)
            self.logger.info(f"模型已保存到: {save_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
            
    def _get_term_embeddings(self, terms: List[str]) -> Optional[np.ndarray]:
        """获取术语的向量表示
        
        Args:
            terms: 术语列表
            
        Returns:
            Optional[np.ndarray]: 术语向量矩阵，shape为(len(terms), hidden_size)
        """
        try:
            # 将术语转换为模型输入
            inputs = self.tokenizer(terms, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取BERT的输出
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
            # 使用最后一层隐藏状态的第一个token ([CLS]) 作为句子表示
            last_hidden_state = outputs.hidden_states[-1]
            embeddings = last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"获取术语向量失败: {str(e)}")
            return None
            
    def _cluster_terms(self, embeddings: np.ndarray, n_clusters: int = None) -> Optional[np.ndarray]:
        """对术语进行聚类
        
        Args:
            embeddings: 术语向量矩阵
            n_clusters: 聚类数量，如果为None则自动确定
            
        Returns:
            Optional[np.ndarray]: 聚类标签
        """
        try:
            if n_clusters is None:
                n_clusters = max(2, min(len(embeddings) // 2, 5))
                
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"术语聚类失败: {str(e)}")
            return None
            
    def refine_terms(self, term_dict: Dict[str, int]) -> Dict[str, int]:
        """优化术语列表，合并相似术语
        
        Args:
            term_dict: 术语及其频率字典
            
        Returns:
            Dict[str, int]: 优化后的术语字典
        """
        try:
            terms = list(term_dict.keys())
            if len(terms) < 2:
                return term_dict.copy()
                
            # 获取术语向量
            embeddings = self._get_term_embeddings(terms)
            if embeddings is None:
                return term_dict.copy()
                
            # 聚类
            clusters = self._cluster_terms(embeddings)
            if clusters is None:
                return term_dict.copy()
                
            # 合并相似术语
            refined_terms = {}
            cluster_terms = {}
            
            # 按聚类分组术语
            for term, cluster in zip(terms, clusters):
                if cluster not in cluster_terms:
                    cluster_terms[cluster] = []
                cluster_terms[cluster].append(term)
                
            # 在每个聚类中选择频率最高的术语作为代表
            for cluster, cluster_term_list in cluster_terms.items():
                if len(cluster_term_list) == 1:
                    term = cluster_term_list[0]
                    refined_terms[term] = term_dict[term]
                else:
                    # 选择频率最高的术语
                    representative_term = max(cluster_term_list, key=lambda x: term_dict[x])
                    # 合并频率
                    total_freq = sum(term_dict[term] for term in cluster_term_list)
                    refined_terms[representative_term] = total_freq
                    
            return refined_terms
            
        except Exception as e:
            self.logger.error(f"术语优化失败: {str(e)}")
            return term_dict.copy()
