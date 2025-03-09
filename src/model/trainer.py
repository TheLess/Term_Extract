from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
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
            # 初始化分词器
            if self.config.model.bert['use_local']:
                model_path = self.config.model.bert['local_path']
                # 确保model_path是Path对象
                if isinstance(model_path, str):
                    model_path = Path(model_path)
                if not model_path.exists():
                    self.logger.warning(f"本地模型路径不存在: {model_path}")
                    raise OSError(f"找不到本地模型: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.model = AutoModelForMaskedLM.from_pretrained(str(model_path))
                self.logger.info(f"从本地加载模型: {model_path}")
            else:
                model_name = self.config.model.bert['name']
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForMaskedLM.from_pretrained(model_name)
                self.logger.info(f"在线加载模型: {model_name}")

            # 配置LoRA
            peft_config = LoraConfig(
                r=self.config.model.lora['r'],
                lora_alpha=self.config.model.lora['alpha'],
                target_modules=self.config.model.lora['target_modules'],
                lora_dropout=self.config.model.lora['dropout'],
                bias=self.config.model.lora['bias']
            )
            
            # 应用LoRA配置
            self.model = get_peft_model(self.model, peft_config)
            
            # 移动模型到指定设备
            self.model.to(self.device)
            self.logger.info(f"使用设备: {self.device}")
            
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
            max_length = 256  # 减小序列长度以节省内存
            
            def tokenize_and_mask(examples):
                texts = examples['text_clean']
                if not isinstance(texts, list):
                    texts = [texts]
                
                # 使用tokenizer的batch处理功能
                tokenized = self.tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                input_ids = tokenized['input_ids'].tolist()
                attention_mask = tokenized['attention_mask'].tolist()
                labels = [[token_id if (token_id != self.tokenizer.pad_token_id and np.random.rand() < 0.15) else -100 
                          for token_id in sequence] for sequence in input_ids]
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
            
            dataset = datasets.Dataset.from_pandas(df)
            processed_dataset = dataset.map(
                tokenize_and_mask,
                batched=True,
                batch_size=32,  # 添加batch_size限制
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
                output_dir=str(self.config.output_dir / "checkpoints"),
                num_train_epochs=self.config.model.training['num_epochs'],
                per_device_train_batch_size=4,  # 减小批量大小
                gradient_accumulation_steps=4,  # 添加梯度累积
                learning_rate=self.config.model.training['learning_rate'],
                warmup_steps=self.config.model.training['warmup_steps'],
                max_grad_norm=self.config.model.training['max_grad_norm'],
                weight_decay=self.config.model.training['weight_decay'],
                logging_dir=str(self.config.output_dir / "logs"),
                logging_steps=10,
                save_strategy="no",
                remove_unused_columns=False,
                fp16=True  # 启用混合精度训练
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=encoded_data,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=True,
                    mlm_probability=0.15
                )
            )
            
            self.logger.info("开始训练模型...")
            trainer.train()
            
            # 保存模型
            save_path = self.config.output_dir / 'models' / 'final_model'
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(save_path))
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
            # 对术语进行编码
            encoded = self.tokenizer(
                terms,
                padding=True,
                truncation=True,
                max_length=256,  # 使用与训练时相同的最大长度
                return_tensors='pt'
            )
            
            # 将数据移动到正确的设备
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
            # 使用最后一层的[CLS]向量作为术语表示
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
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
                # 更合理的聚类数量计算，确保大型术语集不会被过度聚类
                n_samples = len(embeddings)
                # 使用术语数量的平方根作为基准，但有最小和最大限制
                n_clusters = min(max(int(np.sqrt(n_samples)), 5), n_samples // 2)
                # 确保至少有2个聚类
                n_clusters = max(2, n_clusters)
                
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
                
            self.logger.info(f"优化前术语数量: {len(terms)}")
                
            # 获取术语向量
            embeddings = self._get_term_embeddings(terms)
            if embeddings is None:
                return term_dict.copy()
                
            # 聚类
            clusters = self._cluster_terms(embeddings)
            if clusters is None:
                return term_dict.copy()
                
            # 记录聚类数量
            unique_clusters = len(set(clusters))
            self.logger.info(f"聚类数量: {unique_clusters}")
                
            # 合并相似术语
            refined_terms = {}
            cluster_terms = {}
            
            # 按聚类分组术语
            for term, cluster in zip(terms, clusters):
                if cluster not in cluster_terms:
                    cluster_terms[cluster] = []
                cluster_terms[cluster].append(term)
            
            # 打印每个聚类中的术语数量    
            cluster_sizes = {cluster: len(terms) for cluster, terms in cluster_terms.items()}
            self.logger.info(f"聚类大小分布: {cluster_sizes}")
                
            # 改进的术语选择策略：
            # 对于小聚类（<=5个术语）保留所有术语
            # 对于中等聚类（6-20个术语）保留频率最高的前3个
            # 对于大聚类（>20个术语）保留频率最高的前5个
            for cluster, cluster_term_list in cluster_terms.items():
                if len(cluster_term_list) <= 5:
                    # 保留小聚类中的所有术语
                    for term in cluster_term_list:
                        refined_terms[term] = term_dict[term]
                else:
                    # 按频率排序术语
                    sorted_terms = sorted(cluster_term_list, key=lambda x: term_dict[x], reverse=True)
                    
                    # 确定要保留的术语数量
                    if len(cluster_term_list) <= 20:
                        keep_count = min(3, len(sorted_terms))
                    else:
                        keep_count = min(5, len(sorted_terms))
                        
                    # 保留高频术语
                    for term in sorted_terms[:keep_count]:
                        refined_terms[term] = term_dict[term]
            
            self.logger.info(f"优化后术语数量: {len(refined_terms)}")
            return refined_terms
            
        except Exception as e:
            self.logger.error(f"术语优化失败: {str(e)}")
            return term_dict.copy()
