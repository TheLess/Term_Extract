#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, Dict

from config import Config
from src.data_processing import DataProcessor
from src.model import ModelTrainer

def setup_logging(config: Config) -> None:
    """设置日志配置"""
    log_file = config.logs_dir / 'main.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='游戏本地化术语提取工具')
    
    # 输入输出参数
    parser.add_argument('--input', type=str, help='输入的翻译文件路径')
    parser.add_argument('--output', type=str, help='输出的术语库文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, help='自定义配置文件路径')
    
    # 模型相关参数
    parser.add_argument('--use-local-model', action='store_true', help='使用本地模型')
    parser.add_argument('--local-model-path', type=str, help='本地模型路径')
    parser.add_argument('--train', action='store_true', help='是否训练模型')
    parser.add_argument('--no-train', action='store_true', help='跳过模型训练')
    
    # 其他参数
    parser.add_argument('--term-db', type=str, help='现有术语库路径')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    parser.add_argument('--min-freq', type=int, help='最小术语频率')
    parser.add_argument('--min-length', type=int, help='最小术语长度')
    parser.add_argument('--max-length', type=int, help='最大术语长度')
    
    return parser.parse_args()

class TermExtractor:
    def __init__(self, config: Config):
        """初始化术语提取器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_processor = DataProcessor(config)
        self.trainer = ModelTrainer(config)
        
    def update_config_from_args(self, args: argparse.Namespace) -> None:
        """根据命令行参数更新配置
        
        Args:
            args: 命令行参数
        """
        # 更新输入输出路径
        if args.input:
            self.config.translations_file = Path(args.input)
            
        if args.output:
            self.config.term_db_file = Path(args.output)
            # 确保输出目录存在
            self.config.term_db_file.parent.mkdir(parents=True, exist_ok=True)
            
        if args.output_dir:
            output_dir = Path(args.output_dir)
            self.config.output_dir = output_dir
            self.config.model_dir = output_dir / 'models'
            self.config.terms_dir = output_dir / 'terms'
            self.config.logs_dir = output_dir / 'logs'
            
        # 更新模型配置
        if args.use_local_model:
            self.config.bert['use_local'] = True
            if args.local_model_path:
                model_path = Path(args.local_model_path)
                self.config.bert['local_path'] = model_path
                
        # 更新术语提取参数
        if args.min_freq:
            self.config.min_term_freq = args.min_freq
            
        if args.min_length:
            self.config.min_term_length = args.min_length
            
        if args.max_length:
            self.config.max_term_length = args.max_length
            
        # 更新训练配置
        if args.train:
            self.config.train_model = True
        
        if args.no_train:
            self.config.train_model = False
            
        # 更新日志级别
        if args.log_level:
            self.config.log_level = args.log_level
            
    def run(self, skip_training: bool = False) -> Optional[Dict[str, int]]:
        """运行术语提取流程
        
        Args:
            skip_training: 是否跳过模型训练
            
        Returns:
            Optional[Dict[str, int]]: 提取的术语字典，失败时返回None
        """
        try:
            # 加载数据
            self.logger.info("开始加载数据...")
            df = self.data_processor.load_data()
            if df is None:
                return None
                
            # 初始术语提取
            self.logger.info("开始提取初始术语...")
            initial_terms = self.data_processor.build_term_corpus(df['text_clean'].tolist())
            
            if not skip_training and self.config.train_model:
                # 模型训练和优化
                self.logger.info("开始模型训练...")
                self.trainer.train(df)
                
                # 迭代优化术语
                self.logger.info("开始优化术语...")
                optimization_rounds = self.config.model.term_refinement['optimization_rounds']
                for epoch in range(optimization_rounds):
                    self.logger.info(f"第 {epoch + 1} 轮优化...")
                    refined_terms = self.trainer.refine_terms(initial_terms)
                    initial_terms = refined_terms
            
            # 保存结果
            self.logger.info(f"保存术语库到 {self.config.term_db_file}")
            self.data_processor.save_term_db(initial_terms)
            
            return initial_terms
            
        except Exception as e:
            self.logger.error(f"处理过程中出错: {str(e)}", exc_info=True)
            return None

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 加载配置
        config_path = None
        if args.config:
            config_path = args.config
        elif Path('config/default_settings.json').exists():
            config_path = 'config/default_settings.json'
            
        config = Config(config_path)
        
        # 设置日志
        log_level = getattr(logging, config.log_level)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.logs_dir / 'main.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # 创建术语提取器
        extractor = TermExtractor(config)
        extractor.update_config_from_args(args)
        
        # 检查是否提供了输入文件
        if not config.translations_file.exists():
            if not args.input:
                logger.error("未提供输入文件，请使用 --input 参数指定翻译文件")
                return 1
            else:
                logger.error(f"指定的输入文件不存在: {config.translations_file}")
                return 1
        
        # 运行提取流程
        skip_training = args.no_train or not config.train_model
        terms = extractor.run(skip_training=skip_training)
        
        if terms is not None:
            logger.info(f"成功提取 {len(terms)} 个术语")
            return 0
        else:
            logger.error("术语提取失败")
            return 1
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
