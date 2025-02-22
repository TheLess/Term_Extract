# 游戏本地化术语提取工具

这是一个基于机器学习的游戏本地化术语提取和管理工具。它可以从多语言翻译文件中自动提取和管理术语，帮助保持翻译的一致性。

## 功能特点

- 支持多语言（中文、英文、日文、韩文）
- 基于BERT的术语提取
- 使用LoRA进行模型微调
- 术语相似度聚类
- 完整的日志记录
- 灵活的配置系统
- 命令行界面

## 安装

1. 克隆仓库：
```bash
git clone <repository_url>
cd TranslateCheck
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

1. 生成示例数据（可选）：
```bash
python scripts/generate_sample_data.py
```

2. 运行术语提取：
```bash
python main.py --input data/sample/sample_translations.xlsx --output outputs/terms/extracted_terms.json
```

3. 训练模型（可选）：
```bash
python main.py --train --input data/sample/sample_translations.xlsx
```

## 配置

配置文件位于 `config` 目录：

- `data_config.py`: 数据处理相关配置
- `model_config.py`: 模型训练相关配置

主要配置项：

```python
# 数据配置
data:
  min_term_length: 2        # 最小术语长度
  max_term_length: 20       # 最大术语长度
  min_frequency: 2          # 最小出现频率
  clean_punctuation: true   # 是否清理标点符号

# 模型配置
model:
  base_model_name: "bert-base-chinese"  # 在线模型名称
  local_model_path: "path/to/model"     # 本地模型路径（可选）
  use_local_model: false                # 是否使用本地模型
  train_batch_size: 32                  # 训练批次大小
  num_train_epochs: 3                   # 训练轮数
  learning_rate: 2e-5                   # 学习率
  warmup_ratio: 0.1                     # 预热比例
```

## 项目结构

```
project/
├── config/           # 配置文件
├── data/            # 数据文件
│   └── sample/      # 示例数据
├── src/             # 源代码
│   ├── data_processing/  # 数据处理模块
│   └── model/           # 模型训练模块
├── outputs/         # 输出文件
│   ├── models/     # 保存的模型
│   ├── terms/      # 术语库
│   └── logs/       # 日志文件
├── scripts/         # 实用脚本
├── tests/           # 测试文件
└── main.py         # 主程序
```

## 使用示例

### 基本使用

```bash
# 使用在线模型
python main.py --input path/to/translations.xlsx --output path/to/terms.json

# 使用本地模型
python main.py --input path/to/translations.xlsx --use-local-model --local-model-path path/to/local/model

# 使用现有术语库进行提取
python main.py --input path/to/translations.xlsx --term-db path/to/existing_terms.json
```

### 高级选项

```bash
# 指定配置文件
python main.py --config path/to/custom_config.py

# 设置日志级别
python main.py --log-level DEBUG

# 跳过模型训练
python main.py --no-train

# 指定输出目录
python main.py --output-dir path/to/output
```

## 开发

### 运行测试

```bash
python -m unittest discover tests
```

### 生成示例数据

```bash
python scripts/generate_sample_data.py
```

## 注意事项

1. 模型加载支持两种方式：
   - 在线模式：从Hugging Face下载BERT模型（需要网络连接）
   - 本地模式：从指定路径加载已下载的模型（无需网络连接）
2. GPU加速需要安装CUDA支持的PyTorch版本
3. 大型数据集处理可能需要较长时间

## 许可证

本项目采用 MIT 许可证
