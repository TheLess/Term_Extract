# 游戏本地化术语提取工具

这是一个基于机器学习的游戏本地化术语提取和管理工具。它可以从多语言翻译文件中自动提取和管理术语，帮助保持翻译的一致性。

## 功能特点

- 支持多语言（中文、英文 、日语、韩语）
  - 注：目前只支持中文，其他更多语音在开发中
- 基于BERT的术语提取
  - 目前的模型基于chinese-macbert-large
- 使用LoRA进行模型微调和高效训练
- 术语相似度聚类和智能分组
- 高级文本清理和预处理
  - 智能标点符号处理
  - 数字和特殊字符过滤
  - 自定义清理规则配置
- 完善的测试覆盖
  - 数据处理单元测试
  - 模型训练验证
  - 端到端集成测试

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
  remove_numbers: true      # 是否移除数字
  custom_filters: []        # 自定义过滤规则

# 模型配置
model:
  base_model_name: "bert-base-chinese"  # 在线模型名称
  local_model_path: "path/to/model"     # 本地模型路径（可选）
  use_local_model: false                # 是否使用本地模型
  train_batch_size: 32                  # 训练批次大小
  num_train_epochs: 3                   # 训练轮数
  learning_rate: 2e-5                   # 学习率
  warmup_ratio: 0.1                     # 预热比例
  lora:                                 # LoRA配置
    r: 8                               # LoRA秩
    alpha: 16                          # LoRA alpha
    dropout: 0.1                       # LoRA dropout
```

## 项目结构

```
project/
├── config/           # 配置文件
├── data/            # 数据文件
│   └── sample/      # 示例数据
├── src/             # 源代码
│   ├── data_processing/  # 数据处理模块
│   │   ├── data_processor.py    # 数据处理核心
│   │   └── term_extractor.py    # 术语提取器
│   └── model/           # 模型训练模块
│       ├── trainer.py          # 模型训练器
│       └── embeddings.py       # 词向量处理
├── outputs/         # 输出文件
│   ├── models/     # 保存的模型
│   ├── terms/      # 术语库
│   └── logs/       # 日志文件
├── scripts/         # 实用脚本
├── tests/           # 测试文件
│   ├── test_data_processor.py  # 数据处理测试
│   └── test_model_trainer.py   # 模型训练测试
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

### 高级功能

1. 使用自定义清理规则：

```bash
python main.py --input data/translations.xlsx --config custom_clean_config.py
```

1. 使用LoRA训练：

```bash
python main.py --train --use-lora --lora-r 8 --lora-alpha 16
```

1. 运行测试：

```bash
python -m pytest tests/
```

### 注意事项

1. 数据准备：
   - 输入文件需要包含源语言和目标语言列
   - 支持Excel (.xlsx) 和CSV格式
   - 建议先使用小数据集测试配置

2. 模型训练：
   - 使用LoRA可以显著减少显存占用
   - 建议先在小数据集上验证训练参数
   - 可以通过调整LoRA参数优化训练效果

3. 环境配置注意事项：
   - 稳定环境：
     - 3.13>python>=3.12 
     - torch>=2.6.0+cu126
       - cuda需要与torch版本匹配，详见torch官网，记得cuda装完配置系统环境变量
     - spacy>=3.8.4
   - 编译环境相关：
     - 最好装上以下的依赖库。不然在install各种库时，可能会报很多错：
       - ninja 
       - cmake
       - cython
     - 其他常见问题（Windows环境）：
       - gensim / spacy ：
         - Windows系统报错时，大概率缺少fortran编译环境，推荐安装MSYS mingw64（非python环境）
         - 配置好MSYS mingw64系统环境后，通过`pacman -S mingw-w64-x86_64-openmp`等命令安装数学计算相关环境
         - 然后记得把MSYS mingw64的bin | include “目录加入到系统环境变量中
     - 虚拟环境问题：
       - conda环境太老时（python版本过低），直接更新conda会失败，卸载conda重新安装
       - 重新安装conda过程中，如果碰到报错，可能是注册表没清理干净，检查注册表，删除这里：`HKEY_CURRENT_USER\Software\Microsoft\Command Processor`里的`autorun`
       - 同样如果虚拟环境的python环境升级或降级，会碰到找不到pip相关路径的问题，也和注册表有关，可以尝试删除python相关注册表信息，然后清空环境重新安装

## 开发

### 运行测试

```bash
python -m unittest discover tests
```

### 生成示例数据

```bash
python scripts/generate_sample_data.py
```

## 许可证

本项目采用 MIT 许可证
