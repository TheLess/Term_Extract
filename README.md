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

- `data_config.py`: 数据处理相关配置（文件路径、数据清理参数等）
- `model_config.py`: 模型训练相关配置（模型参数、LoRA配置等）
- `__init__.py`: 配置整合和目录结构初始化

主要配置项：

```python
# 数据配置 (data_config.py)
class DataConfig:
    def __init__(self):
        self.data_dir = Path('data')
        self.translations_file = self.data_dir / 'translations.xlsx'
        self.term_db_file = self.data_dir / 'term_database.json'
        
        # 数据处理参数
        self.min_term_length = 2    # 最小术语长度
        self.min_term_freq = 2      # 最小术语频率
        self.max_term_length = 10   # 最大术语长度
        
        # 数据清理参数
        self.remove_punctuation = True   # 是否移除标点
        self.remove_numbers = False      # 是否移除数字
        self.lowercase = False           # 是否转小写（针对英文）

# 模型配置 (model_config.py)
class ModelConfig:
    def __init__(self):
        self.bert = {
            'name': 'hfl/chinese-macbert-large',  # 在线模型名称
            'local_path': Path('models/chinese-macbert-large'),  # 本地模型路径
            'use_local': True,  # 是否使用本地模型
        }
        
        # LoRA配置
        self.lora = {
            'r': 8,              # LoRA秩
            'alpha': 32,         # LoRA alpha参数
            'dropout': 0.1,      # Dropout率
            'bias': 'none',      # 是否使用偏置项
            'target_modules': ['query', 'value']  # 目标模块
        }
```

## 项目结构

```
project/
├── config/                 # 配置文件
│   ├── __init__.py        # 配置整合
│   ├── data_config.py     # 数据处理配置
│   └── model_config.py    # 模型配置
├── data/                  # 数据文件
│   └── sample/           # 示例数据
├── src/                   # 源代码
│   ├── data_processing/  # 数据处理模块
│   │   ├── __init__.py
│   │   └── data_processor.py  # 数据处理核心
│   └── model/            # 模型训练模块
│       ├── __init__.py
│       └── trainer.py    # 模型训练器
├── outputs/              # 输出文件
│   ├── models/          # 保存的模型
│   ├── terms/           # 术语库
│   └── logs/            # 日志文件
├── scripts/              # 实用脚本
│   └── generate_sample_data.py  # 示例数据生成
├── tests/               # 测试文件
│   ├── __init__.py
│   ├── conftest.py     # pytest配置
│   ├── test_data_processor.py  # 数据处理测试
│   └── test_model_trainer.py   # 模型训练测试
├── requirements.txt     # 项目依赖
└── main.py             # 主程序
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
# 运行所有测试
python -m pytest

# 运行特定测试文件
python -m pytest tests/test_data_processor.py

# 运行特定测试类或方法
python -m pytest tests/test_data_processor.py::TestDataProcessor::test_term_extraction_basic
```

### 生成示例数据

```bash
python scripts/generate_sample_data.py
```

## 许可证

本项目采用 MIT 许可证
