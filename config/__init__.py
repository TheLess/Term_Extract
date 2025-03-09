"""配置模块

提供项目的配置管理功能
"""

from pathlib import Path

# 导入单一配置类
from .config import Config

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 导出Config类，使其可以被直接导入
__all__ = ['Config']
