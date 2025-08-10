"""
基于EEG的吸烟成瘾评估系统

此包提供了完整的EEG数据处理、特征提取、机器学习和结果分析功能。
主要包含以下模块：
- data: 数据加载、预处理和验证
- features: 特征提取（频谱、ERP、微状态、功能连接）
- models: 机器学习分类器和评估
- utils: 工具函数和配置管理

作者: EEG Research Team
版本: 0.1.0
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "EEG Research Team"
__email__ = "research@example.com"

# 导入主要模块，方便用户直接使用
from . import data
from . import features
from . import models
from . import utils

# 定义包的公共API
__all__ = [
    "data",
    "features", 
    "models",
    "utils",
    "__version__",
    "__author__",
    "__email__"
]

