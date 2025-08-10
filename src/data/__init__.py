"""
数据模块 - 负责EEG数据的加载、预处理和验证

此模块提供了完整的数据处理流程：
- EEGDataLoader: 支持多种格式的EEG数据加载
- EEGPreprocessor: EEG数据预处理（滤波、ICA、伪迹去除等）
- DataValidator: 数据质量验证和完整性检查

支持的数据格式：
- Neuroscan (.cnt)
- EEGLAB (.set/.fdt)  
- Curry8 (.cdt/.dpa)
- EGI (.raw)
"""

# 导入主要类，方便用户直接使用
from .loader import EEGDataLoader
from .preprocessor import EEGPreprocessor
from .validator import DataValidator

# 定义模块的公共API
__all__ = [
    "EEGDataLoader",
    "EEGPreprocessor", 
    "DataValidator"
]

