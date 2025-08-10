"""
特征提取模块 - 负责从EEG数据中提取各种特征

此模块提供了多种EEG特征提取方法：
- SpectralFeatureExtractor: 频谱特征（功率谱、频带功率等）
- ERPFeatureExtractor: 事件相关电位特征（N1、P2、LPP等）
- MicrostateFeatureExtractor: 微状态特征（持续时间、覆盖率等）
- ConnectivityFeatureExtractor: 功能连接特征（PLV、相关性等）
- FeatureFusion: 特征融合和集成

支持的特征类型：
- 时域特征：ERP成分、振幅、潜伏期
- 频域特征：功率谱、频带功率、频谱熵
- 时频特征：小波变换、时频分析
- 空间特征：微状态、地形图
- 连接特征：功能连接、网络拓扑
"""

# 导入各个特征提取器
from .spectral import SpectralFeatureExtractor
from .erp import ERPFeatureExtractor
from .microstates import MicrostateFeatureExtractor
from .connectivity import ConnectivityFeatureExtractor
from .fusion import FeatureFusion

# 定义模块的公共API
__all__ = [
    "SpectralFeatureExtractor",
    "ERPFeatureExtractor",
    "MicrostateFeatureExtractor", 
    "ConnectivityFeatureExtractor",
    "FeatureFusion"
]

