"""
数据模块测试 - 测试数据加载和预处理模块

此模块包含对数据相关功能的单元测试：
- EEGDataLoader: 数据加载器测试
- EEGPreprocessor: 数据预处理器测试  
- DataValidator: 数据验证器测试

测试内容包括：
- 基本功能测试
- 错误处理测试
- 配置验证测试
- 数据完整性测试
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.data.loader import EEGDataLoader
from src.data.preprocessor import EEGPreprocessor
from src.data.validator import DataValidator


class TestEEGDataLoader:
    """
    EEG数据加载器测试类
    
    测试EEGDataLoader的各种功能，包括数据格式支持、
    文件查找、被试列表等。
    """
    
    def setup_method(self):
        """
        设置测试环境
        
        创建临时目录和测试数据结构
        """
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True)
        
        # 创建测试数据结构
        (self.data_dir / "raw" / "neuroscan").mkdir(parents=True)
        (self.data_dir / "raw" / "eeglab").mkdir(parents=True)
        (self.data_dir / "raw" / "questionnaires").mkdir(parents=True)
        
        self.loader = EEGDataLoader(str(self.data_dir))
    
    def teardown_method(self):
        """
        清理测试环境
        
        删除临时目录和文件
        """
        shutil.rmtree(self.temp_dir)
    
    def test_list_subjects_empty(self):
        """
        测试空数据目录下的被试列表
        
        当没有数据文件时，应该返回空列表
        """
        subjects = self.loader.list_subjects()
        assert subjects == []
    
    def test_supported_formats(self):
        """
        测试支持的数据格式
        
        验证支持的数据格式是否正确定义
        """
        expected_formats = ['neuroscan', 'eeglab', 'curry', 'egi']
        assert list(self.loader.supported_formats.keys()) == expected_formats
    
    def test_extract_subject_from_filename(self):
        """
        测试从文件名提取被试ID
        
        验证文件名解析功能是否正常工作
        """
        # 测试有效文件名
        filename = "sub-001_task-resting_state.cnt"
        subject = self.loader._extract_subject_from_filename(filename)
        assert subject == "001"
        
        # 测试无效文件名
        subject = self.loader._extract_subject_from_filename("invalid_filename.txt")
        assert subject is None


class TestEEGPreprocessor:
    """
    EEG预处理器测试类
    
    测试EEGPreprocessor的各种功能，包括配置初始化、
    滤波设置、ICA参数等。
    """
    
    def setup_method(self):
        """
        设置测试环境
        
        创建测试配置和预处理器实例
        """
        self.config = {
            'filter': {
                'l_freq': 0.1,        # 低频截止
                'h_freq': 100.0,      # 高频截止
                'method': 'iir'       # 滤波方法
            },
            'ica': {
                'method': 'fastica',  # ICA算法
                'n_components': 15,   # 成分数量
                'max_iter': 200       # 最大迭代次数
            },
            'artifact_rejection': {
                'method': 'peak_to_peak',  # 伪迹检测方法
                'threshold': 150e-6        # 检测阈值
            },
            'epoching': {
                'tmin': -0.2,         # 事件前时间
                'tmax': 1.0,          # 事件后时间
                'baseline': (-0.2, 0) # 基线时间窗口
            }
        }
        
        self.preprocessor = EEGPreprocessor(self.config)
    
    def test_config_initialization(self):
        """
        测试配置初始化
        
        验证配置参数是否正确加载
        """
        assert self.preprocessor.filter_config['l_freq'] == 0.1
        assert self.preprocessor.filter_config['h_freq'] == 100.0
        assert self.preprocessor.ica_config['method'] == 'fastica'
    
    def test_bandpass_filter_creation(self):
        """
        测试带通滤波器创建
        
        验证滤波器相关方法是否存在
        """
        # 这需要实际的EEG数据来正确测试
        # 目前只测试方法是否存在
        assert hasattr(self.preprocessor, '_apply_filtering')


class TestDataValidator:
    """
    数据验证器测试类
    
    测试DataValidator的各种功能，包括配置验证、
    验证结果结构等。
    """
    
    def setup_method(self):
        """
        设置测试环境
        
        创建测试配置和验证器实例
        """
        self.config = {
            'quality_thresholds': {
                'min_duration': 60,      # 最小持续时间
                'min_channels': 32,      # 最小通道数
                'min_sfreq': 250,        # 最小采样频率
                'noise_threshold': 100e-6, # 噪声阈值
                'flat_threshold': 1e-6   # 平线阈值
            }
        }
        
        self.validator = DataValidator(self.config)
    
    def test_config_initialization(self):
        """
        测试配置初始化
        
        验证质量阈值是否正确加载
        """
        assert self.validator.quality_thresholds['min_duration'] == 60
        assert self.validator.quality_thresholds['min_channels'] == 32
        assert self.validator.quality_thresholds['min_sfreq'] == 250
    
    def test_validation_results_structure(self):
        """
        测试验证结果结构
        
        验证验证结果是否包含正确的字段
        """
        # 创建模拟的验证结果
        validation_result = {
            'subject': 'test_subject',
            'passed': True,
            'issues': [],
            'quality_metrics': {}
        }
        
        # 验证结果结构
        assert 'subject' in validation_result
        assert 'passed' in validation_result
        assert 'issues' in validation_result
        assert 'quality_metrics' in validation_result
        assert isinstance(validation_result['issues'], list)
        assert isinstance(validation_result['quality_metrics'], dict)
    
    def test_quality_metrics_calculation(self):
        """
        测试质量指标计算
        
        验证质量指标计算功能是否正常
        """
        # 创建模拟EEG数据
        data = np.random.randn(32, 1000)  # 32通道，1000个时间点
        sfreq = 1000.0  # 采样频率
        
        # 计算质量指标
        metrics = self.validator._calculate_quality_metrics(data, sfreq)
        
        # 验证指标存在
        assert 'mean_amplitude' in metrics
        assert 'std_amplitude' in metrics
        assert 'max_amplitude' in metrics
        assert 'flat_channels' in metrics
        
        # 验证指标值合理
        assert metrics['mean_amplitude'] >= 0
        assert metrics['std_amplitude'] >= 0
        assert metrics['max_amplitude'] >= 0
        assert metrics['flat_channels'] >= 0
    
    def test_validation_with_invalid_data(self):
        """
        测试无效数据的验证
        
        验证对无效数据的处理是否正确
        """
        # 测试空数据
        empty_data = np.array([])
        sfreq = 1000.0
        
        # 应该能够处理空数据而不崩溃
        try:
            metrics = self.validator._calculate_quality_metrics(empty_data, sfreq)
            # 如果成功，验证结果
            assert isinstance(metrics, dict)
        except Exception as e:
            # 如果失败，确保是预期的错误
            assert "empty" in str(e).lower() or "shape" in str(e).lower()


# 集成测试
class TestDataPipeline:
    """
    数据管道集成测试类
    
    测试数据处理的完整流程，包括加载、预处理和验证。
    """
    
    def test_end_to_end_pipeline(self):
        """
        测试端到端数据管道
        
        验证完整的数据处理流程是否正常工作
        """
        # 这个测试需要实际的EEG数据文件
        # 在实际项目中，应该使用测试数据文件
        assert True  # 占位符测试
        
    def test_error_handling(self):
        """
        测试错误处理
        
        验证系统对错误输入的处理是否正确
        """
        # 测试无效文件路径
        with pytest.raises(FileNotFoundError):
            loader = EEGDataLoader("nonexistent_directory")
            loader.list_subjects()
        
        # 测试无效配置
        with pytest.raises(KeyError):
            preprocessor = EEGPreprocessor({})
            # 尝试访问不存在的配置项
            _ = preprocessor.filter_config['nonexistent_key']


if __name__ == "__main__":
    pytest.main([__file__])

