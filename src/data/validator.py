"""
数据验证工具 - 验证EEG数据质量和完整性

此模块提供了全面的数据质量检查功能：
- 原始数据验证：持续时间、通道数、采样率、信号质量
- 分段数据验证：试次数量、试次质量、基线稳定性
- 特征数据验证：特征完整性、数值范围、缺失值
- 质量报告生成：汇总验证结果和问题统计

验证内容包括：
- 数据完整性检查
- 信号质量评估
- 伪迹检测
- 统计指标计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

import mne

logger = logging.getLogger(__name__)


class DataValidator:
    """
    数据验证器 - 验证EEG数据质量和完整性
    
    此类提供了全面的数据质量检查功能，确保输入数据
    符合分析要求，包括数据完整性、信号质量和统计指标。
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据验证器
        
        Args:
            config: 包含验证参数的配置字典
        """
        self.config = config
        self.quality_thresholds = config.get('quality_thresholds', {})
        
    def validate_raw_data(self, raw: mne.io.Raw, subject: str) -> Dict:
        """
        验证原始EEG数据质量
        
        Args:
            raw: MNE Raw对象
            subject: 被试ID
            
        Returns:
            包含验证结果的字典
        """
        validation_results = {
            'subject': subject,
            'passed': True,           # 是否通过验证
            'issues': [],             # 发现的问题列表
            'quality_metrics': {}     # 质量指标
        }
        
        # 获取基本数据属性
        data = raw.get_data()
        sfreq = raw.info['sfreq']        # 采样频率
        n_channels = len(raw.ch_names)   # 通道数
        duration = raw.times[-1]         # 数据持续时间
        
        # 持续时间检查
        min_duration = self.quality_thresholds.get('min_duration', 60)  # 秒
        if duration < min_duration:
            validation_results['passed'] = False
            validation_results['issues'].append(f"数据持续时间过短: {duration:.1f}s < {min_duration}s")
        
        # 通道数量检查
        min_channels = self.quality_thresholds.get('min_channels', 32)
        if n_channels < min_channels:
            validation_results['passed'] = False
            validation_results['issues'].append(f"通道数量不足: {n_channels} < {min_channels}")
        
        # 采样频率检查
        min_sfreq = self.quality_thresholds.get('min_sfreq', 250)
        if sfreq < min_sfreq:
            validation_results['passed'] = False
            validation_results['issues'].append(f"采样频率过低: {sfreq} < {min_sfreq}")
        
        # 信号质量检查
        quality_metrics = self._calculate_quality_metrics(data, sfreq)
        validation_results['quality_metrics'] = quality_metrics
        
        # 检查过度噪声
        noise_threshold = self.quality_thresholds.get('noise_threshold', 100e-6)
        if quality_metrics['mean_amplitude'] > noise_threshold:
            validation_results['passed'] = False
            validation_results['issues'].append(f"噪声过大: {quality_metrics['mean_amplitude']:.2e} > {noise_threshold:.2e}")
        
        # 检查平线通道
        flat_threshold = self.quality_thresholds.get('flat_threshold', 1e-6)
        flat_channels = quality_metrics['flat_channels']
        if flat_channels > 0:
            validation_results['issues'].append(f"发现 {flat_channels} 个平线通道")
        
        return validation_results
    
    def _calculate_quality_metrics(self, data: np.ndarray, sfreq: float) -> Dict:
        """
        计算EEG数据的质量指标
        
        Args:
            data: EEG数据数组
            sfreq: 采样频率
            
        Returns:
            包含质量指标的字典
        """
        metrics = {}
        
        # 振幅统计
        metrics['mean_amplitude'] = np.mean(np.abs(data))
        metrics['std_amplitude'] = np.std(data)
        metrics['max_amplitude'] = np.max(np.abs(data))
        
        # 信噪比估计
        signal_power = np.mean(data ** 2)
        noise_power = np.var(data - np.mean(data, axis=1, keepdims=True))
        metrics['snr_estimate'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        # 平线通道检测
        flat_threshold = self.quality_thresholds.get('flat_threshold', 1e-6)
        flat_channels = np.sum(np.std(data, axis=1) < flat_threshold)
        metrics['flat_channels'] = flat_channels
        
        # 异常值检测
        z_scores = np.abs((data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True))
        outliers = np.sum(z_scores > 3)
        metrics['outliers'] = outliers
        metrics['outlier_ratio'] = outliers / data.size
        
        # 频谱质量
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(256, data.shape[1]//4))
        metrics['dominant_frequency'] = freqs[np.argmax(np.mean(psd, axis=0))]
        metrics['spectral_entropy'] = -np.sum(psd * np.log(psd + 1e-10)) / np.sum(psd)
        
        return metrics
    
    def validate_epochs(self, epochs: mne.Epochs, subject: str) -> Dict:
        """
        验证分段数据质量
        
        Args:
            epochs: MNE Epochs对象
            subject: 被试ID
            
        Returns:
            包含验证结果的字典
        """
        validation_results = {
            'subject': subject,
            'passed': True,
            'issues': [],
            'quality_metrics': {}
        }
        
        # 获取分段信息
        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        epoch_duration = epochs.tmax - epochs.tmin
        
        # 试次数量检查
        min_epochs = self.quality_thresholds.get('min_epochs', 10)
        if n_epochs < min_epochs:
            validation_results['passed'] = False
            validation_results['issues'].append(f"试次数量不足: {n_epochs} < {min_epochs}")
        
        # 试次持续时间检查
        min_epoch_duration = self.quality_thresholds.get('min_epoch_duration', 0.5)
        if epoch_duration < min_epoch_duration:
            validation_results['passed'] = False
            validation_results['issues'].append(f"试次持续时间过短: {epoch_duration:.2f}s < {min_epoch_duration}s")
        
        # 计算分段质量指标
        quality_metrics = self._calculate_epoch_quality_metrics(epochs)
        validation_results['quality_metrics'] = quality_metrics
        
        # 检查试次间变异性
        max_cv = self.quality_thresholds.get('max_cv', 0.5)
        if quality_metrics['coefficient_of_variation'] > max_cv:
            validation_results['issues'].append(f"试次间变异性过大: {quality_metrics['coefficient_of_variation']:.3f} > {max_cv}")
        
        return validation_results
    
    def _calculate_epoch_quality_metrics(self, epochs: mne.Epochs) -> Dict:
        """
        计算分段数据的质量指标
        
        Args:
            epochs: MNE Epochs对象
            
        Returns:
            包含质量指标的字典
        """
        metrics = {}
        
        # 获取分段数据
        data = epochs.get_data()
        
        # 试次统计
        metrics['n_epochs'] = len(epochs)
        metrics['n_channels'] = len(epochs.ch_names)
        metrics['epoch_duration'] = epochs.tmax - epochs.tmin
        
        # 振幅统计
        metrics['mean_amplitude'] = np.mean(np.abs(data))
        metrics['std_amplitude'] = np.std(data)
        metrics['max_amplitude'] = np.max(np.abs(data))
        
        # 试次间一致性
        mean_epoch = np.mean(data, axis=0)
        epoch_variances = np.var(data, axis=0)
        metrics['inter_trial_consistency'] = 1 - np.mean(epoch_variances) / np.var(mean_epoch)
        
        # 变异系数
        metrics['coefficient_of_variation'] = np.std(data) / np.mean(np.abs(data))
        
        # 基线稳定性
        baseline_data = epochs.get_data()[:, :, epochs.time_as_index(0)[0]]
        metrics['baseline_stability'] = np.std(baseline_data)
        
        return metrics
    
    def validate_features(self, features: pd.DataFrame) -> Dict:
        """
        验证特征数据质量
        
        Args:
            features: 特征DataFrame
            
        Returns:
            包含验证结果的字典
        """
        validation_results = {
            'passed': True,
            'issues': [],
            'quality_metrics': {}
        }
        
        # 基本统计
        n_samples = len(features)
        n_features = len(features.columns)
        
        validation_results['quality_metrics']['n_samples'] = n_samples
        validation_results['quality_metrics']['n_features'] = n_features
        
        # 缺失值检查
        missing_values = features.isnull().sum()
        missing_ratio = missing_values / n_samples
        
        if missing_ratio.max() > 0.5:
            validation_results['passed'] = False
            validation_results['issues'].append(f"特征缺失值过多: {missing_ratio.max():.2%}")
        
        validation_results['quality_metrics']['missing_ratio'] = missing_ratio.to_dict()
        
        # 异常值检查
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features) > 0:
            z_scores = np.abs((numeric_features - numeric_features.mean()) / numeric_features.std())
            outliers = (z_scores > 3).sum()
            outlier_ratio = outliers / numeric_features.size
            
            if outlier_ratio > 0.1:
                validation_results['issues'].append(f"异常值比例过高: {outlier_ratio:.2%}")
            
            validation_results['quality_metrics']['outlier_ratio'] = outlier_ratio
        
        # 特征相关性检查
        if len(numeric_features) > 1:
            corr_matrix = numeric_features.corr()
            high_corr_pairs = np.where(np.abs(corr_matrix) > 0.95)
            high_corr_count = len(set(zip(high_corr_pairs[0], high_corr_pairs[1]))) - len(corr_matrix)
            
            if high_corr_count > 0:
                validation_results['issues'].append(f"发现 {high_corr_count} 对高相关特征")
            
            validation_results['quality_metrics']['high_correlation_pairs'] = high_corr_count
        
        return validation_results
    
    def generate_validation_report(self, validation_results: List[Dict]) -> pd.DataFrame:
        """
        生成验证报告
        
        Args:
            validation_results: 验证结果列表
            
        Returns:
            包含验证摘要的DataFrame
        """
        report_data = []
        
        for result in validation_results:
            if 'subject' in result:
                # 原始数据或分段数据验证结果
                report_data.append({
                    'subject': result['subject'],
                    'passed': result['passed'],
                    'n_issues': len(result['issues']),
                    'issues': '; '.join(result['issues']) if result['issues'] else 'None'
                })
            else:
                # 特征数据验证结果
                report_data.append({
                    'data_type': 'features',
                    'passed': result['passed'],
                    'n_issues': len(result['issues']),
                    'issues': '; '.join(result['issues']) if result['issues'] else 'None'
                })
        
        report_df = pd.DataFrame(report_data)
        
        # 添加统计摘要
        summary = {
            'total_subjects': len([r for r in validation_results if 'subject' in r]),
            'passed_subjects': len([r for r in validation_results if 'subject' in r and r['passed']]),
            'failed_subjects': len([r for r in validation_results if 'subject' in r and not r['passed']]),
            'pass_rate': len([r for r in validation_results if 'subject' in r and r['passed']]) / 
                        len([r for r in validation_results if 'subject' in r]) if len([r for r in validation_results if 'subject' in r]) > 0 else 0
        }
        
        logger.info(f"验证报告摘要: {summary}")
        
        return report_df

