"""
频谱特征提取模块 - 提取EEG信号的频谱特征
Spectral feature extraction module for EEG signals.
"""

import logging
from typing import Dict, List, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """频谱特征提取器 - 提取功率谱密度、频带功率等特征"""

    def __init__(self, bands: Dict[str, Tuple[float, float]], 
                 window_length: float = 2.0, overlap: float = 0.5,
                 method: str = 'welch'):
        """初始化频谱特征提取器

        Args:
            bands: 频带定义，格式为 {'band_name': (fmin, fmax)}
            window_length: 窗口长度（秒）
            overlap: 窗口重叠比例
            method: 功率谱估计方法 ('welch', 'fft', 'multitaper')
        """
        self.bands = bands
        self.window_length = window_length
        self.overlap = overlap
        self.method = method
        
        logger.info(f"初始化频谱特征提取器: {len(bands)} 个频带, 方法={method}")

    def extract_features(self, raw: mne.io.Raw) -> Dict[str, Union[float, np.ndarray]]:
        """从原始EEG数据中提取频谱特征

        Args:
            raw: MNE原始数据对象

        Returns:
            包含频谱特征的字典
        """
        logger.info(f"开始提取频谱特征: {raw.info['nchan']} 通道, {raw.n_times} 采样点")
        
        # 获取数据
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        features = {}
        
        # 1. 计算功率谱密度
        psd_features = self._compute_power_spectral_density(data, sfreq)
        features.update(psd_features)
        
        # 2. 计算频带功率
        band_power_features = self._compute_band_power(data, sfreq)
        features.update(band_power_features)
        
        # 3. 计算频谱熵
        spectral_entropy_features = self._compute_spectral_entropy(data, sfreq)
        features.update(spectral_entropy_features)
        
        # 4. 计算峰值频率
        peak_freq_features = self._compute_peak_frequency(data, sfreq)
        features.update(peak_freq_features)
        
        # 5. 计算频谱边缘频率
        edge_freq_features = self._compute_spectral_edge_frequency(data, sfreq)
        features.update(edge_freq_features)
        
        # 6. 计算频带比率
        ratio_features = self._compute_band_ratios(band_power_features)
        features.update(ratio_features)
        
        logger.info(f"频谱特征提取完成: {len(features)} 个特征")
        return features

    def _compute_power_spectral_density(self, data: np.ndarray, sfreq: float) -> Dict[str, np.ndarray]:
        """计算功率谱密度

        Args:
            data: EEG数据矩阵 (channels, samples)
            sfreq: 采样频率

        Returns:
            功率谱密度特征字典
        """
        features = {}
        
        # 计算窗口参数
        nperseg = int(self.window_length * sfreq)
        noverlap = int(nperseg * self.overlap)
        
        for i, channel_data in enumerate(data):
            # 使用Welch方法计算功率谱密度
            freqs, psd = signal.welch(channel_data, sfreq, nperseg=nperseg, 
                                    noverlap=noverlap, method=self.method)
            
            # 存储主要频段的PSD
            for band_name, (fmin, fmax) in self.bands.items():
                # 找到频带内的频率索引
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                if np.any(band_mask):
                    # 计算频带内的平均PSD
                    mean_psd = np.mean(psd[band_mask])
                    features[f'psd_{band_name}_ch{i}'] = mean_psd
                    
                    # 计算频带内的最大PSD
                    max_psd = np.max(psd[band_mask])
                    features[f'psd_max_{band_name}_ch{i}'] = max_psd
                    
                    # 计算频带内的PSD标准差
                    std_psd = np.std(psd[band_mask])
                    features[f'psd_std_{band_name}_ch{i}'] = std_psd
        
        return features

    def _compute_band_power(self, data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """计算各频带功率

        Args:
            data: EEG数据矩阵
            sfreq: 采样频率

        Returns:
            频带功率特征字典
        """
        features = {}
        
        for band_name, (fmin, fmax) in self.bands.items():
            # 设计带通滤波器
            nyquist = sfreq / 2
            low = fmin / nyquist
            high = fmax / nyquist
            
            # 使用Butterworth滤波器
            b, a = signal.butter(4, [low, high], btype='band')
            
            # 对每个通道应用滤波器
            for i, channel_data in enumerate(data):
                # 滤波
                filtered_data = signal.filtfilt(b, a, channel_data)
                
                # 计算功率（均方根）
                power = np.mean(filtered_data ** 2)
                features[f'power_{band_name}_ch{i}'] = power
                
                # 计算功率的对数
                log_power = np.log10(power + 1e-10)  # 避免log(0)
                features[f'log_power_{band_name}_ch{i}'] = log_power
        
        return features

    def _compute_spectral_entropy(self, data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """计算频谱熵

        Args:
            data: EEG数据矩阵
            sfreq: 采样频率

        Returns:
            频谱熵特征字典
        """
        features = {}
        
        for i, channel_data in enumerate(data):
            # 计算功率谱
            freqs, psd = signal.welch(channel_data, sfreq)
            
            # 归一化功率谱
            psd_norm = psd / np.sum(psd)
            
            # 计算频谱熵
            spec_entropy = entropy(psd_norm)
            features[f'spectral_entropy_ch{i}'] = spec_entropy
        
        return features

    def _compute_peak_frequency(self, data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """计算峰值频率

        Args:
            data: EEG数据矩阵
            sfreq: 采样频率

        Returns:
            峰值频率特征字典
        """
        features = {}
        
        for i, channel_data in enumerate(data):
            # 计算功率谱
            freqs, psd = signal.welch(channel_data, sfreq)
            
            # 找到峰值频率
            peak_idx = np.argmax(psd)
            peak_freq = freqs[peak_idx]
            features[f'peak_frequency_ch{i}'] = peak_freq
            
            # 计算主要频段的峰值频率
            for band_name, (fmin, fmax) in self.bands.items():
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                if np.any(band_mask):
                    band_psd = psd[band_mask]
                    band_freqs = freqs[band_mask]
                    band_peak_idx = np.argmax(band_psd)
                    band_peak_freq = band_freqs[band_peak_idx]
                    features[f'peak_freq_{band_name}_ch{i}'] = band_peak_freq
        
        return features

    def _compute_spectral_edge_frequency(self, data: np.ndarray, sfreq: float, 
                                       percentiles: List[float] = [50, 90, 95]) -> Dict[str, float]:
        """计算频谱边缘频率

        Args:
            data: EEG数据矩阵
            sfreq: 采样频率
            percentiles: 百分位数列表

        Returns:
            频谱边缘频率特征字典
        """
        features = {}
        
        for i, channel_data in enumerate(data):
            # 计算功率谱
            freqs, psd = signal.welch(channel_data, sfreq)
            
            # 计算累积功率
            cumsum_psd = np.cumsum(psd)
            total_power = cumsum_psd[-1]
            
            # 计算各百分位数的边缘频率
            for percentile in percentiles:
                threshold = total_power * percentile / 100
                edge_idx = np.argmax(cumsum_psd >= threshold)
                edge_freq = freqs[edge_idx]
                features[f'edge_freq_{percentile}_ch{i}'] = edge_freq
        
        return features

    def _compute_band_ratios(self, band_power_features: Dict[str, float]) -> Dict[str, float]:
        """计算频带功率比率

        Args:
            band_power_features: 频带功率特征字典

        Returns:
            频带比率特征字典
        """
        features = {}
        
        # 提取各频带的总功率
        band_powers = {}
        for feature_name, value in band_power_features.items():
            if feature_name.startswith('power_') and not feature_name.startswith('log_power_'):
                parts = feature_name.split('_')
                if len(parts) >= 3:
                    band_name = parts[1]
                    if band_name not in band_powers:
                        band_powers[band_name] = []
                    band_powers[band_name].append(value)
        
        # 计算各频带的总功率
        total_powers = {}
        for band_name, powers in band_powers.items():
            total_powers[band_name] = np.mean(powers)
        
        # 计算比率
        if 'alpha' in total_powers and 'theta' in total_powers:
            features['alpha_theta_ratio'] = total_powers['alpha'] / (total_powers['theta'] + 1e-10)
        
        if 'beta' in total_powers and 'alpha' in total_powers:
            features['beta_alpha_ratio'] = total_powers['beta'] / (total_powers['alpha'] + 1e-10)
        
        if 'theta' in total_powers and 'beta' in total_powers:
            features['theta_beta_ratio'] = total_powers['theta'] / (total_powers['beta'] + 1e-10)
        
        # 计算总功率
        total_power = sum(total_powers.values())
        features['total_power'] = total_power
        
        return features
