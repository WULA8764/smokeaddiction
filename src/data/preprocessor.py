"""
EEG数据预处理工具 - 提供完整的EEG数据预处理流程

此模块实现了标准的EEG数据预处理步骤：
- 滤波：带通滤波、陷波滤波
- ICA：独立成分分析，去除眼动、心电等伪迹
- 伪迹拒绝：基于振幅和形态的试次拒绝
- 分段：事件相关分段和基线校正

预处理流程：
1. 滤波 → 2. ICA → 3. 伪迹拒绝 → 4. 分段
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne.filter import filter_data
from scipy import signal

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    EEG数据预处理器 - 实现完整的预处理流程
    
    此类提供了标准化的EEG数据预处理功能，包括滤波、ICA伪迹去除、
    伪迹拒绝和分段等步骤。
    """
    
    def __init__(self, config: Dict):
        """
        初始化预处理器
        
        Args:
            config: 包含预处理参数的配置字典
        """
        self.config = config
        # 提取各步骤的配置参数
        self.filter_config = config.get('filter', {})           # 滤波配置
        self.ica_config = config.get('ica', {})                 # ICA配置
        self.artifact_config = config.get('artifact_rejection', {})  # 伪迹拒绝配置
        self.epoching_config = config.get('epoching', {})       # 分段配置
        
    def preprocess_raw(self, raw: mne.io.Raw, subject: str) -> mne.io.Raw:
        """
        对原始数据进行完整的预处理流程
        
        Args:
            raw: MNE Raw对象
            subject: 被试ID（用于日志记录）
            
        Returns:
            预处理后的Raw对象
        """
        logger.info(f"开始为被试 {subject} 进行预处理")
        
        # 步骤1：滤波
        raw_filtered = self._apply_filtering(raw, subject)
        
        # 步骤2：ICA伪迹去除
        raw_ica = self._apply_ica(raw_filtered, subject)
        
        # 步骤3：伪迹拒绝
        raw_clean = self._apply_artifact_rejection(raw_ica, subject)
        
        logger.info(f"完成被试 {subject} 的预处理")
        return raw_clean
    
    def _apply_filtering(self, raw: mne.io.Raw, subject: str) -> mne.io.Raw:
        """
        应用带通滤波
        
        Args:
            raw: MNE Raw对象
            subject: 被试ID
            
        Returns:
            滤波后的Raw对象
        """
        logger.info(f"为被试 {subject} 应用滤波")
        
        # 获取滤波参数
        l_freq = self.filter_config.get('l_freq', 0.1)      # 低频截止
        h_freq = self.filter_config.get('h_freq', 100.0)    # 高频截止
        method = self.filter_config.get('method', 'iir')    # 滤波方法
        
        # 根据方法类型应用滤波
        if method == 'iir':
            iir_params = self.filter_config.get('iir_params', {})
            raw_filtered = raw.copy().filter(
                l_freq=l_freq, 
                h_freq=h_freq, 
                method='iir',
                iir_params=iir_params
            )
        else:
            raw_filtered = raw.copy().filter(
                l_freq=l_freq, 
                h_freq=h_freq, 
                method='fir'
            )
            
        return raw_filtered
    
    def _apply_ica(self, raw: mne.io.Raw, subject: str) -> mne.io.Raw:
        """
        应用ICA进行伪迹去除
        
        Args:
            raw: MNE Raw对象
            subject: 被试ID
            
        Returns:
            ICA清理后的Raw对象
        """
        logger.info(f"为被试 {subject} 应用ICA")
        
        # 获取ICA参数
        method = self.ica_config.get('method', 'fastica')
        n_components = self.ica_config.get('n_components', 15)
        max_iter = self.ica_config.get('max_iter', 200)
        
        # 创建ICA对象
        ica = ICA(
            n_components=n_components,
            method=method,
            max_iter=max_iter,
            random_state=42
        )
        
        # 拟合ICA
        ica.fit(raw)
        
        # 自动检测和去除伪迹成分
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
        
        # 去除检测到的伪迹成分
        bad_components = eog_indices + ecg_indices
        if bad_components:
            ica.exclude = bad_components
            raw_cleaned = ica.apply(raw)
            logger.info(f"被试 {subject}: 去除了 {len(bad_components)} 个伪迹成分")
        else:
            raw_cleaned = raw.copy()
            logger.info(f"被试 {subject}: 未检测到明显的伪迹成分")
            
        return raw_cleaned
    
    def _apply_artifact_rejection(self, raw: mne.io.Raw, subject: str) -> mne.io.Raw:
        """
        应用伪迹拒绝
        
        Args:
            raw: MNE Raw对象
            subject: 被试ID
            
        Returns:
            伪迹拒绝后的Raw对象
        """
        logger.info(f"为被试 {subject} 应用伪迹拒绝")
        
        # 获取拒绝阈值
        reject = self.artifact_config.get('reject', {})
        flat = self.artifact_config.get('flat', {})
        
        # 应用伪迹拒绝
        raw_cleaned = raw.copy()
        raw_cleaned.drop_bad(reject=reject, flat=flat)
        
        logger.info(f"被试 {subject}: 伪迹拒绝完成")
        return raw_cleaned
    
    def create_epochs(self, raw: mne.io.Raw, events: np.ndarray, subject: str) -> mne.Epochs:
        """
        创建事件相关分段
        
        Args:
            raw: MNE Raw对象
            events: 事件数组
            subject: 被试ID
            
        Returns:
            MNE Epochs对象
        """
        logger.info(f"为被试 {subject} 创建分段")
        
        # 获取分段参数
        tmin = self.epoching_config.get('tmin', -0.2)
        tmax = self.epoching_config.get('tmax', 1.0)
        baseline = self.epoching_config.get('baseline', (-0.2, 0))
        preload = self.epoching_config.get('preload', True)
        
        # 创建分段
        epochs = mne.Epochs(
            raw,
            events,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=preload,
            verbose=False
        )
        
        logger.info(f"被试 {subject}: 创建了 {len(epochs)} 个分段")
        return epochs
    
    def preprocess_epochs(self, epochs: mne.Epochs, subject: str) -> mne.Epochs:
        """
        对分段进行预处理
        
        Args:
            epochs: MNE Epochs对象
            subject: 被试ID
            
        Returns:
            预处理后的Epochs对象
        """
        logger.info(f"为被试 {subject} 预处理分段")
        
        # 获取拒绝参数
        reject = self.artifact_config.get('reject', {})
        flat = self.artifact_config.get('flat', {})
        
        # 应用分段级别的伪迹拒绝
        epochs_cleaned = epochs.copy()
        epochs_cleaned.drop_bad(reject=reject, flat=flat)
        
        logger.info(f"被试 {subject}: 分段预处理完成，保留 {len(epochs_cleaned)} 个分段")
        return epochs_cleaned
    
    def save_preprocessed_data(self, raw: mne.io.Raw, subject: str, task: str, output_dir: str):
        """
        保存预处理后的数据
        
        Args:
            raw: 预处理后的Raw对象
            subject: 被试ID
            task: 任务名称
            output_dir: 输出目录
        """
        output_path = Path(output_dir) / f"sub-{subject}_task-{task}_preprocessed_raw.fif"
        raw.save(output_path, overwrite=True)
        logger.info(f"保存预处理数据: {output_path}")
    
    def save_epochs(self, epochs: mne.Epochs, subject: str, task: str, output_dir: str):
        """
        保存分段数据
        
        Args:
            epochs: Epochs对象
            subject: 被试ID
            task: 任务名称
            output_dir: 输出目录
        """
        output_path = Path(output_dir) / f"sub-{subject}_task-{task}_epochs.fif"
        epochs.save(output_path, overwrite=True)
        logger.info(f"保存分段数据: {output_path}")
    
    def get_preprocessing_summary(self, raw_original: mne.io.Raw, raw_processed: mne.io.Raw) -> Dict:
        """
        获取预处理摘要信息
        
        Args:
            raw_original: 原始Raw对象
            raw_processed: 预处理后的Raw对象
            
        Returns:
            包含预处理摘要的字典
        """
        summary = {
            'original_duration': raw_original.times[-1],
            'processed_duration': raw_processed.times[-1],
            'original_channels': len(raw_original.ch_names),
            'processed_channels': len(raw_processed.ch_names),
            'bad_channels_removed': len(raw_original.info['bads']) - len(raw_processed.info['bads']),
            'filter_applied': True,
            'ica_applied': True
        }
        
        return summary

