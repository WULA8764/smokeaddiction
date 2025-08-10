#!/usr/bin/env python3
"""
任务特征提取测试脚本 - 测试任务特定的特征提取和融合流程
Test script for task-specific feature extraction and fusion pipeline.
"""

import logging
from pathlib import Path
from typing import Dict

import mne
import numpy as np
import pandas as pd

from src.features.spectral import SpectralFeatureExtractor
from src.features.erp import ERPFeatureExtractor
from src.features.microstates import MicrostateFeatureExtractor
from src.features.connectivity import ConnectivityFeatureExtractor
from src.features.fusion import FeatureFusion
from src.utils.helpers import setup_logging, create_output_dirs

logger = logging.getLogger(__name__)


def main():
    """测试任务特定特征提取和融合的主函数"""
    # 设置日志
    setup_logging("INFO")
    
    # 创建输出目录
    output_dir = Path("outputs/test_features")
    create_output_dirs(str(output_dir))
    
    logger.info("开始测试任务特定特征提取和融合流程")

    # 定义测试任务
    tasks = ["resting_state", "visual_cue_task", "odor_task"]
    subjects = ["sub_001", "sub_002", "sub_003"]
    
    # 检查是否有预处理数据，如果没有则生成测试数据
    test_data_dir = Path("data/processed")
    if not test_data_dir.exists() or not any(test_data_dir.glob("*_raw.fif")):
        logger.info("未找到预处理数据，生成测试数据")
        generate_test_data(test_data_dir, subjects, tasks)

    # 初始化特征提取器
    logger.info("初始化特征提取器")
    
    # 频谱特征提取器
    spectral_extractor = SpectralFeatureExtractor(
        bands={'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)},
        window_length=2.0,
        overlap=0.5,
        method='welch'
    )

    # ERP特征提取器
    erp_extractor = ERPFeatureExtractor(
        components=['N1', 'P2', 'LPP'],
        time_windows={'N1': (0.08, 0.12), 'P2': (0.15, 0.25), 'LPP': (0.4, 0.8)},
        baseline=(-0.2, 0)
    )

    # 微状态特征提取器
    microstate_extractor = MicrostateFeatureExtractor(
        n_states=4,
        max_iterations=100,
        threshold=0.5,
        min_segment_length=10
    )

    # 功能连接特征提取器
    connectivity_extractor = ConnectivityFeatureExtractor(
        methods=['plv', 'correlation'],
        bands={'alpha': (8, 13), 'beta': (13, 30)},
        window_length=2.0,
        overlap=0.5
    )

    # 为每个任务提取特征
    all_task_features = {}
    
    for task_name in tasks:
        logger.info(f"测试任务: {task_name}")
        task_features = {}
        
        for subject in subjects:
            logger.info(f"测试受试者 {subject} 的任务 {task_name}")
            
            try:
                # 加载测试数据
                raw_file = test_data_dir / f"{subject}_{task_name}_raw.fif"
                epochs_file = test_data_dir / f"{subject}_{task_name}_epochs.fif"
                
                raw = None
                epochs = None
                
                if raw_file.exists():
                    raw = mne.io.read_raw_fif(raw_file, preload=True)
                    logger.info(f"已加载原始数据: {raw.info}")
                
                if epochs_file.exists():
                    epochs = mne.read_epochs(epochs_file, preload=True)
                    logger.info(f"已加载分段数据: {epochs.info}")
                
                subject_features = {}
                
                # 从原始数据提取特征
                if raw is not None:
                    # 提取频谱特征
                    spectral_features = spectral_extractor.extract_features(raw)
                    subject_features['spectral'] = spectral_features
                    logger.info(f"已提取频谱特征: {len(spectral_features)} 个特征")
                    
                    # 提取微状态特征
                    microstate_features = microstate_extractor.extract_features(raw)
                    subject_features['microstates'] = microstate_features
                    logger.info(f"已提取微状态特征: {len(microstate_features)} 个特征")
                    
                    # 提取功能连接特征
                    connectivity_features = connectivity_extractor.extract_features(raw)
                    subject_features['connectivity'] = connectivity_features
                    logger.info(f"已提取连接特征: {len(connectivity_features)} 个特征")
                
                # 从分段数据提取ERP特征（仅对事件相关任务）
                if epochs is not None and task_name in ["visual_cue_task", "odor_task"]:
                    erp_features = erp_extractor.extract_features(epochs)
                    subject_features['erp'] = erp_features
                    logger.info(f"已提取ERP特征: {len(erp_features)} 个特征")
                
                task_features[subject] = subject_features
                
            except Exception as e:
                logger.error(f"测试受试者 {subject} 任务 {task_name} 时出错: {e}")
                continue
        
        all_task_features[task_name] = task_features
        
        # 保存单个任务的特征
        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        task_df = convert_features_to_dataframe(task_features, task_name)
        if not task_df.empty:
            task_df.to_csv(task_output_dir / f"{task_name}_features.csv", index=False)
            task_df.to_parquet(task_output_dir / f"{task_name}_features.parquet", index=False)
            logger.info(f"已保存 {task_name} 特征: {len(task_df)} 个受试者, {len(task_df.columns)-1} 个特征")

    # 测试特征融合
    logger.info("测试特征融合")
    fusion = FeatureFusion(
        feature_types=['spectral', 'microstates', 'connectivity', 'erp'],
        scaling_method='standard',
        use_pca=False,
        n_components=50,
        feature_selection='none',
        n_features=100
    )
    
    fused_features = fusion.fuse_all_subjects(all_task_features)
    
    # 保存融合后的特征
    fusion_output_dir = output_dir / "fused"
    fusion_output_dir.mkdir(parents=True, exist_ok=True)
    
    if fused_features is not None and not fused_features.empty:
        fused_features.to_csv(fusion_output_dir / "fused_features.csv", index=False)
        fused_features.to_parquet(fusion_output_dir / "fused_features.parquet", index=False)
        logger.info(f"已保存融合特征: {len(fused_features)} 个受试者, {len(fused_features.columns)-1} 个特征")
        
        # 保存预处理器
        fusion.save_preprocessors(fusion_output_dir)
        logger.info("已保存特征预处理器")
        
        # 打印特征摘要
        summary = fusion.create_feature_summary(fused_features)
        logger.info("特征摘要:")
        logger.info(f"  总受试者数: {summary['total_subjects']}")
        logger.info(f"  总特征数: {summary['total_features']}")
        logger.info(f"  特征类型: {list(summary['feature_types'].keys())}")
        
        for feature_type, count in summary['feature_types'].items():
            logger.info(f"    {feature_type}: {count} 个特征")
    
    logger.info("任务特定特征提取和融合测试完成！")


def generate_test_data(data_dir: Path, subjects: list, tasks: list):
    """生成测试用的EEG数据"""
    logger.info("生成测试EEG数据")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模拟的EEG数据
    n_channels = 64
    sfreq = 1000  # 采样频率1000Hz
    
    for subject in subjects:
        for task in tasks:
            logger.info(f"生成 {subject} {task} 的测试数据")
            
            # 生成原始数据
            duration = 300 if task == "resting_state" else 180  # 静息态5分钟，其他任务3分钟
            n_samples = int(duration * sfreq)
            
            # 创建模拟EEG信号
            times = np.arange(n_samples) / sfreq
            
            # 生成不同频段的信号
            alpha_signal = np.sin(2 * np.pi * 10 * times)  # 10Hz alpha波
            beta_signal = np.sin(2 * np.pi * 20 * times)   # 20Hz beta波
            theta_signal = np.sin(2 * np.pi * 6 * times)   # 6Hz theta波
            
            # 添加噪声
            noise = np.random.normal(0, 0.1, n_samples)
            
            # 组合信号
            signal = alpha_signal + 0.5 * beta_signal + 0.3 * theta_signal + noise
            
            # 创建多通道数据
            data = np.tile(signal, (n_channels, 1))
            
            # 添加通道间差异
            for i in range(n_channels):
                data[i] += np.random.normal(0, 0.05, n_samples)
            
            # 创建MNE Raw对象
            ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            
            raw = mne.io.RawArray(data, info)
            
            # 保存原始数据
            raw_file = data_dir / f"{subject}_{task}_raw.fif"
            raw.save(raw_file, overwrite=True)
            
            # 为事件相关任务创建分段数据
            if task in ["visual_cue_task", "odor_task"]:
                # 创建模拟事件
                n_events = 30
                event_times = np.linspace(5, duration - 5, n_events)
                events = np.column_stack([
                    (event_times * sfreq).astype(int),
                    np.zeros(n_events, dtype=int),
                    np.ones(n_events, dtype=int)
                ])
                
                # 创建事件字典
                event_dict = {'stimulus': 1}
                
                # 创建分段数据
                epochs = mne.Epochs(raw, events, event_id=event_dict, 
                                  tmin=-0.2, tmax=1.0, baseline=(-0.2, 0),
                                  preload=True)
                
                # 保存分段数据
                epochs_file = data_dir / f"{subject}_{task}_epochs.fif"
                epochs.save(epochs_file, overwrite=True)
    
    logger.info("测试数据生成完成")


def convert_features_to_dataframe(task_features: Dict[str, Dict], task_name: str) -> pd.DataFrame:
    """将任务特征转换为DataFrame格式"""
    rows = []
    
    for subject, features in task_features.items():
        row = {'subject': subject, 'task': task_name}
        
        # 遍历每种特征类型
        for feature_type, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                for feature_name, feature_value in feature_dict.items():
                    if isinstance(feature_value, (int, float, str)):
                        # 直接存储标量值
                        row[f"{feature_type}_{feature_name}"] = feature_value
                    elif isinstance(feature_value, np.ndarray):
                        # 展平数组并存储
                        for i, val in enumerate(feature_value.flatten()):
                            row[f"{feature_type}_{feature_name}_{i}"] = val
                    elif isinstance(feature_value, list):
                        # 展平列表并存储
                        for i, val in enumerate(feature_value):
                            row[f"{feature_type}_{feature_name}_{i}"] = val
        
        rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
