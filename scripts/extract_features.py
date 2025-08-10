#!/usr/bin/env python3
"""
特征提取脚本 - 从预处理后的EEG数据中提取多种特征
Feature extraction script for extracting multiple features from preprocessed EEG data.
"""

import logging
from pathlib import Path
from typing import Dict

import hydra
import mne
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.loader import EEGDataLoader
from src.features.spectral import SpectralFeatureExtractor
from src.features.erp import ERPFeatureExtractor
from src.features.microstates import MicrostateFeatureExtractor
from src.features.connectivity import ConnectivityFeatureExtractor
from src.features.fusion import FeatureFusion
from src.utils.helpers import setup_logging, create_output_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """主特征提取流程 - 为每个任务分别提取特征，然后融合"""
    # 设置日志和输出目录
    setup_logging(cfg.global.log_level)
    create_output_dirs(cfg.global.output_dir)

    logger.info("开始特征提取流程")
    logger.info(f"配置信息: {OmegaConf.to_yaml(cfg)}")

    # 初始化数据加载器
    data_loader = EEGDataLoader(
        raw_data_dir=cfg.data.raw_data_dir,
        processed_data_dir=cfg.data.processed_data_dir
    )

    # 获取受试者列表
    subjects = data_loader.list_subjects()
    logger.info(f"找到 {len(subjects)} 个受试者: {subjects}")

    # 初始化特征提取器
    logger.info("初始化特征提取器")
    
    # 频谱特征提取器 - 提取功率谱密度、频带功率等
    spectral_extractor = SpectralFeatureExtractor(
        bands=cfg.features.spectral.bands,
        window_length=cfg.features.spectral.window_length,
        overlap=cfg.features.spectral.overlap,
        method=cfg.features.spectral.method
    )

    # ERP特征提取器 - 提取事件相关电位特征（N1、P2、LPP等）
    erp_extractor = None
    if cfg.features.erp.enabled:
        erp_extractor = ERPFeatureExtractor(
            components=cfg.features.erp.components,
            time_windows=cfg.features.erp.time_windows,
            baseline=cfg.features.erp.baseline
        )

    # 微状态特征提取器 - 提取微状态分析特征
    microstate_extractor = MicrostateFeatureExtractor(
        n_states=cfg.features.microstates.n_states,
        max_iterations=cfg.features.microstates.max_iterations,
        threshold=cfg.features.microstates.threshold,
        min_segment_length=cfg.features.microstates.min_segment_length
    )

    # 功能连接特征提取器 - 提取相位锁定值、相关性等连接特征
    connectivity_extractor = ConnectivityFeatureExtractor(
        methods=cfg.features.connectivity.methods,
        bands=cfg.features.connectivity.bands,
        window_length=cfg.features.connectivity.window_length,
        overlap=cfg.features.connectivity.overlap
    )

    # 为每个任务分别处理特征提取
    all_task_features = {}

    for task_config in cfg.data.tasks:
        task_name = task_config.name
        logger.info(f"正在处理任务: {task_name}")

        task_features = {}

        for subject in subjects:
            logger.info(f"正在处理受试者 {subject} 的任务 {task_name}")

            try:
                # 加载此任务的预处理数据
                raw_file = data_loader.processed_data_dir / f"{subject}_{task_name}_raw.fif"
                epochs_file = data_loader.processed_data_dir / f"{subject}_{task_name}_epochs.fif"

                if not raw_file.exists() and not epochs_file.exists():
                    logger.warning(f"未找到受试者 {subject} 任务 {task_name} 的预处理数据")
                    continue

                raw = None
                epochs = None

                # 加载原始数据（用于频谱、微状态、连接特征）
                if raw_file.exists():
                    raw = mne.io.read_raw_fif(raw_file, preload=True)
                    logger.info(f"已加载 {subject} {task_name} 的原始数据: {raw.info}")

                # 加载分段数据（用于ERP特征）
                if epochs_file.exists():
                    epochs = mne.read_epochs(epochs_file, preload=True)
                    logger.info(f"已加载 {subject} {task_name} 的分段数据: {epochs.info}")

                subject_features = {}

                # 从原始数据提取特征
                if raw is not None:
                    # 提取频谱特征（功率谱密度、频带功率等）
                    spectral_features = spectral_extractor.extract_features(raw)
                    subject_features['spectral'] = spectral_features

                    # 提取微状态特征（微状态持续时间、覆盖率等）
                    microstate_features = microstate_extractor.extract_features(raw)
                    subject_features['microstates'] = microstate_features

                    # 提取功能连接特征（相位锁定值、相关性等）
                    connectivity_features = connectivity_extractor.extract_features(raw)
                    subject_features['connectivity'] = connectivity_features

                # 从分段数据提取ERP特征（仅对事件相关任务）
                if epochs is not None and erp_extractor is not None:
                    if task_name in ["visual_cue_task", "odor_task"]:
                        erp_features = erp_extractor.extract_features(epochs)
                        subject_features['erp'] = erp_features

                task_features[subject] = subject_features

            except Exception as e:
                logger.error(f"处理受试者 {subject} 任务 {task_name} 时出错: {e}")
                continue

        all_task_features[task_name] = task_features

        # 保存单个任务的特征
        task_output_dir = Path(cfg.global.output_dir) / "features" / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        task_df = _convert_features_to_dataframe(task_features, task_name)
        if not task_df.empty:
            task_df.to_csv(task_output_dir / f"{task_name}_features.csv", index=False)
            task_df.to_parquet(task_output_dir / f"{task_name}_features.parquet", index=False)
            logger.info(f"已保存 {task_name} 特征: {len(task_df)} 个受试者, {len(task_df.columns)-1} 个特征")

    # 融合所有任务的特征
    logger.info("正在融合所有任务的特征")
    fusion = FeatureFusion(
        feature_types=cfg.features.fusion.feature_types,
        scaling_method=cfg.features.fusion.scaling_method,
        use_pca=cfg.features.fusion.use_pca,
        n_components=cfg.features.fusion.n_components,
        feature_selection=cfg.features.fusion.feature_selection,
        n_features=cfg.features.fusion.n_features
    )

    fused_features = fusion.fuse_all_subjects(all_task_features)

    # 保存融合后的特征
    fusion_output_dir = Path(cfg.global.output_dir) / "features" / "fused"
    fusion_output_dir.mkdir(parents=True, exist_ok=True)

    if fused_features is not None and not fused_features.empty:
        fused_features.to_csv(fusion_output_dir / "fused_features.csv", index=False)
        fused_features.to_parquet(fusion_output_dir / "fused_features.parquet", index=False)
        logger.info(f"已保存融合特征: {len(fused_features)} 个受试者, {len(fused_features.columns)-1} 个特征")

        # 保存预处理器以便后续使用
        fusion.save_preprocessors(fusion_output_dir)
        logger.info("已保存特征预处理器")

    logger.info("特征提取流程完成")


def _convert_features_to_dataframe(task_features: Dict[str, Dict], task_name: str) -> pd.DataFrame:
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
