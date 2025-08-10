#!/usr/bin/env python3
"""
主实验流程脚本 - 运行完整的EEG吸烟成瘾评估分析流程
Main experiment pipeline script for EEG-based smoking addiction assessment.
"""

import logging
from pathlib import Path
from typing import Dict, List

import hydra
import mne
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.loader import EEGDataLoader
from src.data.preprocessor import EEGPreprocessor
from src.data.validator import DataValidator
from src.features.spectral import SpectralFeatureExtractor
from src.features.erp import ERPFeatureExtractor
from src.features.microstates import MicrostateFeatureExtractor
from src.features.connectivity import ConnectivityFeatureExtractor
from src.features.fusion import FeatureFusion
from src.models.classifiers import EEGClassifier
from src.models.evaluator import ModelEvaluator
from src.models.interpreter import ModelInterpreter
from src.utils.helpers import setup_logging, create_output_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """主实验流程 - 从数据加载到模型评估的完整流程"""
    # 设置日志和输出目录
    setup_logging(cfg.global.log_level)
    create_output_dirs(cfg.global.output_dir)

    logger.info("开始EEG吸烟成瘾评估实验流程")
    logger.info(f"配置信息: {OmegaConf.to_yaml(cfg)}")

    # 步骤1: 数据加载和验证
    logger.info("步骤1: 数据加载和验证")
    data_loader = EEGDataLoader(
        raw_data_dir=cfg.data.raw_data_dir,
        processed_data_dir=cfg.data.processed_data_dir
    )

    # 获取受试者列表
    subjects = data_loader.list_subjects()
    logger.info(f"找到 {len(subjects)} 个受试者: {subjects}")

    # 数据验证
    validator = DataValidator(
        quality_thresholds=cfg.data.quality_thresholds,
        event_codes=cfg.data.event_codes
    )

    # 验证数据质量
    valid_subjects = []
    for subject in subjects:
        try:
            is_valid = validator.validate_subject_data(data_loader, subject)
            if is_valid:
                valid_subjects.append(subject)
            else:
                logger.warning(f"受试者 {subject} 数据质量不达标，跳过")
        except Exception as e:
            logger.error(f"验证受试者 {subject} 数据时出错: {e}")
            continue

    logger.info(f"通过质量验证的受试者: {len(valid_subjects)}/{len(subjects)}")

    # 步骤2: 数据预处理
    logger.info("步骤2: 数据预处理")
    preprocessor = EEGPreprocessor(
        filter_settings=cfg.preprocessing.filtering,
        artifact_removal=cfg.preprocessing.artifact_removal,
        epoching=cfg.preprocessing.epoching,
        baseline_correction=cfg.preprocessing.baseline_correction
    )

    # 为每个任务和受试者进行预处理
    for task_config in cfg.data.tasks:
        task_name = task_config.name
        logger.info(f"预处理任务: {task_name}")
        
        for subject in valid_subjects:
            logger.info(f"预处理受试者 {subject} 的任务 {task_name}")
            
            try:
                # 加载原始数据
                raw_data = data_loader.load_raw_data(subject, task_name)
                if raw_data is None:
                    logger.warning(f"未找到受试者 {subject} 任务 {task_name} 的原始数据")
                    continue

                # 应用预处理流程
                processed_raw = preprocessor.preprocess_raw(raw_data)
                
                # 保存预处理后的原始数据
                raw_output_file = data_loader.processed_data_dir / f"{subject}_{task_name}_raw.fif"
                processed_raw.save(raw_output_file, overwrite=True)
                logger.info(f"已保存预处理后的原始数据: {raw_output_file}")

                # 如果是事件相关任务，创建分段数据
                if task_name in ["visual_cue_task", "odor_task"]:
                    epochs = preprocessor.create_epochs(processed_raw, task_config.events)
                    if epochs is not None:
                        epochs_output_file = data_loader.processed_data_dir / f"{subject}_{task_name}_epochs.fif"
                        epochs.save(epochs_output_file, overwrite=True)
                        logger.info(f"已保存分段数据: {epochs_output_file}")

            except Exception as e:
                logger.error(f"预处理受试者 {subject} 任务 {task_name} 时出错: {e}")
                continue

    # 步骤3: 特征提取（任务特定）
    logger.info("步骤3: 为每个任务提取特征")
    
    # 初始化特征提取器
    spectral_extractor = SpectralFeatureExtractor(
        bands=cfg.features.spectral.bands,
        window_length=cfg.features.spectral.window_length,
        overlap=cfg.features.spectral.overlap,
        method=cfg.features.spectral.method
    )

    erp_extractor = None
    if cfg.features.erp.enabled:
        erp_extractor = ERPFeatureExtractor(
            components=cfg.features.erp.components,
            time_windows=cfg.features.erp.time_windows,
            baseline=cfg.features.erp.baseline
        )

    microstate_extractor = MicrostateFeatureExtractor(
        n_states=cfg.features.microstates.n_states,
        max_iterations=cfg.features.microstates.max_iterations,
        threshold=cfg.features.microstates.threshold,
        min_segment_length=cfg.features.microstates.min_segment_length
    )

    connectivity_extractor = ConnectivityFeatureExtractor(
        methods=cfg.features.connectivity.methods,
        bands=cfg.features.connectivity.bands,
        window_length=cfg.features.connectivity.window_length,
        overlap=cfg.features.connectivity.overlap
    )

    # 为每个任务提取特征
    all_task_features = {}
    
    for task_config in cfg.data.tasks:
        task_name = task_config.name
        logger.info(f"提取任务 {task_name} 的特征")
        
        task_features = {}
        
        for subject in valid_subjects:
            logger.info(f"提取受试者 {subject} 任务 {task_name} 的特征")
            
            try:
                # 加载预处理数据
                raw_file = data_loader.processed_data_dir / f"{subject}_{task_name}_raw.fif"
                epochs_file = data_loader.processed_data_dir / f"{subject}_{task_name}_epochs.fif"

                raw = None
                epochs = None

                if raw_file.exists():
                    raw = mne.io.read_raw_fif(raw_file, preload=True)

                if epochs_file.exists():
                    epochs = mne.read_epochs(epochs_file, preload=True)

                subject_features = {}

                # 从原始数据提取特征
                if raw is not None:
                    spectral_features = spectral_extractor.extract_features(raw)
                    subject_features['spectral'] = spectral_features

                    microstate_features = microstate_extractor.extract_features(raw)
                    subject_features['microstates'] = microstate_features

                    connectivity_features = connectivity_extractor.extract_features(raw)
                    subject_features['connectivity'] = connectivity_features

                # 从分段数据提取ERP特征
                if epochs is not None and erp_extractor is not None:
                    if task_name in ["visual_cue_task", "odor_task"]:
                        erp_features = erp_extractor.extract_features(epochs)
                        subject_features['erp'] = erp_features

                task_features[subject] = subject_features

            except Exception as e:
                logger.error(f"提取受试者 {subject} 任务 {task_name} 特征时出错: {e}")
                continue

        all_task_features[task_name] = task_features

    # 步骤4: 特征融合
    logger.info("步骤4: 融合所有任务的特征")
    fusion = FeatureFusion(
        feature_types=cfg.features.fusion.feature_types,
        scaling_method=cfg.features.fusion.scaling_method,
        use_pca=cfg.features.fusion.use_pca,
        n_components=cfg.features.fusion.n_components,
        feature_selection=cfg.features.fusion.feature_selection,
        n_features=cfg.features.fusion.n_features
    )

    fused_features = fusion.fuse_all_subjects(all_task_features)

    if fused_features is None or fused_features.empty:
        logger.error("特征融合失败，无法继续")
        return

    # 步骤5: 加载问卷数据
    logger.info("步骤5: 加载问卷数据")
    questionnaire_data = {}
    
    for subject in valid_subjects:
        try:
            ftnd_score = data_loader.load_ftnd_data(subject)
            bis11_score = data_loader.load_bis11_data(subject)
            
            questionnaire_data[subject] = {
                'ftnd_score': ftnd_score,
                'bis11_score': bis11_score
            }
        except Exception as e:
            logger.warning(f"加载受试者 {subject} 问卷数据时出错: {e}")
            # 使用默认值
            questionnaire_data[subject] = {
                'ftnd_score': 0,
                'bis11_score': 0
            }

    # 步骤6: 准备标签
    logger.info("步骤6: 准备分类标签")
    
    # 基于FTND分数创建标签（示例：FTND >= 6为吸烟成瘾）
    labels = []
    for subject in fused_features['subject']:
        if subject in questionnaire_data:
            ftnd_score = questionnaire_data[subject]['ftnd_score']
            # 根据FTND分数判断是否为吸烟成瘾（0=非成瘾，1=成瘾）
            label = 1 if ftnd_score >= 6 else 0
        else:
            # 如果没有问卷数据，使用随机标签（仅用于测试）
            label = np.random.randint(0, 2)
        
        labels.append(label)

    labels = np.array(labels)
    logger.info(f"标签分布: 非成瘾={np.sum(labels==0)}, 成瘾={np.sum(labels==1)}")

    # 步骤7: 模型训练和评估
    logger.info("步骤7: 模型训练和评估")
    
    # 准备特征矩阵（排除subject列）
    feature_cols = [col for col in fused_features.columns if col != 'subject']
    X = fused_features[feature_cols].values
    y = labels

    # 初始化分类器
    classifier = EEGClassifier(
        models=cfg.models.classifiers,
        cv_folds=cfg.models.cross_validation.folds,
        random_state=cfg.global.random_state
    )

    # 训练和评估模型
    results = classifier.train_and_evaluate(X, y)
    
    # 保存模型结果
    results_output_dir = Path(cfg.global.output_dir) / "results"
    results_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_output_dir / "model_results.csv", index=False)
    logger.info(f"已保存模型结果: {results_output_dir / 'model_results.csv'}")

    # 步骤8: 模型评估
    logger.info("步骤8: 详细模型评估")
    evaluator = ModelEvaluator()
    
    # 获取最佳模型
    best_model_name = results_df.loc[results_df['accuracy'].idxmax(), 'model_name']
    best_model = classifier.get_best_model()
    
    # 详细评估
    evaluation_results = evaluator.evaluate_model(best_model, X, y, cv_folds=5)
    
    # 保存评估结果
    evaluation_results.to_csv(results_output_dir / "evaluation_results.csv", index=False)
    logger.info(f"已保存评估结果: {results_output_dir / 'evaluation_results.csv'}")

    # 步骤9: 模型解释
    logger.info("步骤9: 模型解释和特征重要性分析")
    interpreter = ModelInterpreter()
    
    # 分析特征重要性
    feature_importance = interpreter.analyze_feature_importance(
        best_model, X, y, feature_names=feature_cols
    )
    
    # 保存特征重要性
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(results_output_dir / "feature_importance.csv", index=False)
    logger.info(f"已保存特征重要性: {results_output_dir / 'feature_importance.csv'}")

    # 步骤10: 保存结果
    logger.info("步骤10: 保存所有结果")
    
    # 保存融合特征
    fused_features.to_csv(results_output_dir / "fused_features.csv", index=False)
    fused_features.to_parquet(results_output_dir / "fused_features.parquet", index=False)
    
    # 保存标签
    labels_df = pd.DataFrame({
        'subject': fused_features['subject'],
        'label': labels
    })
    labels_df.to_csv(results_output_dir / "labels.csv", index=False)
    
    # 保存问卷数据
    questionnaire_df = pd.DataFrame([
        {
            'subject': subject,
            'ftnd_score': data['ftnd_score'],
            'bis11_score': data['bis11_score']
        }
        for subject, data in questionnaire_data.items()
    ])
    questionnaire_df.to_csv(results_output_dir / "questionnaire_data.csv", index=False)
    
    # 保存预处理器
    fusion.save_preprocessors(results_output_dir)
    
    logger.info("实验流程完成！")
    logger.info(f"所有结果已保存到: {results_output_dir}")


if __name__ == "__main__":
    main()
