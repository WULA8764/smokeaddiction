"""
特征融合模块 - 将多种特征类型融合为统一的特征集
Feature fusion module for combining multiple feature types into unified feature sets.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureFusion:
    """特征融合类 - 融合和整合多种特征类型"""

    def __init__(self, feature_types: List[str], scaling_method: str = 'standard',
                 use_pca: bool = False, n_components: int = 50,
                 feature_selection: str = 'mutual_info', n_features: int = 100):
        """初始化特征融合模块

        Args:
            feature_types: 要融合的特征类型列表
            scaling_method: 标准化方法 ('standard', 'minmax', 'none')
            use_pca: 是否应用PCA降维
            n_components: PCA组件数量
            feature_selection: 特征选择方法 ('mutual_info', 'f_classif', 'rf', 'none')
            n_features: 要选择的特征数量
        """
        self.feature_types = feature_types
        self.scaling_method = scaling_method
        self.use_pca = use_pca
        self.n_components = n_components
        self.feature_selection = feature_selection
        self.n_features = n_features

        # 初始化标准化器和转换器
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}

    def fuse_features(self, task_features: Dict[str, Dict], subject: str) -> Dict[str, Union[float, int, str]]:
        """为单个受试者融合多种特征类型

        Args:
            task_features: 任务名称到特征字典的映射
            subject: 受试者ID

        Returns:
            融合后的特征字典
        """
        logger.info(f"正在融合受试者 {subject} 的特征")

        fused_features = {'subject': subject}

        # 融合每个任务的特征
        for task_name, task_feature_dict in task_features.items():
            if subject in task_feature_dict:
                subject_features = task_feature_dict[subject]

                # 融合此任务的每种特征类型
                for feature_type, feature_dict in subject_features.items():
                    if isinstance(feature_dict, dict):
                        for feature_name, feature_value in feature_dict.items():
                            if isinstance(feature_value, (int, float, str)):
                                # 使用任务名称作为前缀来区分不同任务的特征
                                fused_features[f'{task_name}_{feature_type}_{feature_name}'] = feature_value
                            elif isinstance(feature_value, np.ndarray):
                                # 展平数组
                                for i, val in enumerate(feature_value.flatten()):
                                    fused_features[f'{task_name}_{feature_type}_{feature_name}_{i}'] = val
                            elif isinstance(feature_value, list):
                                # 展平列表
                                for i, val in enumerate(feature_value):
                                    fused_features[f'{task_name}_{feature_type}_{feature_name}_{i}'] = val

        return fused_features

    def fuse_all_subjects(self, all_task_features: Dict[str, Dict]) -> Optional[pd.DataFrame]:
        """融合所有受试者的特征

        Args:
            all_task_features: 所有任务的特征字典

        Returns:
            融合后的特征DataFrame
        """
        logger.info("开始融合所有受试者的特征")

        # 收集所有受试者
        all_subjects = set()
        for task_features in all_task_features.values():
            all_subjects.update(task_features.keys())

        logger.info(f"找到 {len(all_subjects)} 个受试者")

        # 为每个受试者融合特征
        fused_subjects = []
        for subject in all_subjects:
            try:
                fused_features = self.fuse_features(all_task_features, subject)
                fused_subjects.append(fused_features)
            except Exception as e:
                logger.error(f"融合受试者 {subject} 特征时出错: {e}")
                continue

        if not fused_subjects:
            logger.error("没有成功融合任何受试者的特征")
            return None

        # 转换为DataFrame
        fused_df = pd.DataFrame(fused_subjects)
        logger.info(f"成功融合 {len(fused_df)} 个受试者的特征")

        # 预处理特征
        processed_df = self.preprocess_features(fused_df)
        return processed_df

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理融合后的特征

        Args:
            df: 原始特征DataFrame

        Returns:
            预处理后的特征DataFrame
        """
        logger.info("开始预处理特征")

        # 处理缺失值
        df = self._handle_missing_values(df)

        # 分离特征列和元数据列
        feature_cols = [col for col in df.columns if col != 'subject']
        metadata_cols = ['subject']

        if not feature_cols:
            logger.warning("没有找到特征列")
            return df

        # 应用标准化
        df = self._apply_scaling(df, feature_cols)

        # 应用降维
        if self.use_pca:
            df = self._apply_dimensionality_reduction(df, feature_cols)

        # 应用特征选择
        df = self._apply_feature_selection(df, feature_cols)

        logger.info(f"特征预处理完成，最终特征数量: {len([col for col in df.columns if col != 'subject'])}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值

        Args:
            df: 输入DataFrame

        Returns:
            处理缺失值后的DataFrame
        """
        # 保留subject列
        subject_col = df['subject'] if 'subject' in df.columns else None

        # 对数值列用中位数填充，其他列用众数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # 对其他列用众数填充
        other_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in other_cols:
            if col != 'subject':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])

        # 恢复subject列
        if subject_col is not None:
            df['subject'] = subject_col

        return df

    def _apply_scaling(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """应用特征标准化

        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表

        Returns:
            标准化后的DataFrame
        """
        if self.scaling_method == 'none':
            return df

        # 选择标准化器
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"未知的标准化方法: {self.scaling_method}")
            return df

        # 应用标准化
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        self.scalers[self.scaling_method] = scaler

        logger.info(f"应用了 {self.scaling_method} 标准化")
        return df_scaled

    def _apply_dimensionality_reduction(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """应用降维

        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表

        Returns:
            降维后的DataFrame
        """
        if not self.use_pca or len(feature_cols) <= self.n_components:
            return df

        # 应用PCA
        pca = PCA(n_components=min(self.n_components, len(feature_cols)))
        pca_features = pca.fit_transform(df[feature_cols])

        # 创建新的DataFrame
        df_reduced = df.copy()
        df_reduced = df_reduced.drop(columns=feature_cols)

        # 添加PCA特征
        for i in range(pca_features.shape[1]):
            df_reduced[f'pca_component_{i}'] = pca_features[:, i]

        self.pca_transformers['pca'] = pca
        logger.info(f"PCA降维完成，从 {len(feature_cols)} 个特征降至 {pca_features.shape[1]} 个特征")
        return df_reduced

    def _apply_feature_selection(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """应用特征选择

        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表

        Returns:
            特征选择后的DataFrame
        """
        if self.feature_selection == 'none' or len(feature_cols) <= self.n_features:
            return df

        # 选择特征选择方法
        if self.feature_selection == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        elif self.feature_selection == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=self.n_features)
        elif self.feature_selection == 'rf':
            # 使用随机森林特征重要性
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            # 这里需要标签，暂时跳过
            logger.warning("随机森林特征选择需要标签，暂时跳过")
            return df
        else:
            logger.warning(f"未知的特征选择方法: {self.feature_selection}")
            return df

        # 应用特征选择（注意：这里需要标签，暂时跳过）
        logger.warning("特征选择需要标签，暂时跳过")
        return df

    def create_feature_summary(self, df: pd.DataFrame) -> Dict:
        """创建特征摘要

        Args:
            df: 特征DataFrame

        Returns:
            特征摘要字典
        """
        summary = {
            'total_subjects': len(df),
            'total_features': len(df.columns) - 1,  # 减去subject列
            'feature_types': {},
            'missing_values': df.isnull().sum().to_dict(),
            'basic_stats': {}
        }

        # 按特征类型分组统计
        feature_cols = [col for col in df.columns if col != 'subject']
        for col in feature_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                feature_type = f"{parts[0]}_{parts[1]}"  # task_feature_type
                if feature_type not in summary['feature_types']:
                    summary['feature_types'][feature_type] = 0
                summary['feature_types'][feature_type] += 1

        # 基本统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['basic_stats'] = df[numeric_cols].describe().to_dict()

        return summary

    def save_preprocessors(self, output_dir: Path) -> None:
        """保存预处理器

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config = {
            'feature_types': self.feature_types,
            'scaling_method': self.scaling_method,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'feature_selection': self.feature_selection,
            'n_features': self.n_features
        }

        with open(output_dir / 'fusion_config.pkl', 'wb') as f:
            pickle.dump(config, f)

        # 保存预处理器
        for name, preprocessor in self.scalers.items():
            with open(output_dir / f'{name}_scaler.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)

        for name, preprocessor in self.pca_transformers.items():
            with open(output_dir / f'{name}_pca.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)

        for name, preprocessor in self.feature_selectors.items():
            with open(output_dir / f'{name}_selector.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)

        logger.info(f"预处理器已保存到 {output_dir}")

    def load_preprocessors(self, input_dir: Path) -> None:
        """加载预处理器

        Args:
            input_dir: 输入目录
        """
        input_dir = Path(input_dir)

        # 加载配置
        with open(input_dir / 'fusion_config.pkl', 'rb') as f:
            config = pickle.load(f)

        self.feature_types = config['feature_types']
        self.scaling_method = config['scaling_method']
        self.use_pca = config['use_pca']
        self.n_components = config['n_components']
        self.feature_selection = config['feature_selection']
        self.n_features = config['n_features']

        # 加载预处理器
        for scaler_file in input_dir.glob('*_scaler.pkl'):
            name = scaler_file.stem.replace('_scaler', '')
            with open(scaler_file, 'rb') as f:
                self.scalers[name] = pickle.load(f)

        for pca_file in input_dir.glob('*_pca.pkl'):
            name = pca_file.stem.replace('_pca', '')
            with open(pca_file, 'rb') as f:
                self.pca_transformers[name] = pickle.load(f)

        for selector_file in input_dir.glob('*_selector.pkl'):
            name = selector_file.stem.replace('_selector', '')
            with open(selector_file, 'rb') as f:
                self.feature_selectors[name] = pickle.load(f)

        logger.info(f"预处理器已从 {input_dir} 加载")

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换新数据

        Args:
            df: 新数据DataFrame

        Returns:
            转换后的DataFrame
        """
        logger.info("转换新数据")

        # 分离特征列和元数据列
        feature_cols = [col for col in df.columns if col != 'subject']
        metadata_cols = ['subject']

        if not feature_cols:
            logger.warning("没有找到特征列")
            return df

        # 应用保存的预处理器
        df_transformed = df.copy()

        # 应用标准化
        if self.scaling_method in self.scalers:
            df_transformed[feature_cols] = self.scalers[self.scaling_method].transform(df[feature_cols])

        # 应用PCA
        if 'pca' in self.pca_transformers:
            pca_features = self.pca_transformers['pca'].transform(df[feature_cols])
            df_transformed = df_transformed.drop(columns=feature_cols)
            for i in range(pca_features.shape[1]):
                df_transformed[f'pca_component_{i}'] = pca_features[:, i]

        # 应用特征选择
        if self.feature_selection in self.feature_selectors:
            df_transformed = self.feature_selectors[self.feature_selection].transform(df_transformed)

        logger.info("新数据转换完成")
        return df_transformed
