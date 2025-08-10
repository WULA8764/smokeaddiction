"""
EEG数据加载工具 - 支持多种格式的EEG数据加载

此模块提供了统一的EEG数据加载接口，支持以下格式：
- Neuroscan (.cnt)
- EEGLAB (.set/.fdt)
- Curry8 (.cdt/.dpa)
- EGI (.raw)

主要功能：
- 自动格式检测
- 批量数据加载
- 问卷数据加载
- 数据信息获取
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_cnt, read_raw_eeglab, read_raw_egi
import eeglabio

logger = logging.getLogger(__name__)


class EEGDataLoader:
    """
    EEG数据加载器 - 支持多种格式的EEG数据加载
    
    此类提供了统一的接口来加载不同格式的EEG数据，
    包括Neuroscan、EEGLAB、Curry8和EGI格式。
    """
    
    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        # 支持的数据格式定义
        self.supported_formats = {
            'neuroscan': ['.cnt'],      # Neuroscan格式
            'eeglab': ['.set', '.fdt'], # EEGLAB格式
            'curry': ['.cdt', '.dpa'],  # Curry8格式
            'egi': ['.raw']             # EGI格式
        }
        
    def list_subjects(self, task: Optional[str] = None) -> List[str]:
        """
        列出所有可用的被试
        
        Args:
            task: 可选的任务过滤器
            
        Returns:
            被试ID列表
        """
        subjects = set()
        
        # 遍历所有支持的数据格式
        for format_name, extensions in self.supported_formats.items():
            for ext in extensions:
                pattern = f"*{ext}"
                if task:
                    pattern = f"*task-{task}*{ext}"
                    
                # 查找匹配的文件
                files = glob.glob(str(self.data_dir / "raw" / format_name / pattern))
                for file_path in files:
                    subject = self._extract_subject_from_filename(file_path)
                    if subject:
                        subjects.add(subject)
                        
        return sorted(list(subjects))
    
    def load_raw_data(self, subject: str, task: str, format_type: str = "auto") -> mne.io.Raw:
        """
        加载指定被试和任务的原始EEG数据
        
        Args:
            subject: 被试ID
            task: 任务名称
            format_type: 数据格式类型（'auto', 'neuroscan', 'eeglab', 'curry', 'egi'）
            
        Returns:
            MNE Raw对象
            
        Raises:
            FileNotFoundError: 当找不到数据文件时
            ValueError: 当格式不支持时
        """
        # 自动检测格式
        if format_type == "auto":
            format_type = self._detect_format(subject, task)
            
        # 查找数据文件
        file_path = self._find_data_file(subject, task, format_type)
        if not file_path:
            raise FileNotFoundError(f"No data file found for subject {subject}, task {task}")
            
        logger.info(f"Loading {format_type} data: {file_path}")
        
        # 根据格式类型加载数据
        if format_type == "neuroscan":
            raw = read_raw_cnt(file_path, preload=True)
        elif format_type == "eeglab":
            raw = read_raw_eeglab(file_path, preload=True)
        elif format_type == "curry":
            raw = mne.io.read_raw_curry(file_path, preload=True)
        elif format_type == "egi":
            raw = read_raw_egi(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        return raw
    
    def load_questionnaire_data(self, subject: str, questionnaire: str) -> pd.DataFrame:
        """
        加载问卷数据（FTND、BIS-11）
        
        Args:
            subject: 被试ID
            questionnaire: 问卷类型（'ftnd', 'bis11'）
            
        Returns:
            问卷数据DataFrame
            
        Raises:
            FileNotFoundError: 当找不到问卷文件时
        """
        # 构建问卷文件路径
        questionnaire_dir = self.data_dir / "questionnaires"
        file_patterns = {
            'ftnd': f"sub-{subject}_ftnd.csv",
            'bis11': f"sub-{subject}_bis11.csv"
        }
        
        if questionnaire not in file_patterns:
            raise ValueError(f"Unsupported questionnaire: {questionnaire}")
            
        file_path = questionnaire_dir / file_patterns[questionnaire]
        if not file_path.exists():
            raise FileNotFoundError(f"Questionnaire file not found: {file_path}")
            
        # 加载问卷数据
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {questionnaire} data for subject {subject}")
        
        return data
    
    def load_all_subjects_data(self, task: str, subjects: Optional[List[str]] = None) -> Dict[str, mne.io.Raw]:
        """
        批量加载所有被试的数据
        
        Args:
            task: 任务名称
            subjects: 被试列表，如果为None则加载所有被试
            
        Returns:
            被试ID到Raw对象的字典
        """
        if subjects is None:
            subjects = self.list_subjects(task)
            
        data_dict = {}
        for subject in subjects:
            try:
                raw = self.load_raw_data(subject, task)
                data_dict[subject] = raw
                logger.info(f"Successfully loaded data for subject {subject}")
            except Exception as e:
                logger.warning(f"Failed to load data for subject {subject}: {e}")
                
        return data_dict
    
    def _detect_format(self, subject: str, task: str) -> str:
        """
        自动检测数据格式
        
        Args:
            subject: 被试ID
            task: 任务名称
            
        Returns:
            检测到的格式类型
        """
        # 遍历所有支持格式，查找匹配的文件
        for format_name, extensions in self.supported_formats.items():
            for ext in extensions:
                pattern = f"*sub-{subject}*task-{task}*{ext}"
                files = glob.glob(str(self.data_dir / "raw" / format_name / pattern))
                if files:
                    return format_name
                    
        raise ValueError(f"Could not detect format for subject {subject}, task {task}")
    
    def _find_data_file(self, subject: str, task: str, format_type: str) -> Optional[Path]:
        """
        查找数据文件路径
        
        Args:
            subject: 被试ID
            task: 任务名称
            format_type: 数据格式类型
            
        Returns:
            数据文件路径，如果找不到则返回None
        """
        if format_type not in self.supported_formats:
            return None
            
        format_dir = self.data_dir / "raw" / format_type
        for ext in self.supported_formats[format_type]:
            pattern = f"*sub-{subject}*task-{task}*{ext}"
            files = glob.glob(str(format_dir / pattern))
            if files:
                return Path(files[0])
                
        return None
    
    def _extract_subject_from_filename(self, file_path: str) -> Optional[str]:
        """
        从文件名中提取被试ID
        
        Args:
            file_path: 文件路径
            
        Returns:
            被试ID，如果无法提取则返回None
        """
        filename = Path(file_path).name
        
        # 尝试从文件名中提取被试ID
        # 假设文件名格式为: sub-{subject_id}_task-{task}_...
        if 'sub-' in filename:
            parts = filename.split('_')
            for part in parts:
                if part.startswith('sub-'):
                    return part.replace('sub-', '')
                    
        return None
    
    def get_data_info(self, subject: str, task: str) -> Dict:
        """
        获取数据信息
        
        Args:
            subject: 被试ID
            task: 任务名称
            
        Returns:
            包含数据信息的字典
        """
        try:
            raw = self.load_raw_data(subject, task)
            info = {
                'subject': subject,
                'task': task,
                'n_channels': len(raw.ch_names),
                'n_samples': len(raw.times),
                'duration': raw.times[-1],
                'sfreq': raw.info['sfreq'],
                'channels': raw.ch_names,
                'format': self._detect_format(subject, task)
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get data info for subject {subject}, task {task}: {e}")
            return {}

