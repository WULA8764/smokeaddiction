"""
辅助工具函数 - 为EEG分析系统提供通用工具函数

此模块提供了各种辅助功能：
- 日志系统设置
- 目录创建和管理
- 文件操作工具
- 时间格式化
- 安全文件名处理

主要功能：
- 日志配置和输出
- 输出目录结构创建
- 项目路径管理
- 文件大小和时长格式化
"""

import logging
from pathlib import Path
from typing import Optional
import os
import time
import re

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    设置日志系统配置
    
    Args:
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: 可选的日志文件路径
    """
    # 创建日志格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"日志系统设置完成 - 级别: {level}")


def create_output_dirs(base_dir: str, subdirs: Optional[list] = None):
    """
    创建输出目录结构
    
    Args:
        base_dir: 基础输出目录
        subdirs: 要创建的子目录列表
    """
    if subdirs is None:
        # 默认的输出目录结构
        subdirs = [
            "processed",    # 预处理数据
            "features",     # 特征数据
            "models",       # 训练模型
            "reports",      # 分析报告
            "figures",      # 图表图像
            "dashboards"    # 交互式仪表板
        ]
    
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 创建所有子目录
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"已在 {base_dir} 中创建输出目录结构")


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    # 查找pyproject.toml或setup.py来识别项目根目录
    current = Path.cwd()
    
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    
    # 回退到当前目录
    return Path.cwd()


def ensure_dir(path: str):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """
    获取文件大小（MB）
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小（MB）
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def format_duration(seconds: float) -> str:
    """
    格式化持续时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def safe_filename(filename: str) -> str:
    """
    生成安全的文件名
    
    Args:
        filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    # 移除或替换不安全的字符
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格和点
    safe_name = re.sub(r'\s+', '_', safe_name.strip())
    safe_name = re.sub(r'\.+', '.', safe_name)
    # 确保不以点开头或结尾
    safe_name = safe_name.strip('.')
    
    return safe_name


def get_timestamp() -> str:
    """
    获取当前时间戳
    
    Returns:
        格式化的时间戳字符串
    """
    return time.strftime("%Y%m%d_%H%M%S")


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小显示
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def check_disk_space(path: str) -> float:
    """
    检查磁盘可用空间
    
    Args:
        path: 路径
        
    Returns:
        可用空间（GB）
    """
    try:
        statvfs = os.statvfs(path)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        return free_bytes / (1024**3)  # 转换为GB
    except OSError:
        return 0.0


def validate_file_path(file_path: str) -> bool:
    """
    验证文件路径是否有效
    
    Args:
        file_path: 文件路径
        
    Returns:
        路径是否有效
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件扩展名（包含点）
    """
    return Path(file_path).suffix


def is_valid_yaml_file(file_path: str) -> bool:
    """
    检查是否为有效的YAML文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否为有效的YAML文件
    """
    try:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True
    except Exception:
        return False

