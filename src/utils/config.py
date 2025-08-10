"""
配置管理工具 - 管理配置文件和设置

此模块提供了完整的配置管理功能：
- 配置文件加载和保存
- 配置合并和覆盖
- 配置验证和模式检查
- 深度字典合并

支持的配置格式：
- YAML格式配置文件
- 嵌套字典结构
- 配置验证模式
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    配置管理器 - 管理配置文件和设置
    
    此类提供了完整的配置管理功能，包括配置文件的加载、保存、
    合并和验证，支持YAML格式的配置文件。
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 包含配置文件的目录
        """
        self.config_dir = Path(config_dir)
        self.configs = {}  # 缓存已加载的配置
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_name: 配置文件名（不包含.yaml扩展名）
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 当配置文件不存在时
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        # 读取YAML配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 缓存配置
        self.configs[config_name] = config
        logger.info(f"已加载配置: {config_name}")
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_name: str):
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            config_name: 配置文件名
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为YAML格式
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        # 更新缓存
        self.configs[config_name] = config
        logger.info(f"已保存配置: {config_name}")
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        获取已加载的配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置字典，如果未加载则返回None
        """
        return self.configs.get(config_name)
    
    def merge_configs(self, base_config: str, override_config: str) -> Dict[str, Any]:
        """
        合并两个配置
        
        Args:
            base_config: 基础配置名称
            override_config: 覆盖配置名称
            
        Returns:
            合并后的配置字典
        """
        # 加载两个配置
        base = self.load_config(base_config)
        override = self.load_config(override_config)
        
        # 深度合并
        merged = self._deep_merge(base, override)
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
            
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in override.items():
            # 如果键存在于基础字典中且都是字典类型，则递归合并
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                # 否则直接覆盖
                result[key] = value
                
        return result
    
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        验证配置是否符合模式
        
        Args:
            config: 要验证的配置字典
            schema: 验证模式字典
            
        Returns:
            是否通过验证
        """
        try:
            # 检查必需字段
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in config:
                    logger.error(f"缺少必需字段: {field}")
                    return False
            
            # 检查字段类型
            field_types = schema.get('types', {})
            for field, expected_type in field_types.items():
                if field in config:
                    if not isinstance(config[field], expected_type):
                        logger.error(f"字段 {field} 类型错误: 期望 {expected_type}, 实际 {type(config[field])}")
                        return False
            
            # 检查字段值范围
            field_ranges = schema.get('ranges', {})
            for field, (min_val, max_val) in field_ranges.items():
                if field in config:
                    value = config[field]
                    if value < min_val or value > max_val:
                        logger.error(f"字段 {field} 值超出范围: {value} 不在 [{min_val}, {max_val}] 内")
                        return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有已加载的配置
        
        Returns:
            所有配置的字典
        """
        return self.configs.copy()
    
    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """
        重新加载配置文件
        
        Args:
            config_name: 配置名称
            
        Returns:
            重新加载的配置字典
        """
        # 从缓存中移除
        if config_name in self.configs:
            del self.configs[config_name]
        
        # 重新加载
        return self.load_config(config_name)
    
    def list_config_files(self) -> list:
        """
        列出所有可用的配置文件
        
        Returns:
            配置文件列表
        """
        config_files = []
        if self.config_dir.exists():
            for file_path in self.config_dir.glob("*.yaml"):
                config_files.append(file_path.stem)
        
        return sorted(config_files)

