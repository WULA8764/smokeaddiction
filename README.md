# EEG吸烟成瘾评估系统

基于EEG信号的吸烟成瘾评估系统，整合静息态、视觉线索和嗅闻任务的脑电数据，结合FTND和BIS-11问卷，进行特征提取、融合与分类分析。

## 项目概述

本系统旨在通过多任务EEG数据分析来评估吸烟成瘾程度，主要特点：

- **多任务分析**: 支持静息态、视觉线索任务、嗅闻任务三种EEG数据
- **多格式支持**: 支持Neuroscan、EEGLAB、Curry8、EGI等多种数据格式
- **特征丰富**: 提取频谱、ERP、微状态、功能连接等多种特征
- **任务特定**: 为每个任务分别提取特征，提供更精细的分析
- **可重现**: 使用Hydra配置管理，确保实验可重现性

## 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd smokeaddiction

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 2. 测试任务特定特征提取

```bash
# 测试更新后的特征提取流程
make test-features
```

### 3. 运行完整流程

```bash
# 运行完整分析流程
make pipeline

# 或分步运行：
make preprocess    # 数据预处理
make features      # 特征提取（任务特定）
make train         # 模型训练
make evaluate      # 模型评估
make report        # 生成报告
```

## 特征提取（任务特定）

系统现在为每个任务分别提取特征（静息态、视觉线索、嗅闻任务），而不是汇总处理。这提供了更详细的任务特定神经模式分析。

```python
from src.features.spectral import SpectralFeatureExtractor
from src.features.fusion import FeatureFusion

# 特征按任务提取，然后跨任务融合
# 每个任务获得自己的特征集：
# - resting_state: 频谱、微状态、连接特征
# - visual_cue_task: 频谱、微状态、连接、ERP特征
# - odor_task: 频谱、微状态、连接、ERP特征
```

## 项目结构

```
smokeaddiction/
├── configs/                 # 配置文件
│   ├── default.yaml        # 主配置
│   ├── data.yaml           # 数据配置
│   ├── preprocessing.yaml  # 预处理配置
│   ├── features.yaml       # 特征提取配置
│   └── models.yaml         # 模型配置
├── src/                    # 源代码
│   ├── data/              # 数据处理模块
│   ├── features/          # 特征提取模块
│   ├── models/            # 机器学习模块
│   └── utils/             # 工具函数
├── scripts/               # 执行脚本
├── data/                  # 数据目录
├── outputs/               # 输出结果
└── tests/                 # 测试代码
```

## 主要功能

### 数据加载与预处理
- 支持多种EEG数据格式（Neuroscan、EEGLAB、Curry8、EGI）
- 自动数据质量验证
- 完整的预处理流程（滤波、去伪迹、分段等）

### 特征提取
- **频谱特征**: 功率谱密度、频带功率、频谱熵等
- **ERP特征**: N1、P2、LPP等事件相关电位特征
- **微状态特征**: 微状态持续时间、覆盖率、出现频率等
- **功能连接**: 相位锁定值、相关性、互信息等

### 机器学习
- 多种分类器（SVM、随机森林、KNN、逻辑回归）
- 交叉验证和超参数优化
- 特征重要性分析
- 模型解释性分析

## 配置说明

系统使用Hydra进行配置管理，主要配置文件：

- `configs/default.yaml`: 主配置文件，包含所有模块的默认设置
- `configs/data.yaml`: 数据路径、格式、质量阈值等
- `configs/features.yaml`: 特征提取参数设置
- `configs/models.yaml`: 机器学习模型配置

## 输出结果

系统生成多种输出格式：

- **CSV/Parquet**: 特征数据和结果表格
- **图像**: 可视化图表和脑地形图
- **HTML报告**: 交互式分析报告
- **模型文件**: 训练好的模型和预处理器

## 开发指南

### 代码质量
- 使用Ruff进行代码格式化
- MyPy进行类型检查
- Pytest进行单元测试
- Pre-commit钩子确保代码质量

### 添加新特征
1. 在`src/features/`下创建新的特征提取器
2. 更新`configs/features.yaml`配置
3. 在`scripts/extract_features.py`中集成新特征
4. 添加相应的测试

### 添加新模型
1. 在`src/models/`下实现新的分类器
2. 更新`configs/models.yaml`配置
3. 在`scripts/train_models.py`中集成新模型

## 常见问题

### Q: 如何处理不同格式的EEG数据？
A: 系统通过`src/data/loader.py`中的`EEGDataLoader`类自动识别和处理不同格式的数据。

### Q: 如何调整特征提取参数？
A: 修改`configs/features.yaml`中的相应参数，系统会自动应用新的设置。

### Q: 如何添加新的任务类型？
A: 在`configs/data.yaml`中的`tasks`列表中添加新任务配置。

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至项目维护者

---

**注意**: 本系统仅供研究使用，不应用于临床诊断。使用前请确保遵守相关伦理规范和法律法规。
