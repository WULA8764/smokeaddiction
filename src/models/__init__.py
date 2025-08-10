"""
机器学习模型模块 - 负责分类模型训练、评估和解释

此模块提供了完整的机器学习流程：
- EEGClassifier: 多种分类器的训练和预测
- ModelEvaluator: 模型性能评估和比较
- ModelInterpreter: 模型解释和特征重要性分析

支持的分类器：
- 支持向量机 (SVM)
- 随机森林 (Random Forest)
- K近邻 (KNN)
- 逻辑回归 (Logistic Regression)

评估功能：
- 交叉验证
- 多种评估指标（准确率、精确率、召回率、F1等）
- ROC曲线和混淆矩阵
- 特征重要性分析
- 模型解释（SHAP、LIME）
"""

# 导入主要类，方便用户直接使用
from .classifiers import EEGClassifier
from .evaluator import ModelEvaluator
from .interpreter import ModelInterpreter

# 定义模块的公共API
__all__ = [
    "EEGClassifier",
    "ModelEvaluator",
    "ModelInterpreter"
]

