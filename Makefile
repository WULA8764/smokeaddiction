# 基于EEG的吸烟成瘾评估系统 - Makefile
# 提供便捷的命令来管理项目开发、数据处理和模型训练

.PHONY: help install dev-install test lint format clean
.PHONY: data preprocess features train evaluate report dashboard
.PHONY: pipeline setup run-experiment validate-data benchmark docs release

# 默认目标：显示帮助信息
help:
	@echo "基于EEG的吸烟成瘾评估系统 - 可用命令："
	@echo ""
	@echo "环境设置："
	@echo "  install        ## 安装项目依赖"
	@echo "  dev-install    ## 安装开发依赖"
	@echo "  setup          ## 完整项目设置"
	@echo ""
	@echo "代码质量："
	@echo "  test           ## 运行测试"
	@echo "  lint           ## 代码检查"
	@echo "  format         ## 代码格式化"
	@echo "  clean          ## 清理临时文件"
	@echo ""
	@echo "数据处理："
	@echo "  data           ## 准备数据"
	@echo "  preprocess     ## 预处理EEG数据"
	@echo "  features       ## 提取特征（任务特定）"
	@echo "  test-features  ## 测试特征提取"
	@echo ""
	@echo "模型训练："
	@echo "  train          ## 训练模型"
	@echo "  evaluate       ## 评估模型"
	@echo "  benchmark      ## 模型性能基准测试"
	@echo ""
	@echo "结果输出："
	@echo "  report         ## 生成报告"
	@echo "  dashboard      ## 启动交互式仪表板"
	@echo ""
	@echo "完整流程："
	@echo "  pipeline       ## 运行完整分析流程"
	@echo "  run-experiment ## 运行实验"
	@echo "  validate-data  ## 验证数据质量"
	@echo ""
	@echo "文档和发布："
	@echo "  docs           ## 生成文档"
	@echo "  release        ## 发布新版本"

# 环境设置
install:
	## 安装项目依赖
	pip install -e .

dev-install:
	## 安装开发依赖
	pip install -e ".[dev]"

setup:
	## 完整项目设置
	@echo "设置项目环境..."
	pip install -e ".[dev]"
	pre-commit install
	@echo "项目设置完成！"

# 代码质量检查
test:
	## 运行测试
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	## 代码检查
	ruff check src/ tests/ scripts/
	mypy src/

format:
	## 代码格式化
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

clean:
	## 清理临时文件
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf outputs/ logs/ temp/

# 数据处理
data:
	## 准备数据
	@echo "请将原始EEG数据放置在 data/raw/ 目录中"
	@echo "支持格式：Neuroscan (.cnt), EEGLAB (.set/.fdt), Curry8 (.cdt/.dpa), EGI (.raw)"

preprocess:
	## 预处理EEG数据
	python scripts/preprocess.py

features:
	## 提取特征（任务特定）
	python scripts/extract_features.py

test-features:
	## 测试特征提取
	python scripts/test_task_features.py

# 模型训练和评估
train:
	## 训练模型
	python scripts/train_models.py

evaluate:
	## 评估模型
	python scripts/evaluate_models.py

benchmark:
	## 模型性能基准测试
	python scripts/benchmark_models.py

# 结果输出
report:
	## 生成报告
	python scripts/generate_report.py

dashboard:
	## 启动交互式仪表板
	python scripts/start_dashboard.py

# 完整流程
pipeline:
	## 运行完整分析流程
	@echo "开始完整EEG分析流程..."
	python scripts/run_experiment.py

run-experiment:
	## 运行实验
	python scripts/run_experiment.py

validate-data:
	## 验证数据质量
	python scripts/validate_data.py

# 文档和发布
docs:
	## 生成文档
	sphinx-build -b html docs/ docs/_build/html

release:
	## 发布新版本
	@echo "请手动更新版本号并创建发布标签"
	@echo "1. 更新 pyproject.toml 中的版本号"
	@echo "2. 提交更改：git commit -am 'Release vX.Y.Z'"
	@echo "3. 创建标签：git tag -a vX.Y.Z -m 'Release vX.Y.Z'"
	@echo "4. 推送：git push origin main --tags"
