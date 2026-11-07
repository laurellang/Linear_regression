# 树模型

本模型：原始 39 个输入＋逐行均值＋200 棵随机森林。开发验证 MAE 为 **0.00276236**、RMSE 为 **0.00350186**、R² 为 **0.961094**。方法与图表见 [分析文档](../../../../docs/analysis/tree.md)。

## 目录

| 文件 | 作用 |
| --- | --- |
| `model.py` | 均值特征与模型构造 |
| `run.py` | 数据检查、训练、指标、图表和可选三折验证 |
| `test_tree.py` | 当前模型的行为测试 |
| `params.json` | 固定参数、数据指纹与已核验结果 |
| `requirements.txt` | 已验证依赖 |

## 安装与复现

已验证环境为 Python 3.13.17。以下 Git Bash 命令在仓库根目录执行：

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r src/linear_regression/experiments/tree/requirements.txt
export PYTHONPATH=src
./.venv/Scripts/python.exe -m linear_regression.experiments.tree.run --threads 4
```

默认读取 `dataset_UCI/raw.csv`，使用公共接口固定划分 19,200 条训练／4,800 条开发验证样本，输出至 `outputs/tree/`：

- `metrics.json`：实际计算的指标和运行元数据。
- `predictions.csv`：按原始行号对齐的真实值、预测值和残差。
- `prediction_vs_true.png`、`residuals.png`：当前模型图表。

需要复核当前模型的训练内三折表现时，增加 `--cv`：

```bash
./.venv/Scripts/python.exe -m linear_regression.experiments.tree.run --threads 4 --cv
```

这会额外生成 `cv_metrics.json`、`oof_predictions.csv`，不会搜索参数。独立测试文件不进入上述流程。

## 测试与组员调用

```bash
./.venv/Scripts/python.exe -m unittest discover -s src/tests -v
./.venv/Scripts/python.exe -m unittest linear_regression.experiments.tree.test_tree -v
```

```python
from linear_regression.experiments.tree.model import build_model

model = build_model('optimized')
model.fit(split.X_train, split.y_train)  # 输入保持原始39列
prediction = model.predict(split.X_eval)
```

每次构造均返回未训练模型，不读取数据或本地输出。入口仅支持 `optimized`。固定参数不是训练好的权重文件，每次运行会重新拟合。


