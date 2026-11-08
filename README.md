# Linear_regression：核物理数据回归

| 角色 | 主要修改位置 | 负责范围 | 给 AI 的起始说明 |
| --- | --- | --- | --- |
| 0 号（仓库维护） | `README.md`、`AGENTS.md`、`docs/architecture.md`、`pyproject.toml`、`src/linear_regression/experiment.py`、`src/linear_regression/reporting.py`、`src/tests/` 中对应测试、`src/linear_regression/experiments/final/` | 维护目录与接口约定、共享评估和结果工具、依赖与运行说明，最后整合比较图、最终测试和报告 | [0 号提示词](#0-号仓库维护) |
| 1 号（数据统计与预处理） | `src/linear_regression/data.py`、`src/linear_regression/experiments/data/`、`src/tests/` 中对应测试、`docs/analysis/data.md` | 数据概况与统计、缺失和异常检查、列解析、必要预处理与固定训练/验证划分；说明数据处理的理由 | [1 号提示词](#1-号数据统计与预处理) |
| 2 号（线性路线） | `src/linear_regression/experiments/linear/`、`docs/analysis/linear.md` | 一种线性或正则化回归算法，默认与优化方案、验证结果、预测图和分析 | [2 号提示词](#2-号线性路线) |
| 3 号（树模型路线） | `src/linear_regression/experiments/tree/`、`docs/analysis/tree.md` | 随机森林模型、验证结果、预测图和分析；不要求优化前后对照 | [3 号提示词](#3-号树模型路线) |
| 4 号（拓展路线） | `src/linear_regression/experiments/robustness/`、`docs/analysis/robustness.md` | 训练样本量和输入噪声实验，比较线性与树模型的泛化与鲁棒性 | [4 号提示词](#4-号拓展路线) |

0 号和 4 号由同一位同学承担，因此团队仍为四人。所有 Python 源码和测试都位于 `src/`。公共代码如需调整，任何成员都可以直接修改 `main`，但必须保持 [接口约定](docs/architecture.md) 一致，并先在本地运行相关测试。数据读取、两条模型路线和鲁棒性实验入口已实现；0 号的通用训练评价、结果输出和最终测试入口仍待完成。

## 先读什么

- [作业要求](docs/机器学习-作业1-回归算法（作业要求）-2026%20%281%29.docx)：全部必做项与选做方向。
- [架构与接口约定](docs/architecture.md)：目录归属、公共函数、模型复用方式和验收条件。
- [数据预处理分析](docs/analysis/data.md)：已完成的数据检查、处理理由、接口字段和报告参考文字。
- [术语表](GLOSSARY.md)：`k-inf`、`PPPF` 与燃料棒富集度。
- [代理协作入口](AGENTS.md)：GitHub Issues 与文档位置。

## 数据与实验约定

数据来自 [UCI Lattice-physics（ID 1091）](https://archive.ics.uci.edu/dataset/1091/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results))，许可为 CC BY 4.0。`dataset_UCI/raw.csv` 有 24,000 行，`dataset_UCI/test.csv` 有 360 行；两者均无表头，每行是以空白分隔的 41 个数值。UCI 说明有 39 个燃料棒富集度输入和两个连续输出，但没有给出文件列顺序。项目采用**待来源确认的列约定**：按从 1 开始计数，第 1 列为 `k-inf`，第 2 列为 `PPPF`，第 3–41 列为输入。这与各列的数值范围高度吻合。

本项目只预测 `k-inf`。输入仅用后 39 列；`PPPF` 也是输出，不能作为输入。`ucimlrepo.fetch_ucirepo(id=1091)` 无法直接导入该数据集，因为 UCI 没提供标准化数据地址；使用仓库内下载的原始文件。

所有模型用 `raw.csv` 的同一训练/验证划分：80%/20%，随机种子 42。预处理参数只从训练部分学习。1 号负责公共的数据检查与划分；各模型特有的标准化等变换由模型负责人放入各自的 Pipeline。统一报告 MAE、RMSE、R²；线性路线包含 L2 参数优化，树路线不要求优化前后对照。`test.csv` 只在模型和参数确定后由 0 号统一运行最终评估，不参与调参、特征选择或鲁棒性试验。

## 已完成的数据工作

- `src/linear_regression/data.py`：按空白解析并严格检查 41 列、数值和有限值；提取 39 个输入及 `k-inf`，排除 `PPPF`；提供固定验证划分和最终评价划分。
- `src/linear_regression/experiments/data/run.py`：检查开发数据并生成统计、目标分布图和划分清单；不读取独立测试集。
- `src/tests/test_data.py`：3 项测试覆盖列解析与错误处理、划分可复现及互斥、最终评价输入重叠检查等行为。
- `docs/analysis/data.md`：记录实测结果、处理依据和模型组员交接说明。

对 `raw.csv` 的检查未发现缺失、非有限值、重复输入或常数特征；39 个输入的范围为 0.700003153–4.999999766，特征 IQR 检查未标记数值。因此保留全部 24,000 条样本及 39 个特征，不做填补、类别编码、删行或全局标准化。上述检查不能替代列含义的来源确认，列顺序假设仍保留。默认划分得到 **19,200 条训练样本、4,800 条验证样本**；独立测试集尚未用于最终评估。

## 运行数据检查

数据模块和审计脚本需要 **Python 3.10+、NumPy**。当前尚无 `pyproject.toml`，先在仓库根目录设置 `PYTHONPATH` 运行。以下为 PowerShell 命令；如当前 Python 环境缺少 NumPy，先执行 `python -m pip install numpy`。

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s src/tests -v
python -m linear_regression.experiments.data.run
```

审计结果写入 `outputs/data/`：

| 文件 | 内容 |
| --- | --- |
| `audit.json` | 数据检查摘要、源文件哈希及划分规模 |
| `feature_statistics.csv` | 39 个特征的分位数、均值和标准差 |
| `split.csv` | 训练/验证样本对应的原始行号 |
| `target_distribution.svg` | 开发集 `k-inf` 分布图 |

可用 `--input dataset_UCI/raw.csv --output outputs/data` 指定输入与输出目录。本次数据交付的 3 项测试及审计命令已通过；模型训练需要的 scikit-learn 等依赖与命令后续补充。

## 模型组员如何读取数据

设置上述 `PYTHONPATH` 后，从仓库根目录运行自己的模型脚本，复用公共入口，不自行解析列或重新划分：

```python
from linear_regression.data import load_dataset, make_validation_split

data = load_dataset('dataset_UCI/raw.csv')
split = make_validation_split(data)  # 默认 seed=42，validation_fraction=0.2
# model 为组员构建的模型或 Pipeline
model.fit(split.X_train, split.y_train)
predictions = model.predict(split.X_eval)
```

评价时将预测值与 `split.y_eval` 对齐，并保留同一顺序的 `split.evaluation_ids`（例如 `raw.csv:123`，行号从 1 开始），以核对不同模型使用相同验证样本。需要标准化的模型把变换放入 Pipeline；交叉验证时在每个训练折内拟合。完整字段说明见 [数据分析文档](docs/analysis/data.md#组员调用接口)。

`make_final_split(raw, holdout)` 已提供给最终整合脚本：参数冻结后才读取独立测试文件，以完整开发集训练，并拒绝开发集与测试集之间重复的输入。此函数只提供数据，不训练模型或拟合预处理。

## 交付顺序与协作

1. 1 号已完成数据统计、列解析、校验和固定划分；0 号继续完成共享训练评价、结果工具与依赖配置。公共接口测试可用简易 `fit/predict` 替身，不把它当成正式回归模型。
2. 2 号完成线性算法与优化前后对比；3 号完成随机森林路线。两条路线体现不同建模思路。
3. 4 号复用两种模型开展样本量与输入噪声分析，只使用训练/验证数据。运行协议见 [鲁棒性分析](docs/analysis/robustness.md)。
4. 0 号整合模型性能比较图、精选结果、最终测试和报告。题目提到的报告模板尚未放入仓库。

用 GitHub Issues 认领和记录任务。团队约定直接向 `main` 提交，不使用 PR，也暂不设置推送时自动检查。每次提交前先拉取最新代码，运行相关本地测试或实验，在 Issue 中记录运行命令、结果与提交号。实验生成的指标、预测和图统一放在 `outputs/`，实验解释放在 `docs/analysis/`。

## 运行鲁棒性实验

从仓库根目录运行，需安装 NumPy、scikit-learn 和 Matplotlib。该命令会多次训练完整模型，耗时取决于计算机配置。

```powershell
$env:PYTHONPATH = 'src'
python -m linear_regression.experiments.robustness.run
```

输出在 `outputs/robustness/`：`results.csv`、`protocol.json`、两张趋势图，以及各条件的 `metrics.json` 和 `predictions.csv`。只使用 `raw.csv` 的固定训练/验证划分，不读取 `test.csv`。

## 给 AI 的任务说明

先填写每段中的空白，再把提示词发给 AI。它只提供阅读范围与文件归属，具体任务由同学本人决定；对应 GitHub Issue 可作为补充背景。

### 0 号仓库维护

```text
请先阅读 README.md、AGENTS.md、docs/architecture.md、docs/agents/issue-tracker.md 和作业要求。我是 0 号同学，负责仓库维护；我也承担 4 号的拓展工作，但本次会话按 0 号范围协作。可修改 README.md、AGENTS.md、docs/architecture.md、pyproject.toml、src/linear_regression/experiment.py、src/linear_regression/reporting.py、src/linear_regression/experiments/final/ 和 src/tests/ 中对应的测试。
本次任务要求：【由我填写】
希望你怎样协助：【由我填写】
完成标准：【由我填写】
```

### 1 号数据统计与预处理

```text
请先阅读 README.md、GLOSSARY.md、docs/architecture.md、作业要求、dataset_UCI/ 下的数据文件，以及已有的 src/linear_regression/data.py 和 docs/analysis/data.md（若已创建）。我是 1 号同学，负责数据统计与预处理。可修改 src/linear_regression/data.py、src/linear_regression/experiments/data/、docs/analysis/data.md 和 src/tests/ 中对应的数据测试。
本次任务要求：【由我填写】
希望你怎样协助：【由我填写】
完成标准：【由我填写】
```

### 2 号线性路线

```text
请先阅读 README.md、GLOSSARY.md、docs/architecture.md、作业要求，以及已有的数据代码和分析文档。我是 2 号同学，负责线性路线。可修改 src/linear_regression/experiments/linear/ 和 docs/analysis/linear.md。
本次任务要求：【由我填写】
希望你怎样协助：【由我填写】
完成标准：【由我填写】
```

### 3 号树模型路线

```text
请先阅读 README.md、GLOSSARY.md、docs/architecture.md、作业要求，以及已有的数据代码和分析文档。我是 3 号同学，负责树模型路线。可修改 src/linear_regression/experiments/tree/ 和 docs/analysis/tree.md。
本次任务要求：【由我填写】
希望你怎样协助：【由我填写】
完成标准：【由我填写】
```

### 4 号拓展路线

```text
请先阅读 README.md、GLOSSARY.md、docs/architecture.md、作业要求，以及 2、3 号同学已完成的模型文件与分析文档。我是 4 号同学，负责拓展路线。可修改 src/linear_regression/experiments/robustness/ 和 docs/analysis/robustness.md。
本次任务要求：【由我填写】
希望你怎样协助：【由我填写】
完成标准：【由我填写】
```
