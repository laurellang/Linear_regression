# Linear_regression：核物理数据回归

| 角色 | 主要修改位置 | 负责范围 | 给 AI 的起始说明 |
| --- | --- | --- | --- |
| 0 号（仓库维护） | `README.md`、`AGENTS.md`、`docs/architecture.md`、`pyproject.toml`、`src/linear_regression/experiment.py`、`src/linear_regression/reporting.py`、`src/tests/` 中对应测试、`src/linear_regression/experiments/final/` | 维护目录与接口约定、共享评估和结果工具、依赖与运行说明，最后整合比较图、最终测试和报告 | [0 号提示词](#0-号仓库维护) |
| 1 号（数据统计与预处理） | `src/linear_regression/data.py`、`src/linear_regression/experiments/data/`、`src/tests/` 中对应测试、`docs/analysis/data.md` | 数据概况与统计、缺失和异常检查、列解析、必要预处理与固定训练/验证划分；说明数据处理的理由 | [1 号提示词](#1-号数据统计与预处理) |
| 2 号（线性路线） | `src/linear_regression/experiments/linear/`、`docs/analysis/linear.md` | 一种线性或正则化回归算法，默认与优化方案、验证结果、预测图和分析 | [2 号提示词](#2-号线性路线) |
| 3 号（树模型路线） | `src/linear_regression/experiments/tree/`、`docs/analysis/tree.md` | 一种树或集成回归算法，默认与优化方案、验证结果、预测图和分析 | [3 号提示词](#3-号树模型路线) |
| 4 号（拓展路线） | `src/linear_regression/experiments/robustness/`、`docs/analysis/robustness.md` | 训练样本量和输入噪声实验，比较两种已完成模型的泛化与鲁棒性 | [4 号提示词](#4-号拓展路线) |

0 号和 1 号由同一位同学承担，因此团队仍为四人。表中的目录已建立，具体实现文件按任务创建；所有 Python 源码和测试都位于 `src/`。公共代码如需调整，任何成员都可以直接修改 `main`，但必须保持 [接口约定](docs/architecture.md) 一致，并先在本地运行相关测试。目前仓库只有设计与原始数据，尚无可运行的实验命令。

## 先读什么

- [作业要求](docs/机器学习-作业1-回归算法（作业要求）-2026%20%281%29.docx)：全部必做项与选做方向。
- [架构与接口约定](docs/architecture.md)：目录归属、公共函数、模型复用方式和验收条件。
- [术语表](GLOSSARY.md)：`k-inf`、`PPPF` 与燃料棒富集度。
- [代理协作入口](AGENTS.md)：GitHub Issues 与文档位置。

## 数据与实验约定

数据来自 [UCI Lattice-physics（ID 1091）](https://archive.ics.uci.edu/dataset/1091/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results))，许可为 CC BY 4.0。`dataset_UCI/raw.csv` 有 24,000 行，`dataset_UCI/test.csv` 有 360 行；两者均无表头，每行是以空白分隔的 41 个数值。UCI 说明有 39 个燃料棒富集度输入和两个连续输出，但没有给出文件列顺序。项目采用**待来源确认的列约定**：按从 1 开始计数，第 1 列为 `k-inf`，第 2 列为 `PPPF`，第 3–41 列为输入。这与各列的数值范围高度吻合。

本项目只预测 `k-inf`。输入仅用后 39 列；`PPPF` 也是输出，不能作为输入。`ucimlrepo.fetch_ucirepo(id=1091)` 无法直接导入该数据集，因为 UCI 没提供标准化数据地址；使用仓库内下载的原始文件。

所有模型用 `raw.csv` 的同一训练/验证划分：80%/20%，随机种子 42。预处理参数只从训练部分学习。1 号负责公共的数据检查与划分；各模型特有的标准化等变换由模型负责人放入各自的 Pipeline。模型负责人各自选择并说明调参依据；统一报告 MAE、RMSE、R²。`test.csv` 只在模型和参数确定后由 0 号统一运行最终评估，不参与调参、特征选择或鲁棒性试验。

## 交付顺序与协作

1. 0 号维护公共接口、依赖、评估与结果工具；1 号完成数据统计、预处理、列解析和固定划分。公共接口测试可用简易 `fit/predict` 替身，不把它当成正式回归模型。
2. 2、3 号在公共代码可用后并行完成各自算法、优化前后对比和文字分析。两条路线应体现不同建模思路。
3. 4 号在两种模型可复用后开展样本量与输入噪声分析，只使用训练/验证数据。
4. 0 号整合模型性能比较图、精选结果、最终测试和报告。题目提到的报告模板尚未放入仓库。

用 GitHub Issues 认领和记录任务。团队约定直接向 `main` 提交，不使用 PR，也暂不设置推送时自动检查。每次提交前先拉取最新代码，运行相关本地测试或实验，在 Issue 中记录运行命令、结果与提交号。生成的中间文件放在 `outputs/`，最终采用的图放在 `docs/figures/`，实验解释放在 `docs/analysis/`。

## 给 AI 的任务说明

先填写每段中的空白，再把提示词发给 AI。它只提供阅读范围与文件归属，具体任务由同学本人决定；对应 GitHub Issue 可作为补充背景。

### 0 号仓库维护

```text
请先阅读 README.md、AGENTS.md、docs/architecture.md、docs/agents/issue-tracker.md 和作业要求。我是 0 号同学，负责仓库维护；我也承担 1 号的数据工作，但本次会话按 0 号范围协作。可修改 README.md、AGENTS.md、docs/architecture.md、pyproject.toml、src/linear_regression/experiment.py、src/linear_regression/reporting.py、src/linear_regression/experiments/final/ 和 src/tests/ 中对应的测试。
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
