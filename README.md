# 机器学习 HW1：核物理数据回归

| 成员 | 主要修改位置 | 交付内容 | 给 AI 的起始说明 |
| --- | --- | --- | --- |
| 1 号（仓库负责人） | `src/hw1/`、`tests/`、`pyproject.toml`、`docs/architecture.md`；最后整合 `docs/analysis/` 与 `docs/figures/` | 公共数据、划分、评估、绘图和结果保存代码；接口测试；最终测试与报告整合。不实现正式回归模型 | [1 号提示词](#1-号仓库负责人) |
| 2 号（线性路线） | `experiments/linear/`、`docs/analysis/linear.md` | 一种线性或正则化回归算法，默认与优化方案、验证结果、预测图和分析 | [2 号提示词](#2-号线性路线) |
| 3 号（树模型路线） | `experiments/tree/`、`docs/analysis/tree.md` | 一种树或集成回归算法，默认与优化方案、验证结果、预测图和分析 | [3 号提示词](#3-号树模型路线) |
| 4 号（拓展路线） | `experiments/robustness/`、`docs/analysis/robustness.md` | 训练样本量和输入噪声实验，比较两种已完成模型的泛化与鲁棒性 | [4 号提示词](#4-号拓展路线) |

表中的代码目录是计划中的归属位置，按任务实施时创建。公共代码如需调整，任何成员都可以直接修改 `main`，但必须保持 [接口约定](docs/architecture.md) 一致，并先在本地运行相关测试。目前仓库只有设计与原始数据，尚无可运行的实验命令。

## 先读什么

- [作业要求](docs/机器学习-作业1-回归算法（作业要求）-2026%20%281%29.docx)：全部必做项与选做方向。
- [架构与接口约定](docs/architecture.md)：目录归属、公共函数、模型复用方式和验收条件。
- [术语表](GLOSSARY.md)：`k-inf`、`PPPF` 与燃料棒富集度。
- [代理协作入口](AGENTS.md)：GitHub Issues 与文档位置。

## 数据与实验约定

数据来自 [UCI Lattice-physics（ID 1091）](https://archive.ics.uci.edu/dataset/1091/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results))，许可为 CC BY 4.0。`dataset_UCI/raw.csv` 有 24,000 行，`dataset_UCI/test.csv` 有 360 行；两者均无表头，每行是以空白分隔的 41 个数值。UCI 说明有 39 个燃料棒富集度输入和两个连续输出，但没有给出文件列顺序。项目采用**待来源确认的列约定**：按从 1 开始计数，第 1 列为 `k-inf`，第 2 列为 `PPPF`，第 3–41 列为输入。这与各列的数值范围高度吻合。

本项目只预测 `k-inf`。输入仅用后 39 列；`PPPF` 也是输出，不能作为输入。`ucimlrepo.fetch_ucirepo(id=1091)` 无法直接导入该数据集，因为 UCI 没提供标准化数据地址；使用仓库内下载的原始文件。

所有模型用 `raw.csv` 的同一训练/验证划分：80%/20%，随机种子 42。预处理参数只从训练部分学习。模型负责人各自选择并说明调参依据；统一报告 MAE、RMSE、R²。`test.csv` 只在模型和参数确定后由 1 号统一运行最终评估，不参与调参、特征选择或鲁棒性试验。

## 交付顺序与协作

1. 1 号完成公共框架和接口测试，测试中可使用简易 `fit/predict` 替身，但不提交正式回归模型。
2. 2、3 号在公共框架上并行完成各自算法、优化前后对比和文字分析。两条路线应体现不同建模思路。
3. 4 号在两种模型可复用后开展样本量与输入噪声分析，只使用训练/验证数据。
4. 1 号整合模型性能比较图、精选结果、最终测试和报告。题目提到的报告模板尚未放入仓库。

用 GitHub Issues 认领和记录任务。团队约定直接向 `main` 提交，不使用 PR，也暂不设置推送时自动检查。每次提交前先拉取最新代码，运行相关本地测试或实验，在 Issue 中记录运行命令、结果与提交号。生成的中间文件放在 `outputs/`，最终采用的图放在 `docs/figures/`，实验解释放在 `docs/analysis/`。

## 给 AI 的任务说明

以下提示词可直接作为各自会话的开头；后续以对应 GitHub Issue 的验收条件为准。

### 1 号仓库负责人

```text
先阅读 README.md、AGENTS.md、GLOSSARY.md、docs/architecture.md 和 docs/ 中的作业要求。我是 1 号同学，负责公共代码框架，不负责实现正式回归算法。请按架构文档实现数据解析与校验、固定划分、统一训练评估、图表与结果保存、命令行运行约定和接口测试。用测试中的简易 fit/predict 替身验证公共流程，不把它当成作业模型。保持 test.csv 仅供最终评估。完成后给出实际安装、测试和运行命令，并更新 README.md 的项目状态。
```

### 2 号线性路线

```text
先阅读 README.md、GLOSSARY.md、docs/architecture.md、作业要求和我认领的 GitHub Issue。我是 2 号同学，负责 experiments/linear/ 与 docs/analysis/linear.md。请在公共框架上实现一种线性或正则化回归算法，说明原理与关键参数，比较默认方案和优化方案；必要的标准化放入只在训练数据上拟合的 Pipeline。使用统一验证划分和 MAE、RMSE、R²，生成真实值与预测值对比图并解释结果。不要读取 test.csv。完成后运行相关检查，在 Issue 中记录命令、指标与提交号。
```

### 3 号树模型路线

```text
先阅读 README.md、GLOSSARY.md、docs/architecture.md、作业要求和我认领的 GitHub Issue。我是 3 号同学，负责 experiments/tree/ 与 docs/analysis/tree.md。请在公共框架上实现一种树或集成回归算法，说明原理与关键参数，比较默认方案和优化方案。使用统一验证划分和 MAE、RMSE、R²，生成真实值与预测值对比图并解释结果。不要读取 test.csv。完成后运行相关检查，在 Issue 中记录命令、指标与提交号。
```

### 4 号拓展路线

```text
先阅读 README.md、GLOSSARY.md、docs/architecture.md、作业要求，以及 2、3 号同学已完成的模型文件和我认领的 GitHub Issue。我是 4 号同学，负责 experiments/robustness/ 与 docs/analysis/robustness.md。请复用两位同学选定的模型：样本量实验只改变训练子集，噪声实验只扰动验证输入的副本。记录设置、随机种子、MAE、RMSE、R² 及趋势图，并解释泛化或鲁棒性的差异。实验只用 raw.csv，不读取或改动 test.csv。完成后运行相关检查，在 Issue 中记录命令、结果与提交号。
```
