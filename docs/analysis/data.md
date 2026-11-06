# 数据准备与预处理

本部分为线性路线、树模型路线和鲁棒性实验提供同一套数据读取与训练/验证划分。实测开发数据完整且无重复输入，因此保留全部样本及 39 个特征，不进行缺失填补、类别编码、异常值删除或全局标准化。

## 数据来源与任务

数据为 UCI Lattice-physics（PWR fuel assembly neutronics simulation results，ID 1091），由 MCNP6 模拟得到。39 个输入表示燃料组件八分之一对称区域内各燃料棒的 U-235 富集度，单位为质量百分比。UCI 说明富集度在 0.7–5.0 范围均匀采样，输出包括无限增殖因子 k-inf 和棒功率峰值因子 PPPF。本项目选择预测连续变量 k-inf。

来源：[UCI 数据集说明](https://archive.ics.uci.edu/dataset/1091/lattice-physics%2B%28pwr%2Bfuel%2Bassembly%2Bneutronics%2Bsimulation%2Bresults%29)。引用：Huu Tiep, N. (2024). Lattice-physics (PWR fuel assembly neutronics simulation results) [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5BK64，许可 CC BY 4.0。

原始文件扩展名为 csv，实际内容是无表头、以空白字符分隔的 41 列数值，不能按逗号分隔读取。沿用项目列约定：第 1 列为 k-inf，第 2 列为 PPPF，第 3–41 列为输入。**UCI 页面没有明确给出文件列顺序，这仍是待来源确认的项目假设；数值范围吻合不能代替正式确认。** PPPF 是另一个模拟输出，不属于预测时的 39 个输入，将其排除可防止引入额外输出信息。

## 开发数据检查结果

以下结果由仓库中的 raw.csv 实际计算，不使用独立 test.csv 的分布或标签。原始文件 SHA-256 为 `377fad93fefe79483d45f194c57d886f19a9d28ae383a2a838e413a63fdaa253`。

| 检查项 | 结果 | 处理依据 |
| --- | --- | --- |
| 样本与列数 | 24,000 行，41 列 | 取 39 个输入和 1 个目标 |
| 非数值、缺列及非有限值 | 严格解析通过，均未发现 | 无需填补；遇到坏数据直接报告路径和行号 |
| 重复建模行及重复输入 | 均为 0 | 无需去重；随机划分不存在重复输入跨集合的问题 |
| 常数特征 | 0 列 | 保留全部 39 列 |
| 全部输入范围 | 0.700003153–4.999999766 | 均处于 UCI 给出的范围内，无需裁剪 |
| 逐特征 1.5 IQR 检查 | 标记单元格 0 个 | 描述性检查，无删行操作 |
| 用训练集 IQR 阈值检查验证输入 | 标记单元格 0 个 | 阈值仅从训练部分计算，无删行操作 |
| k-inf 最小值与最大值 | 1.24636、1.38558 | 保留原始目标尺度 |
| k-inf 四分位数 | Q1 1.31365，中位数 1.326085，Q3 1.3376825 | 描述目标分布 |

IQR 指 Q3−Q1，检查区间为 [Q1−1.5×IQR, Q3+1.5×IQR]。这里的 IQR 检查针对输入特征，不是目标值；统计规则不能认定物理错误。即使后续出现标记，也应结合来源调查，不能仅凭目标值极端就删除样本。全开发集描述统计用于报告；当前没有根据这些统计量拟合变换、选特征或删样本。

脚本输出 `outputs/data/audit.json`、各特征的 `feature_statistics.csv`、目标分布图 `target_distribution.svg` 以及原始行号划分清单 `split.csv`。标准差采用总体标准差（ddof=0）。这些生成文件位于被忽略的 outputs 目录，可重现生成。

## 处理方案与防止泄漏

1. 保持原始数据文件不变。读取时检查每行恰好 41 列、全部可转为有限实数；缺失或损坏时明确报错，而不是静默丢行。
2. 提取后 39 列作为 X，第 1 列作为 y；不使用 PPPF。所有输入均为实数，无需独热编码。
3. 使用种子 42 的固定随机排列，以 20% 为验证集，得到训练集 19,200 行、验证集 4,800 行。保留文件名及从 1 开始的原始行号供不同组员核对。当前没有重复输入，因此采用普通随机划分；若换用存在重复或分组相关的数据，必须重新审视划分策略。
4. 公共数据模块不进行标准化。线性或正则化模型负责人可使用 `Pipeline([('scale', StandardScaler()), ('model', Ridge(...))])`，调用 `fit(X_train, y_train)` 时仅从训练集学习均值和标准差，验证时只调用 `predict(X_eval)`。树模型通常可直接使用原始特征。是否缩放及其理由由各模型负责人记录。
5. 交叉验证调参时，应把整个 Pipeline 传入搜索器，在每个训练折内部拟合标准化；不能先对完整开发集调用 `fit_transform` 再交叉验证。固定验证集若用于选参，其结果属于模型选择表现，最终泛化表现以独立测试集为准。
6. test.csv 保留给参数冻结后的最终评估。本次统计脚本不读取它。最终由 `make_final_split` 把完整 raw.csv 作为训练部分、独立文件作为评价部分，检查两者不存在相同输入后交给模型；该函数不拟合任何预处理。两文件有相同输入即报错，比只比较完整行更严格。

## 组员调用接口

```python
from linear_regression.data import load_dataset, make_validation_split

data = load_dataset('dataset_UCI/raw.csv')
split = make_validation_split(data)  # 保持默认种子与比例
model.fit(split.X_train, split.y_train)
prediction = model.predict(split.X_eval)
```

`RegressionDataset` 提供 `X`、`y` 和 `source`。`EvaluationSplit` 提供 `X_train`、`y_train`、`X_eval`、`y_eval`、`stage`、`evaluation_ids`、`train_ids`、`seed` 和 `target`。验证阶段 `stage='validation'`，最终阶段 `stage='test'`，目标固定为 `k-inf`。`evaluation_ids` 的顺序与 `y_eval`、预测值完全对应，标识形如 `raw.csv:123`。更换源文件内容时必须同时核对审计文件中的哈希。

## 本地运行与验证

当前仓库尚无 pyproject.toml，可从仓库根目录设置 PYTHONPATH 运行；本部分运行时只依赖 Python 3.10+ 和 NumPy，无需 pandas 或 scikit-learn。模型实验需要的依赖仍由各模型负责人及仓库维护者配置。PowerShell 示例：

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s src/tests -v
python -m linear_regression.experiments.data.run
# 可选：自定义输入与输出目录
python -m linear_regression.experiments.data.run --input dataset_UCI/raw.csv --output outputs/data
```

本次使用可用的 bundled Python 实测完成，3 项 unittest 全部通过，覆盖目标与 PPPF 排除、无效输入报错、划分可复现及完整互斥、行与标签对齐、最终输入重叠拒绝和返回数组隔离。开发数据审计命令成功输出上述统计与 19,200/4,800 划分。最终测试集评估未运行。

## 报告可用表述

本实验采用 UCI Lattice-physics 数据集，以 39 个燃料棒富集度特征预测 k-inf。按项目暂定列约定读取无表头空白分隔文件，排除另一个输出 PPPF。对 24,000 条开发样本的检查未发现缺失、非有限值、重复输入或常数特征；所有输入均位于来源说明的富集度范围内。因此保留全部样本和特征，不进行填补、编码或异常值删除。以随机种子 42 按 80%/20% 划分训练集与验证集，分别包含 19,200 和 4,800 条样本。公共数据阶段保留原始尺度；需要标准化的模型在 Pipeline 中仅使用训练数据拟合变换，交叉验证时在各训练折内重复拟合。独立 test.csv 保留到模型与参数确定后统一评估，不参与调参或特征选择。文件列顺序仍需进一步由来源确认。
