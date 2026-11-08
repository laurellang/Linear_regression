# 实验结果

按模型保存需要共享的实验结果，供小组复核和报告汇总。

| 位置 | 内容 |
| --- | --- |
| [linear/outcome.md](linear/outcome.md) | 线性回归、L2 正则化的指标和结果说明 |
| `linear/*.png` | 线性模型预测图与性能比较图 |
| `tree/*.json`、`tree/*.csv`、`tree/*.png` | 树模型指标、逐样本预测及图表 |

本次共享树模型的指标、预测数据与图表。`tree_submission.zip` 是本地源码与文档交付归档，本次不上传；后续运行产生的临时结果也不默认上传。

以上结果均来自开发数据的验证或训练内交叉验证，独立 `test.csv` 尚未评估。完整方法分析见 [线性路线](../docs/analysis/linear.md) 和 [树模型路线](../docs/analysis/tree.md)；报告采用的精选图仍放在 `docs/figures/`。
