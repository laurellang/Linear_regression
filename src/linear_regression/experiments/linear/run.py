import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from pathlib import Path
import matplotlib.pyplot as plt


from linear_regression.data import (
    load_dataset,
    make_validation_split,
)
from linear_regression.experiments.linear.model import (
    ManualScaler,
    ManualLinearRegression,
)


LEARNING_RATE = 0.01
N_ITERATIONS = 2000
L2_STRENGTH = 100


def build_model(variant="baseline"):
    """每次返回一个独立、未训练的模型。"""
    if variant == "baseline":
        estimator = ManualLinearRegression(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
            l2_strength=0.0,
        )
    elif variant == "sklearn":
        estimator = LinearRegression()
    elif variant == "optimized":

        estimator = ManualLinearRegression(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
            l2_strength=L2_STRENGTH,
        )
    else:
        raise ValueError(f"不支持的模型方案：{variant}")

    return Pipeline([
        ("scale", ManualScaler()),
        ("model", estimator),
    ])


def calculate_metrics(y_true, y_pred):
    """临时用于本地比较，正式交付时接入公共评价工具。"""
    if y_pred.shape != y_true.shape:
        raise ValueError("预测值与真实值形状不一致")

    if not np.isfinite(y_pred).all():
        raise ValueError("预测出现非有限值，请检查训练过程")

    errors = y_pred - y_true

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    total_sum = np.sum((y_true - y_true.mean()) ** 2)
    if total_sum == 0:
        raise ValueError("目标为常数，当前公式无法计算 R²")

    r2 = 1 - np.sum(errors ** 2) / total_sum

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def plot_predictions(results, y_true):
    output_dir = Path("outputs/linear")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("baseline", "Manual linear regression"),
        ("optimized", "Manual L2 linear regression"),
    ]

    # 两张图使用相同的坐标范围
    all_values = np.concatenate([
        y_true,
        results["baseline"]["predictions"],
        results["optimized"]["predictions"],
    ])

    lower = float(all_values.min())
    upper = float(all_values.max())
    padding = max((upper - lower) * 0.05, 1e-6)

    lower -= padding
    upper += padding

    for variant, title in variants:
        result = results[variant]
        predictions = result["predictions"]
        metrics = result["metrics"]

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(
            y_true,
            predictions,
            s=10,
            alpha=0.25,
            color="#2878B5",
            edgecolors="none",
            label="Validation samples",
        )

        # 理想预测参考线
        ax.plot(
            [lower, upper],
            [lower, upper],
            color="#D95319",
            linestyle="--",
            linewidth=1.5,
            label="Perfect prediction",
        )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("True k-inf")
        ax.set_ylabel("Predicted k-inf")

        if variant == "optimized":
            title += f" (lambda={L2_STRENGTH:g})"
        else:
            title += " (lambda=0)"

        ax.set_title(title)

        metrics_text = (
            f"MAE = {metrics['mae']:.8f}\n"
            f"RMSE = {metrics['rmse']:.8f}\n"
            f"$R^2$ = {metrics['r2']:.8f}"
        )

        ax.text(
            0.04,
            0.96,
            metrics_text,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.9,
            },
        )

        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        # 直接保存在同一个目录，用文件名区分模型
        output_path = (
            output_dir / f"{variant}_prediction_vs_true.png"
        )

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"图片已保存：{output_path.resolve()}")


def plot_comparison(results):
    """Compare the same validation samples without truncating bar axes."""
    variants = ["baseline", "sklearn", "optimized"]
    ids = results["baseline"]["evaluation_ids"]
    if any(results[v]["evaluation_ids"] != ids for v in variants):
        raise ValueError("Models must use identical evaluation rows")
    labels = ["Manual OLS", "sklearn OLS", "Manual L2"]
    colors = ["#2878B5", "#8A9BA8", "#E69F00"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for ax, key, title in zip(axes, ["mae", "rmse", "r2"],
                              ["MAE (lower is better)", "RMSE (lower is better)",
                               "R-squared (higher is better)"]):
        values = [results[v]["metrics"][key] for v in variants]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.bar_label(bars, labels=[f"{x:.8f}" for x in values], padding=5, fontsize=9)
        ax.set_ylim(min(0, min(values)*1.15), max(values)*1.18)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Validation performance | L2 lambda = 100 | n = 4800", fontsize=13)
    output = Path("outputs/linear/model_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Comparison saved: {output.resolve()}")


def main():
    # 1. 所有模型使用同一个训练/验证划分
    data = load_dataset("dataset_UCI/raw.csv")
    split = make_validation_split(data)

    print("训练输入：", split.X_train.shape)
    print("训练目标：", split.y_train.shape)
    print("验证输入：", split.X_eval.shape)
    print("验证目标：", split.y_eval.shape)
    print("手写模型学习率：", LEARNING_RATE)
    print("手写模型迭代次数：", N_ITERATIONS)
    print("选定 L2 强度：", L2_STRENGTH)

    experiments = [
        ("baseline", "手写普通线性回归"),
        ("sklearn", "sklearn 普通线性回归"),
        ("optimized", "手写 L2 线性回归（λ=100）"),
    ]

    results = {}

    # 2. 分别训练并评价
    for variant, name in experiments:
        print(f"\n{'=' * 60}")
        print(f"开始训练：{name}")

        model = build_model(variant)

        start_time = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        fit_seconds = time.perf_counter() - start_time

        # Pipeline 自动使用各自从训练集学到的标准化参数
        train_predictions = model.predict(split.X_train)
        predictions = model.predict(split.X_eval)

        train_metrics = calculate_metrics(
            split.y_train,
            train_predictions,
        )
        validation_metrics = calculate_metrics(
            split.y_eval,
            predictions,
        )

        # 统一计算不含正则化惩罚的训练平方误差
        train_cost = np.mean(
            (train_predictions - split.y_train) ** 2
        ) / 2

        results[variant] = {
            "name": name,
            "fit_seconds": fit_seconds,
            "train_cost": train_cost,
            "train_rmse": train_metrics["rmse"],
            "metrics": validation_metrics,
            "predictions": predictions,
            "evaluation_ids": split.evaluation_ids,
        }

        print(f"\n{name}：训练完成")
        print(f"训练耗时：{fit_seconds:.2f} 秒")
        print(f"训练平方误差 / 2（不含惩罚）：{train_cost:.10f}")
        print("预测形状：", predictions.shape)
        print("前5个预测值：", predictions[:5])
        print("前5个真实值：", split.y_eval[:5])
        print(f"验证 MAE： {validation_metrics['mae']:.8f}")
        print(f"验证 RMSE：{validation_metrics['rmse']:.8f}")
        print(f"验证 R2：  {validation_metrics['r2']:.8f}")

    # 3. 汇总三种模型
    print(f"\n{'=' * 60}")
    print("三种模型比较：")

    for variant, _ in experiments:
        result = results[variant]
        metrics = result["metrics"]

        print(f"\n{result['name']}")
        print(f"  训练 RMSE：{result['train_rmse']:.8f}")
        print(f"  验证 MAE： {metrics['mae']:.8f}")
        print(f"  验证 RMSE：{metrics['rmse']:.8f}")
        print(f"  验证 R2：  {metrics['r2']:.8f}")
        print(f"  训练耗时： {result['fit_seconds']:.2f} 秒")

    # 4. 检查手写普通回归与 sklearn 的预测差异
    prediction_difference = (
        results["baseline"]["predictions"]
        - results["sklearn"]["predictions"]
    )

    print("\n手写普通回归与 sklearn 的验证预测差异：")
    print(
        "平均绝对差："
        f"{np.mean(np.abs(prediction_difference)):.10f}"
    )
    print(
        "最大绝对差："
        f"{np.max(np.abs(prediction_difference)):.10f}"
    )

    # 5. 检查当前 L2 候选是否改善验证 RMSE
    baseline_rmse = results["baseline"]["metrics"]["rmse"]
    l2_rmse = results["optimized"]["metrics"]["rmse"]
    improvement = baseline_rmse - l2_rmse

    print("\nL2 候选与手写普通回归比较：")
    print(f"验证 RMSE 降低量：{improvement:.10f}")
    print("正数表示改善，负数表示变差。")
    print("λ=100 按预定五个候选中的验证 RMSE 最小选定。")


    plot_predictions(results, split.y_eval)
    plot_comparison(results)


if __name__ == "__main__":
    main()