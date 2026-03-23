import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_vs_k(results_df, metrics=["RMSE", "MAE", "NDCG@k"], hue="Model"):
    """
    Plot multiple metrics vs. neighborhood size K in a row.

    Parameters:
        results_df (pd.DataFrame): Results with columns ['K', metrics, 'Model']
        metrics (list): List of metric column names to plot
        hue (str): Column to group lines by (e.g., 'Model', 'Similarity')
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5), sharex=True)

    for i, metric in enumerate(metrics):
        sns.lineplot(data=results_df, x="K", y=metric, hue=hue, style="Similarity", marker="o", ax=axes[i])
        axes[i].set_title(f"{metric} vs K")
        axes[i].set_xlabel("Neighborhood Size (K)")
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
        axes[i].legend().set_title(hue)

    plt.tight_layout()
    plt.show()


def plot_metrics_vs_similarity(results_df, metrics=["RMSE", "NDCG@k"], fixed_k=20, hue="Model"):
    """
    Plot metrics vs. similarity method for a fixed value of K.

    Parameters:
        results_df (pd.DataFrame): Evaluation results
        metrics (list): List of metric column names to plot
        fixed_k (int): Value of K to filter
        hue (str): Column to hue/group lines by
    """
    df_k = results_df[results_df["K"] == fixed_k]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        sns.barplot(data=df_k, x="Similarity", y=metric, hue=hue, ax=axes[i])
        axes[i].set_title(f"{metric} vs Similarity (K={fixed_k})")
        axes[i].grid(True, axis="y")

    plt.tight_layout()
    plt.show()


def plot_runtime_and_throughput(scalability_df):
    """
    Plot prediction time and throughput vs data size.

    Parameters:
        scalability_df (pd.DataFrame): Data with columns 'DataFraction', 'PredTime(s)', 'Throughput(preds/sec)'
    """
    # Runtime
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=scalability_df,
        x=scalability_df["DataFraction"] * 100,
        y="Prediction Time (s)",
        hue="ModelType",
        marker="o"
    )
    plt.title("Prediction Time vs Data Size")
    plt.xlabel("Data Size (% Users)")
    plt.ylabel("Prediction Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Throughput
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=scalability_df,
        x=scalability_df["DataFraction"] * 100,
        y="Throughput(preds/sec)",
        hue="ModelType",
        marker="o"
    )
    plt.title("Throughput vs Data Size")
    plt.xlabel("Data Size (% Users)")
    plt.ylabel("Predictions per Second")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
