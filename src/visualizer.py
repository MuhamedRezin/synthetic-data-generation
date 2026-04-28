"""
visualizer.py — Distribution Plots, Correlation Comparisons, and Metrics Visualization.

Provides functions to visually compare real vs synthetic data quality
and display model performance metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set consistent plot style
sns.set_style("whitegrid")


def _get_plots_dir():
    """Get the outputs/plots directory path and ensure it exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(project_root, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_real_vs_synthetic(real_df, synthetic_df, features=None, save=True):
    """
    Plot KDE overlay comparisons of real vs synthetic data for selected features.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real fraud samples.
    synthetic_df : pd.DataFrame
        Synthetic fraud samples.
    features : list of str, optional
        Features to plot. Defaults to Amount + V1-V5.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if features is None:
        features = ["Amount", "V1", "V2", "V3", "V4", "V5"]

    # Filter to available features
    features = [f for f in features if f in real_df.columns and f in synthetic_df.columns]

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(features):
        ax = axes[i]

        # KDE plots
        real_data = real_df[feature].dropna()
        synthetic_data = synthetic_df[feature].dropna()

        if len(real_data) > 0:
            real_data.plot.kde(ax=ax, label="Real", color="#3498db", linewidth=2)
        if len(synthetic_data) > 0:
            synthetic_data.plot.kde(ax=ax, label="Synthetic", color="#e74c3c", linewidth=2, linestyle="--")

        ax.set_title(f"{feature}: Real vs Synthetic", fontsize=13, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Real vs Synthetic Data — Distribution Comparison (KDE)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "real_vs_synthetic_kde.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_violin_comparison(real_df, synthetic_df, features=None, save=True):
    """
    Plot violin plots comparing real vs synthetic data distributions.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real fraud samples.
    synthetic_df : pd.DataFrame
        Synthetic fraud samples.
    features : list of str, optional
        Features to plot.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if features is None:
        features = ["Amount", "V1", "V2", "V3", "V4", "V5"]

    features = [f for f in features if f in real_df.columns and f in synthetic_df.columns]

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(features):
        ax = axes[i]

        # Create combined DataFrame for violin plot
        real_vals = real_df[feature].dropna().values
        synth_vals = synthetic_df[feature].dropna().values

        combined = pd.DataFrame({
            "Value": np.concatenate([real_vals, synth_vals]),
            "Type": (["Real"] * len(real_vals)) + (["Synthetic"] * len(synth_vals))
        })

        sns.violinplot(
            x="Type", y="Value", data=combined, ax=ax,
            palette={"Real": "#3498db", "Synthetic": "#e74c3c"},
            inner="box", cut=0
        )
        ax.set_title(f"{feature}", fontsize=13, fontweight="bold")
        ax.set_xlabel("")

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Real vs Synthetic Data — Violin Plot Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "real_vs_synthetic_violin.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_correlation_comparison(real_df, synthetic_df, save=True):
    """
    Plot correlation heatmaps for real and synthetic data side by side.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real fraud samples.
    synthetic_df : pd.DataFrame
        Synthetic fraud samples.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Select common numeric columns
    common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
    real_corr = real_df[common_cols].corr()
    synth_corr = synthetic_df[common_cols].corr()

    # Real data correlation
    mask1 = np.triu(np.ones_like(real_corr, dtype=bool))
    sns.heatmap(real_corr, mask=mask1, cmap="RdBu_r", center=0,
                ax=axes[0], cbar_kws={"shrink": 0.8}, square=True)
    axes[0].set_title("Real Fraud Data — Correlation", fontsize=14, fontweight="bold")

    # Synthetic data correlation
    mask2 = np.triu(np.ones_like(synth_corr, dtype=bool))
    sns.heatmap(synth_corr, mask=mask2, cmap="RdBu_r", center=0,
                ax=axes[1], cbar_kws={"shrink": 0.8}, square=True)
    axes[1].set_title("Synthetic Fraud Data — Correlation", fontsize=14, fontweight="bold")

    plt.suptitle("Correlation Structure Comparison: Real vs Synthetic",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "correlation_comparison.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_metrics_comparison(results, save=True):
    """
    Plot a grouped bar chart comparing metrics across all models.

    Parameters
    ----------
    results : dict
        Results dictionary from evaluator.compare_all_models.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    model_keys = [k for k in results.keys() if k != "y_test"]

    metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(display_names))
    width = 0.25
    colors = {"original": "#3498db", "smote": "#e67e22", "ctgan": "#2ecc71"}

    for i, key in enumerate(model_keys):
        values = [results[key][m] for m in metrics_names]
        color = colors.get(key, "#95a5a6")
        bars = ax.bar(x + i * width, values, width, label=results[key]["model_name"],
                      color=color, edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width * (len(model_keys) - 1) / 2)
    ax.set_xticklabels(display_names, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "metrics_comparison.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_class_distribution_before_after(original_df, balanced_df, save=True):
    """
    Plot class distribution before and after synthetic data augmentation.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original imbalanced dataset.
    balanced_df : pd.DataFrame
        Balanced dataset after CTGAN augmentation.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before
    orig_counts = original_df["Class"].value_counts()
    axes[0].bar(["Non-Fraud", "Fraud"], orig_counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[0].set_title("Before Augmentation", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(orig_counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    # After
    bal_counts = balanced_df["Class"].value_counts()
    axes[1].bar(["Non-Fraud", "Fraud"], bal_counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[1].set_title("After CTGAN Augmentation", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(bal_counts.values):
        axes[1].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    plt.suptitle("Class Distribution: Before vs After", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "before_after_distribution.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
