"""
eda.py — Exploratory Data Analysis & Visualizations.

Provides functions to visualize class distribution, feature distributions,
correlation heatmaps, and summary statistics for the credit card fraud dataset.
All plots are saved to outputs/plots/ and returned for inline display.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set consistent plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def _get_plots_dir():
    """Get the outputs/plots directory path and ensure it exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(project_root, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_class_distribution(df, save=True):
    """
    Plot the class distribution as a pie chart and a count plot side by side.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a 'Class' column.
    save : bool
        Whether to save the plot to outputs/plots/.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    class_counts = df["Class"].value_counts()
    labels = ["Non-Fraud (Class 0)", "Fraud (Class 1)"]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0, 0.1)

    axes[0].pie(
        class_counts.values,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.3f%%",
        shadow=True,
        startangle=140,
        textprops={"fontsize": 11}
    )
    axes[0].set_title("Class Distribution (Pie Chart)", fontsize=14, fontweight="bold")

    # Count plot
    sns.countplot(x="Class", data=df, ax=axes[1], palette={0: "#2ecc71", 1: "#e74c3c"},
                  hue="Class", legend=False)
    axes[1].set_title("Class Distribution (Count Plot)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])

    # Add count annotations
    for i, count in enumerate(class_counts.values):
        axes[1].text(i, count + 1000, f"{count:,}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "class_distribution.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_feature_distributions(df, features=None, save=True):
    """
    Plot distributions (histograms with KDE) for selected features, comparing fraud vs non-fraud.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a 'Class' column.
    features : list of str, optional
        List of feature names to plot. Defaults to ['Amount', 'Time', 'V1', 'V2', 'V3', 'V4', 'V5'].
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if features is None:
        features = ["Amount", "Time", "V1", "V2", "V3", "V4", "V5"]

    # Filter to only features that exist in the dataframe
    features = [f for f in features if f in df.columns]

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    fraud = df[df["Class"] == 1]
    non_fraud = df[df["Class"] == 0]

    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(non_fraud[feature], bins=50, alpha=0.5, label="Non-Fraud", color="#2ecc71", density=True)
        ax.hist(fraud[feature], bins=50, alpha=0.7, label="Fraud", color="#e74c3c", density=True)
        ax.set_title(f"Distribution of {feature}", fontsize=13, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions: Fraud vs Non-Fraud", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "feature_distributions.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_correlation_heatmap(df, save=True):
    """
    Plot a correlation heatmap for the dataset features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(20, 16))

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        annot=False,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "correlation_heatmap.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def summary_statistics(df):
    """
    Compute and display summary statistics for fraud vs non-fraud transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a 'Class' column.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (fraud_stats, non_fraud_stats) — descriptive statistics DataFrames.
    """
    fraud = df[df["Class"] == 1]
    non_fraud = df[df["Class"] == 0]

    print("=" * 60)
    print("SUMMARY STATISTICS — FRAUD TRANSACTIONS (Class=1)")
    print("=" * 60)
    fraud_stats = fraud.describe().T
    print(fraud_stats[["count", "mean", "std", "min", "max"]].to_string())

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS — NON-FRAUD TRANSACTIONS (Class=0)")
    print("=" * 60)
    non_fraud_stats = non_fraud.describe().T
    print(non_fraud_stats[["count", "mean", "std", "min", "max"]].to_string())

    # Key comparisons
    print("\n" + "=" * 60)
    print("KEY COMPARISONS")
    print("=" * 60)
    print(f"Average fraud amount:     ${fraud['Amount'].mean():.2f}")
    print(f"Average non-fraud amount: ${non_fraud['Amount'].mean():.2f}")
    print(f"Max fraud amount:         ${fraud['Amount'].max():.2f}")
    print(f"Max non-fraud amount:     ${non_fraud['Amount'].max():.2f}")

    return fraud_stats, non_fraud_stats


def plot_amount_distribution(df, save=True):
    """
    Plot the transaction amount distribution for fraud vs non-fraud.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Amount' and 'Class' columns.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fraud = df[df["Class"] == 1]
    non_fraud = df[df["Class"] == 0]

    # Log-scale histogram
    axes[0].hist(non_fraud["Amount"], bins=100, alpha=0.5, label="Non-Fraud", color="#2ecc71", log=True)
    axes[0].hist(fraud["Amount"], bins=100, alpha=0.7, label="Fraud", color="#e74c3c", log=True)
    axes[0].set_title("Transaction Amount Distribution (Log Scale)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Count (Log)")
    axes[0].legend()

    # Box plot
    amount_data = pd.concat([non_fraud["Amount"].head(5000), fraud["Amount"]]).values
    class_data = ["Non-Fraud"] * min(5000, len(non_fraud)) + ["Fraud"] * len(fraud)
    
    df_plot = pd.DataFrame({
        "Amount": amount_data,
        "Class": class_data
    })
    sns.boxplot(x="Class", y="Amount", data=df_plot, ax=axes[1],
                palette={"Non-Fraud": "#2ecc71", "Fraud": "#e74c3c"})
    axes[1].set_title("Transaction Amount Box Plot", fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "amount_distribution.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
