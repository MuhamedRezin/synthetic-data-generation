"""
evaluator.py — Model Training & Evaluation (Before vs After Augmentation).

Trains Random Forest classifiers on three dataset versions:
1. Original imbalanced dataset
2. SMOTE-augmented dataset
3. CTGAN-augmented dataset

Compares performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def _get_plots_dir():
    """Get the outputs/plots directory path and ensure it exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(project_root, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training labels.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Random seed.

    Returns
    -------
    RandomForestClassifier
        Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and return metrics.

    Parameters
    ----------
    model : estimator
        Trained sklearn model.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        Test labels.
    model_name : str
        Name for display purposes.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS — {model_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")

    return metrics


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE oversampling to the training data.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training labels.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (X_resampled, y_resampled)
    """
    print("\nApplying SMOTE oversampling...")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {len(X_train)} samples (Fraud: {sum(y_train == 1)})")
    print(f"After SMOTE:  {len(X_resampled)} samples (Fraud: {sum(y_resampled == 1)})")

    return X_resampled, y_resampled


def compare_all_models(original_df, synthetic_fraud_df=None, test_size=0.2, random_state=42):
    """
    Train and evaluate Random Forest on three dataset versions:
    1. Original imbalanced
    2. SMOTE-augmented
    3. CTGAN-augmented

    Parameters
    ----------
    original_df : pd.DataFrame
        The original imbalanced dataset.
    synthetic_fraud_df : pd.DataFrame, optional
        Synthetic fraud samples from CTGAN. If None, only compares original and SMOTE.
    test_size : float
        Test set proportion.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with results for each model version.
    """
    # Common test set from original data (never touched)
    X = original_df.drop("Class", axis=1)
    y = original_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. Original imbalanced dataset
    print("\n" + "#" * 60)
    print("MODEL 1: Original Imbalanced Dataset")
    print("#" * 60)
    model_original = train_random_forest(X_train_scaled, y_train)
    results["original"] = evaluate_model(model_original, X_test_scaled, y_test, "Original (Imbalanced)")

    # 2. SMOTE-augmented dataset
    print("\n" + "#" * 60)
    print("MODEL 2: SMOTE-Augmented Dataset")
    print("#" * 60)
    X_smote, y_smote = apply_smote(X_train_scaled, y_train, random_state=random_state)
    model_smote = train_random_forest(X_smote, y_smote)
    results["smote"] = evaluate_model(model_smote, X_test_scaled, y_test, "SMOTE-Augmented")

    # 3. CTGAN-augmented dataset
    if synthetic_fraud_df is not None:
        print("\n" + "#" * 60)
        print("MODEL 3: CTGAN-Augmented Dataset")
        print("#" * 60)

        # Add synthetic fraud to training data only
        X_train_df = pd.DataFrame(X_train, columns=original_df.drop("Class", axis=1).columns)
        y_train_series = y_train.reset_index(drop=True)

        synthetic_X = synthetic_fraud_df.drop("Class", axis=1)
        synthetic_y = synthetic_fraud_df["Class"]

        X_ctgan = pd.concat([X_train_df, synthetic_X], ignore_index=True)
        y_ctgan = pd.concat([y_train_series, synthetic_y], ignore_index=True)

        X_ctgan_scaled = scaler.transform(X_ctgan)
        model_ctgan = train_random_forest(X_ctgan_scaled, y_ctgan)
        results["ctgan"] = evaluate_model(model_ctgan, X_test_scaled, y_test, "CTGAN-Augmented")

    # Store test labels for plotting
    results["y_test"] = y_test

    return results


def plot_confusion_matrices(results, save=True):
    """
    Plot confusion matrices for all evaluated models side by side.

    Parameters
    ----------
    results : dict
        Results dictionary from compare_all_models.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    model_keys = [k for k in results.keys() if k != "y_test"]
    n_models = len(model_keys)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for i, key in enumerate(model_keys):
        cm = results[key]["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Fraud", "Fraud"],
            yticklabels=["Non-Fraud", "Fraud"],
            ax=axes[i]
        )
        axes[i].set_title(f"{results[key]['model_name']}", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "confusion_matrices.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def generate_comparison_table(results):
    """
    Generate a formatted metrics comparison table.

    Parameters
    ----------
    results : dict
        Results dictionary from compare_all_models.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each model.
    """
    model_keys = [k for k in results.keys() if k != "y_test"]

    rows = []
    for key in model_keys:
        r = results[key]
        rows.append({
            "Model": r["model_name"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-Score": f"{r['f1_score']:.4f}",
            "ROC-AUC": f"{r['roc_auc']:.4f}",
        })

    comparison_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    return comparison_df


def plot_roc_curves(results, save=True):
    """
    Plot ROC curves for all evaluated models.

    Parameters
    ----------
    results : dict
        Results dictionary from compare_all_models.
    save : bool
        Whether to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"original": "#3498db", "smote": "#e67e22", "ctgan": "#2ecc71"}
    y_test = results["y_test"]

    model_keys = [k for k in results.keys() if k != "y_test"]

    for key in model_keys:
        y_proba = results[key]["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = results[key]["roc_auc"]
        color = colors.get(key, "#95a5a6")
        ax.plot(fpr, tpr, label=f"{results[key]['model_name']} (AUC = {auc:.4f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = os.path.join(_get_plots_dir(), "roc_curves.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
