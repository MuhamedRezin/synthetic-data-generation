"""
streamlit_app.py — Interactive Streamlit Demo for Synthetic Data Generation.

Self-contained Streamlit app that demonstrates the full pipeline:
- Upload or use default Credit Card Fraud dataset
- Visualize class imbalance
- Generate synthetic fraud samples using CTGAN
- Evaluate data quality
- Compare ML model performance before/after augmentation
- Download the balanced dataset
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("Starting app imports...")
from src.data_loader import load_data, preprocess_data, split_by_class, get_dataset_info
print("Imported data_loader")
from src.eda import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_amount_distribution,
)
print("Imported eda")
from src.synthesizer import (
    train_ctgan,
    generate_samples,
    save_synthetic_data,
    create_balanced_dataset,
    calculate_target_count,
)
print("Imported synthesizer")
from src.evaluator import (
    compare_all_models,
    plot_confusion_matrices,
    generate_comparison_table,
    plot_roc_curves,
)
print("Imported evaluator")
from src.visualizer import (
    plot_real_vs_synthetic,
    plot_violin_comparison,
    plot_correlation_comparison,
    plot_metrics_comparison,
    plot_class_distribution_before_after,
)
print("All imports successful!")


# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Synthetic Data Generator — IBM SkillsBuild",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🔬 Synthetic Data Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Generate synthetic data for imbalanced datasets using CTGAN — IBM SkillsBuild Project</div>',
    unsafe_allow_html=True,
)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()

    # Dataset upload
    st.subheader("📂 Dataset")
    data_source = st.radio(
        "Choose data source:",
        ["Use default (Credit Card Fraud)", "Upload CSV"],
        index=0,
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        target_column = st.text_input("Target column name", value="Class")
        minority_class = st.number_input("Minority class label", value=1, step=1)
    else:
        target_column = "Class"
        minority_class = 1

    st.divider()

    # Synthesis settings
    st.subheader("🧬 Synthesis Settings")
    num_synthetic = st.number_input(
        "Number of synthetic samples",
        min_value=100,
        max_value=500000,
        value=10000,
        step=1000,
        help="Number of synthetic minority class samples to generate.",
    )

    auto_balance = st.checkbox(
        "Auto-balance (match majority class count)",
        value=True,
        help="If checked, generates enough samples to fully balance the dataset.",
    )

    epochs = st.slider(
        "CTGAN training epochs",
        min_value=50,
        max_value=500,
        value=300,
        step=50,
        help="More epochs = better quality but longer training time.",
    )

    st.divider()
    st.markdown("**Built for IBM SkillsBuild**")
    st.markdown("Using CTGAN via SDV")


# ─── Load Data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset(uploaded_file=None):
    """Load dataset from file upload or default path."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        default_path = os.path.join(project_root, "data", "creditcard.csv")
        if not os.path.exists(default_path):
            return None
        df = pd.read_csv(default_path)
    
    # Preprocess inside the cached function to avoid 15-second delays on every run
    df = preprocess_data(df)
    return df


# ─── Main Content Tabs ─────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "🧬 Generate Synthetic Data",
    "📈 Quality Evaluation",
    "🏆 Model Comparison",
])


# Load the dataset
df = load_dataset(uploaded_file)

if df is None:
    st.error(
        "⚠️ Dataset not found! Please either:\n"
        "1. Place `creditcard.csv` in the `data/` directory, or\n"
        "2. Upload a CSV file using the sidebar.\n\n"
        "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("📊 Dataset Overview")

    # Key metrics
    info = get_dataset_info(df)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", f"{info['total_samples']:,}")
    with col2:
        st.metric("Total Features", info["total_features"])
    with col3:
        st.metric("Fraud Samples", f"{info['fraud_count']:,}")
    with col4:
        st.metric("Fraud Percentage", f"{info['fraud_percentage']:.3f}%")

    st.divider()

    # Class distribution plots
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Class Distribution")
        fig_class = plot_class_distribution(df, save=False)
        st.pyplot(fig_class)
        plt.close(fig_class)

    with col_right:
        st.subheader("Transaction Amount Distribution")
        fig_amount = plot_amount_distribution(df, save=False)
        st.pyplot(fig_amount)
        plt.close(fig_amount)

    st.divider()

    # Feature distributions
    st.subheader("Feature Distributions (Fraud vs Non-Fraud)")
    fig_features = plot_feature_distributions(df, save=False)
    st.pyplot(fig_features)
    plt.close(fig_features)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    with st.expander("Show Correlation Heatmap (large plot)"):
        fig_corr = plot_correlation_heatmap(df, save=False)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    # Summary statistics
    st.subheader("Summary Statistics")
    col_fraud, col_nonfraud = st.columns(2)
    with col_fraud:
        st.markdown("**Fraud Transactions (Class=1)**")
        st.dataframe(df[df[target_column] == minority_class].describe().T)
    with col_nonfraud:
        st.markdown("**Non-Fraud Transactions (Class=0)**")
        st.dataframe(df[df[target_column] != minority_class].describe().T)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Generate Synthetic Data
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🧬 Generate Synthetic Data with CTGAN")

    # Split data
    non_fraud_df, fraud_df = split_by_class(df)

    st.info(f"""
    **Current dataset statistics:**
    - Non-Fraud samples: **{len(non_fraud_df):,}**
    - Fraud samples: **{len(fraud_df):,}**
    - Imbalance ratio: **1:{len(non_fraud_df) // max(len(fraud_df), 1)}**
    """)

    if auto_balance:
        target_count = calculate_target_count(df)
        st.success(f"Auto-balance mode: Will generate **{target_count:,}** synthetic fraud samples.")
    else:
        target_count = num_synthetic
        st.info(f"Manual mode: Will generate **{target_count:,}** synthetic fraud samples.")

    st.divider()

    # Training button
    if st.button("🚀 Start CTGAN Training & Generation", type="primary", use_container_width=True):
        with st.spinner("Training CTGAN model... This may take several minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Train CTGAN
            status_text.text("Step 1/3: Training CTGAN synthesizer...")
            progress_bar.progress(10)

            synthesizer = train_ctgan(fraud_df, epochs=epochs, verbose=True)
            progress_bar.progress(60)

            # Generate samples
            status_text.text("Step 2/3: Generating synthetic samples...")
            synthetic_df = generate_samples(synthesizer, target_count)
            progress_bar.progress(80)

            # Create balanced dataset
            status_text.text("Step 3/3: Creating balanced dataset...")
            balanced_df = create_balanced_dataset(df, synthetic_df)
            progress_bar.progress(100)

            status_text.text("✅ Complete!")

        # Store results in session state
        st.session_state["synthetic_df"] = synthetic_df
        st.session_state["balanced_df"] = balanced_df
        st.session_state["fraud_df"] = fraud_df

        st.success(f"✅ Generated **{len(synthetic_df):,}** synthetic fraud samples!")

        # Show before/after
        fig_ba = plot_class_distribution_before_after(df, balanced_df, save=True)
        st.pyplot(fig_ba)
        plt.close(fig_ba)

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_synthetic = synthetic_df.to_csv(index=False)
            st.download_button(
                "📥 Download Synthetic Samples",
                csv_synthetic,
                "synthetic_fraud_samples.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_dl2:
            csv_balanced = balanced_df.to_csv(index=False)
            st.download_button(
                "📥 Download Balanced Dataset",
                csv_balanced,
                "balanced_dataset.csv",
                "text/csv",
                use_container_width=True,
            )

    elif "synthetic_df" in st.session_state:
        st.success("✅ Synthetic data already generated! View results below.")

        fig_ba = plot_class_distribution_before_after(df, st.session_state["balanced_df"], save=False)
        st.pyplot(fig_ba)
        plt.close(fig_ba)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_synthetic = st.session_state["synthetic_df"].to_csv(index=False)
            st.download_button(
                "📥 Download Synthetic Samples",
                csv_synthetic,
                "synthetic_fraud_samples.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_dl2:
            csv_balanced = st.session_state["balanced_df"].to_csv(index=False)
            st.download_button(
                "📥 Download Balanced Dataset",
                csv_balanced,
                "balanced_dataset.csv",
                "text/csv",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Quality Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📈 Synthetic Data Quality Evaluation")

    if "synthetic_df" not in st.session_state:
        st.warning("⚠️ Please generate synthetic data first (Tab 2).")
    else:
        synthetic_df = st.session_state["synthetic_df"]
        fraud_df = st.session_state["fraud_df"]

        # SDMetrics quality evaluation
        st.subheader("📊 SDMetrics Quality Scores")
        try:
            from sdmetrics.reports.single_table import QualityReport

            with st.spinner("Computing quality metrics..."):
                from sdv.metadata import SingleTableMetadata

                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(fraud_df)

                report = QualityReport()
                report.generate(fraud_df, synthetic_df, metadata.to_dict())

                quality_score = report.get_score()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Quality Score", f"{quality_score:.4f}")
                with col2:
                    column_shapes = report.get_properties()
                    if len(column_shapes) > 0:
                        shape_score = column_shapes.iloc[0]["Score"] if "Score" in column_shapes.columns else "N/A"
                        st.metric("Column Shapes", f"{shape_score}")
                with col3:
                    if len(column_shapes) > 1:
                        pair_score = column_shapes.iloc[1]["Score"] if "Score" in column_shapes.columns else "N/A"
                        st.metric("Column Pair Trends", f"{pair_score}")

        except Exception as e:
            st.warning(f"SDMetrics evaluation encountered an issue: {e}")
            st.info("Continuing with visual evaluation...")

        st.divider()

        # KDE comparison plots
        st.subheader("🔍 Distribution Comparison (KDE)")
        fig_kde = plot_real_vs_synthetic(fraud_df, synthetic_df, save=True)
        st.pyplot(fig_kde)
        plt.close(fig_kde)

        st.divider()

        # Violin plots
        st.subheader("🎻 Violin Plot Comparison")
        fig_violin = plot_violin_comparison(fraud_df, synthetic_df, save=True)
        st.pyplot(fig_violin)
        plt.close(fig_violin)

        st.divider()

        # Correlation comparison
        st.subheader("🔗 Correlation Structure Comparison")
        fig_corr_comp = plot_correlation_comparison(fraud_df, synthetic_df, save=True)
        st.pyplot(fig_corr_comp)
        plt.close(fig_corr_comp)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("🏆 ML Model Comparison")

    if "synthetic_df" not in st.session_state:
        st.warning("⚠️ Please generate synthetic data first (Tab 2).")
    else:
        synthetic_df = st.session_state["synthetic_df"]

        if st.button("🏋️ Train & Compare Models", type="primary", use_container_width=True):
            with st.spinner("Training models on all three dataset versions... This may take a minute."):
                results = compare_all_models(df, synthetic_df)
                st.session_state["results"] = results

        if "results" in st.session_state:
            results = st.session_state["results"]

            # Comparison table
            st.subheader("📋 Metrics Comparison Table")
            comparison_df = generate_comparison_table(results)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.divider()

            # Metrics bar chart
            st.subheader("📊 Performance Metrics")
            fig_metrics = plot_metrics_comparison(results, save=True)
            st.pyplot(fig_metrics)
            plt.close(fig_metrics)

            st.divider()

            # Confusion matrices
            st.subheader("🔢 Confusion Matrices")
            fig_cm = plot_confusion_matrices(results, save=True)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.divider()

            # ROC curves
            st.subheader("📈 ROC Curves")
            fig_roc = plot_roc_curves(results, save=True)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

            st.divider()

            # Key findings
            st.subheader("🔑 Key Findings")

            model_keys = [k for k in results.keys() if k != "y_test"]
            best_recall_key = max(model_keys, key=lambda k: results[k]["recall"])
            best_f1_key = max(model_keys, key=lambda k: results[k]["f1_score"])
            best_auc_key = max(model_keys, key=lambda k: results[k]["roc_auc"])

            st.markdown(f"""
            | Metric | Best Model | Score |
            |--------|-----------|-------|
            | **Best Recall** | {results[best_recall_key]['model_name']} | {results[best_recall_key]['recall']:.4f} |
            | **Best F1-Score** | {results[best_f1_key]['model_name']} | {results[best_f1_key]['f1_score']:.4f} |
            | **Best ROC-AUC** | {results[best_auc_key]['model_name']} | {results[best_auc_key]['roc_auc']:.4f} |
            """)
