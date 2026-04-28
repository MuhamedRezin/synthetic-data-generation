# Synthetic Data Generation for Imbalanced Datasets

**IBM SkillsBuild / IBM Academic Initiative Project**

## Overview

This project addresses the class imbalance problem in machine learning by using **CTGAN (Conditional Tabular GAN)** to generate synthetic minority class samples. We demonstrate the full pipeline on the **Credit Card Fraud Detection** dataset, where fraudulent transactions make up less than 0.2% of all records.

### Key Features
- 🔬 **CTGAN-based synthesis** using SDV (Synthetic Data Vault)
- 📊 **Comprehensive EDA** with publication-quality visualizations
- 📈 **Quality evaluation** using SDMetrics
- 🏆 **Model comparison** — Original vs SMOTE vs CTGAN augmentation
- 🌐 **Interactive Streamlit app** for end-to-end demonstration

## Tech Stack

| Library | Purpose |
|---------|---------|
| SDV (sdv) | Synthetic Data Vault — high-level synthesis API |
| CTGAN (ctgan) | Conditional Tabular GAN for tabular data |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models, metrics, preprocessing |
| Imbalanced-learn | SMOTE (baseline comparison) |
| Matplotlib/Seaborn | Visualization |
| SDMetrics | Synthetic data quality evaluation |
| Streamlit | Interactive web application |

## Project Structure

```
synthetic-data-ibm/
│
├── data/
│   └── creditcard.csv                  # Raw dataset (download from Kaggle)
│
├── notebooks/
│   └── synthetic_data_generation.ipynb # Main research notebook
│
├── src/
│   ├── __init__.py                     # Package init
│   ├── data_loader.py                  # Load and preprocess dataset
│   ├── eda.py                          # Exploratory data analysis & visualizations
│   ├── synthesizer.py                  # CTGAN model training and sample generation
│   ├── evaluator.py                    # Model training + evaluation (before vs after)
│   └── visualizer.py                   # Distribution plots, correlation heatmaps
│
├── app/
│   └── streamlit_app.py                # Interactive Streamlit demo
│
├── outputs/
│   ├── synthetic_fraud_samples.csv     # Generated synthetic minority class samples
│   ├── balanced_dataset.csv            # Final balanced dataset
│   └── plots/                          # All saved figures
│
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Place `creditcard.csv` in the `data/` directory

### 3. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/synthetic_data_generation.ipynb
```

### 4. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Pipeline Overview

### Phase 1 — Data Loading & EDA
- Load and preprocess the dataset
- Visualize class distribution, feature distributions, and correlations
- Generate summary statistics for fraud vs non-fraud transactions

### Phase 2 — Synthetic Data Generation
- Train CTGAN on minority class (fraud) samples only
- Generate synthetic fraud samples to balance the dataset
- Save synthetic and balanced datasets

### Phase 3 — Quality Evaluation
- Compute SDMetrics quality scores (Column Shape, Column Pair Trends)
- Visual comparison: KDE plots, violin plots, correlation heatmaps

### Phase 4 — Model Comparison
- Train Random Forest on: Original, SMOTE-augmented, and CTGAN-augmented datasets
- Compare using: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Visualize: Confusion matrices, ROC curves, metrics bar charts

## Expected Results

- CTGAN-augmented model should outperform baseline on **Recall** and **F1-Score**
- Synthetic data quality score (SDMetrics) should be **above 0.75**
- CTGAN should match or outperform SMOTE on fraud detection recall

## IBM Relevance

This project directly maps to real IBM use cases:
- **IBM OpenScale / Watson OpenScale** — AI fairness and bias detection
- **IBM Synthetic Data** tools for privacy-preserving ML in healthcare and finance
- Fraud detection, risk scoring, and regulatory compliance

---

*Submitted under IBM SkillsBuild / IBM Academic Initiative*
