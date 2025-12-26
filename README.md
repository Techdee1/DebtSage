# DebtSage - AI-Powered African Sovereign Debt Crisis Analysis

### Project Overview

**DebtSage** is an advanced analytical platform that leverages machine learning to predict sovereign debt crises and provide actionable policy recommendations for African countries. Using 60+ years of fiscal data across 14 countries, our system achieves 93.4% accuracy in identifying debt distress situations.

---

## 🎯 Key Features

### 1. Machine Learning Crisis Prediction
- **XGBoost Classifier**: 93.4% AUC-ROC, 90.6% precision, 78.4% recall
- **Random Forest & Logistic Regression**: Ensemble validation
- **41 Engineered Features**: Temporal dynamics, volatility measures, fiscal ratios
- **Real-time Risk Scoring**: 0-100% crisis probability for each country

### 2. Comprehensive Fiscal Analysis
- **Debt Sustainability Metrics**: Debt-to-GDP, debt service burden
- **Fiscal Balance Assessment**: Budget deficits, revenue efficiency
- **Social Spending Analysis**: Health & education trade-offs
- **Cross-Country Comparisons**: Fiscal health rankings

### 3. 5-Year Debt Projections
- **Debt Dynamics Modeling**: Interest rates, GDP growth, primary balance
- **Three Scenarios**: Baseline, optimistic, stress
- **Sustainability Thresholds**: 70% debt-to-GDP crisis indicator

### 4. Interactive Dashboard
- **5 Pages**: Overview, Country Analysis, ML Predictor, Projections, Comparisons
- **Real-time Visualizations**: Plotly interactive charts
- **User-Friendly Interface**: Streamlit web application

---

## 📊 Key Findings

### High-Risk Countries (ML Risk Score > 90%)
- 🔴 **Nigeria**: 98.5% risk, extreme fiscal stress
- 🔴 **Egypt**: 98.2% risk, large deficits

### Moderate-Risk Countries (Risk 40-60%)
- 🟡 **Togo**: 51.7% risk, 67% debt-to-GDP
- 🟡 **Ivory Coast**: 48.6% risk, manageable with reforms

### Top Risk Predictors (Feature Importance)
1. **Revenue-to-GDP Rolling Average** (48.4%) - Most critical factor
2. **Inflation Rate Volatility** (6.2%) - Early warning signal
3. **Trade Balance Volatility** (5.0%) - External sector stress
4. **Deficit-to-GDP Persistence** (4.2%) - Structural imbalance
5. **Revenue Generation Capacity** (3.3%) - Fiscal space

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Techdee1/DebtSage.git
cd DebtSage

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r app/requirements.txt
```

### Run Dashboard

```bash
cd app
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
DebtSage/
├── data/                          # Datasets
│   ├── cleaned_panel_data.csv     # 623 country-year observations
│   ├── ml_features_complete.csv   # 41 ML features
│   ├── country_risk_scores.csv    # ML predictions
│   ├── fiscal_metrics_complete.csv # Fiscal indicators
│   ├── debt_projections.csv       # 5-year forecasts
│   └── ...
├── notebooks/                      # Analysis notebooks
│   ├── 00_data_prep.ipynb         # Data cleaning
│   ├── 01_eda.ipynb               # Exploratory analysis
│   ├── 02_feature_engineering.ipynb # ML features
│   ├── 03_ml_debt_crisis.ipynb    # Model training
│   ├── 04_fiscal_metrics.ipynb    # Fiscal analysis
│   └── ...
├── models/                         # Trained ML models
│   ├── xgboost_model.pkl          # Best model (181 KB)
│   ├── random_forest_model.pkl    # Alternative (466 KB)
│   ├── logistic_regression_model.pkl # Baseline
│   └── feature_scaler.pkl         # StandardScaler
├── app/                           # Streamlit dashboard
│   ├── streamlit_app.py          # Main application
│   └── requirements.txt          # Dependencies
├── slides/                        # Presentation materials
│   └── figs/                     # Visualizations
├── PROGRESS_SUMMARY.md           # Detailed methodology
└── README.md                     # This file
```

---

## 🔬 Methodology

### Data Processing Pipeline

1. **Data Collection**: 60+ years of fiscal data (1960-2025) for 14 African countries
2. **Data Cleaning**: Handle missing values, outliers, unit standardization
3. **Feature Engineering**: Create 41 ML features including:
   - Core ratios (debt-to-GDP, deficit-to-GDP, revenue-to-GDP)
   - Temporal features (1yr/3yr changes, rolling averages, volatility)
   - Persistence indicators (consecutive deficit years)

### Machine Learning Pipeline

1. **Crisis Labeling**: Debt > 70% GDP OR Deficit > 5% GDP
2. **Train-Test Split**: Temporal split (73% train, 27% test)
3. **Feature Scaling**: StandardScaler normalization
4. **Model Training**: 3 classifiers with class balancing
5. **Evaluation**: AUC-ROC, precision, recall, F1-score
6. **Deployment**: Export models for real-time prediction

### Debt Dynamics Model

```
Debt(t+1) = Debt(t) × (1 + interest_rate - gdp_growth) + Primary_Deficit(t)
```

- **Baseline**: Historical averages
- **Optimistic**: +2% growth, -30% deficit
- **Stress**: -2% growth, +30% deficit

---

## 📈 Results Summary

### Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **93.4%** | **90.6%** | **78.4%** | **84.1%** | **93.4%** |
| Random Forest | 89.2% | 88.0% | 59.5% | 71.0% | 91.6% |
| Logistic Regression | 82.6% | 70.0% | 37.8% | 49.1% | 82.3% |

### Policy Impact

- **Revenue Mobilization**: 48% of crisis prediction power - strengthen tax systems
- **Inflation Control**: 6% importance - monetary policy critical
- **External Stability**: 5% importance - manage trade balance volatility
- **Fiscal Consolidation**: Target 1-2% deficit reduction annually
- **Debt Restructuring**: Nigeria & Egypt require immediate intervention

---

## 🛠️ Technologies Used

- **Python 3.12**: Core language
- **pandas 2.0+**: Data manipulation
- **scikit-learn 1.3+**: ML models (Random Forest, Logistic Regression)
- **XGBoost 2.0+**: Gradient boosting (best model)
- **Plotly 5.14+**: Interactive visualizations
- **Streamlit 1.28+**: Web dashboard
- **Jupyter Lab**: Notebook development

---

### Data Sources
- World Bank Open Data
- IMF Fiscal Monitor
- African Development Bank Statistics

---

## 📝 License

This project is developed for educational and research purposes.

---

## 📞 Contact

For questions or collaboration:
- GitHub: [@Techdee1](https://github.com/Techdee1)
- Project Repository: [DebtSage](https://github.com/Techdee1/DebtSage)

---

## 🙏 Acknowledgments

Special thanks to:
- Data providers (World Bank, IMF, AfDB)
- Open-source community (scikit-learn, XGBoost, Plotly, Streamlit)

---

**Last Updated**: November 29, 2025  
**Status**: ✅ Complete - Ready for submission
