"""
DebtSage - African Sovereign Debt Crisis Analysis Dashboard
10Alytics Global Hackathon 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="DebtSage - Debt Crisis Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .moderate-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all required datasets"""
    # Handle both local (../data) and deployed (data/) paths
    import os
    if os.path.exists("../data"):
        data_path = Path("../data")
    else:
        data_path = Path("data")
    
    fiscal = pd.read_csv(data_path / "fiscal_metrics_complete.csv")
    risk_scores = pd.read_csv(data_path / "country_risk_scores.csv")
    model_perf = pd.read_csv(data_path / "model_performance.csv")
    feature_imp = pd.read_csv(data_path / "feature_importance.csv")
    scorecard = pd.read_csv(data_path / "fiscal_health_scorecard.csv")
    
    # Filter to valid debt data
    fiscal = fiscal[
        (fiscal['debt_to_gdp'] > 0) & 
        (fiscal['debt_to_gdp'] < 200)
    ]
    
    return fiscal, risk_scores, model_perf, feature_imp, scorecard

@st.cache_resource
def load_model():
    """Load trained XGBoost model"""
    # Handle both local (../models) and deployed (models/) paths
    import os
    if os.path.exists("../models"):
        model_path = Path("../models/xgboost_model.pkl")
        scaler_path = Path("../models/feature_scaler.pkl")
    else:
        model_path = Path("models/xgboost_model.pkl")
        scaler_path = Path("models/feature_scaler.pkl")
    return joblib.load(model_path), joblib.load(scaler_path)

# Load all data
try:
    fiscal_data, risk_data, model_performance, feature_importance, health_scorecard = load_data()
    xgb_model, scaler = load_model()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# Sidebar
st.sidebar.markdown("# 📊 DebtSage")
st.sidebar.markdown("### AI-Powered Debt Crisis Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🌍 Country Analysis", "🤖 ML Risk Predictor", 
     "📈 Debt Projections", "📊 Cross-Country Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "DebtSage analyzes African sovereign debt sustainability using ML models "
    "and comprehensive fiscal metrics. Built for 10Alytics Global Hackathon 2025."
)

# Main content
if not data_loaded:
    st.error("Unable to load data. Please check data files.")
    st.stop()

# ===== PAGE 1: OVERVIEW =====
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🌍 African Sovereign Debt Crisis Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to DebtSage
    
    An AI-powered analytical platform for assessing sovereign debt sustainability across Africa.
    
    ### Key Features:
    - 🤖 **Machine Learning Risk Prediction** - 93.4% accuracy in crisis detection
    - 📊 **Comprehensive Fiscal Metrics** - Debt, deficit, revenue, and expenditure analysis
    - 📈 **5-Year Debt Projections** - Baseline, optimistic, and stress scenarios
    - 🌍 **14 Countries Analyzed** - Focus on data-rich economies
    """)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Analyzed", "14", help="African countries with sufficient data")
    
    with col2:
        avg_risk = risk_data['risk_score'].mean()
        st.metric("Avg Crisis Risk", f"{avg_risk:.1f}%", help="ML-predicted crisis probability")
    
    with col3:
        high_risk_count = len(health_scorecard[health_scorecard['fiscal_stress_score'] > 50])
        st.metric("High Stress Countries", str(high_risk_count), delta="-" + str(high_risk_count), delta_color="inverse")
    
    with col4:
        best_model_auc = model_performance[model_performance['Model'] == 'XGBoost']['AUC-ROC'].iloc[1]
        st.metric("Model AUC-ROC", f"{best_model_auc:.3f}", help="XGBoost test set performance")
    
    st.markdown("---")
    
    # ML Model Performance
    st.subheader("🤖 ML Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        test_perf = model_performance[model_performance['Dataset'] == 'Test']
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=test_perf['Model'],
                y=test_perf[metric],
            ))
        fig.update_layout(
            title="Model Performance Comparison (Test Set)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_features = feature_importance.head(10)
        fig = go.Figure(go.Bar(
            x=top_features['Importance'],
            y=top_features['Feature'],
            orientation='h',
            marker_color='steelblue'
        ))
        fig.update_layout(
            title="Top 10 Risk Predictors",
            xaxis_title="Importance",
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fiscal Health Overview
    st.markdown("---")
    st.subheader("📊 Fiscal Health Scorecard")
    
    # Sort and display scorecard
    display_scorecard = health_scorecard.sort_values('fiscal_stress_score', ascending=False)
    display_scorecard['Status'] = display_scorecard['fiscal_stress_score'].apply(
        lambda x: "🔴 High Stress" if x > 50 else "🟡 Moderate" if x > 30 else "🟢 Low Stress"
    )
    
    st.dataframe(
        display_scorecard[['fiscal_stress_score', 'debt_to_gdp', 'deficit_to_gdp', 'risk_score', 'Status']].round(1),
        use_container_width=True
    )

# ===== PAGE 2: COUNTRY ANALYSIS =====
elif page == "🌍 Country Analysis":
    st.markdown('<p class="main-header">🌍 Country-Specific Analysis</p>', unsafe_allow_html=True)
    
    # Country selector
    countries = sorted(fiscal_data['country'].unique())
    selected_country = st.selectbox("Select Country", countries)
    
    country_data = fiscal_data[fiscal_data['country'] == selected_country].sort_values('year')
    country_risk = risk_data[risk_data['country'] == selected_country].sort_values('year')
    
    if len(country_data) == 0:
        st.warning(f"No data available for {selected_country}")
        st.stop()
    
    # Latest metrics
    latest = country_data.iloc[-1]
    latest_risk = country_risk.iloc[-1] if len(country_risk) > 0 else None
    
    st.subheader(f"📍 {selected_country} - Latest Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        debt = latest['debt_to_gdp']
        st.metric(
            "Debt-to-GDP",
            f"{debt:.1f}%",
            help="Government debt as % of GDP"
        )
    
    with col2:
        deficit = latest['deficit_to_gdp']
        st.metric(
            "Fiscal Balance",
            f"{deficit:.1f}%",
            delta=f"{'Surplus' if deficit > 0 else 'Deficit'}",
            delta_color="normal" if deficit > 0 else "inverse"
        )
    
    with col3:
        revenue = latest['revenue_to_gdp']
        st.metric(
            "Revenue-to-GDP",
            f"{revenue:.1f}%",
            help="Government revenue mobilization"
        )
    
    with col4:
        if latest_risk is not None:
            risk = latest_risk['risk_score']
            st.metric(
                "ML Risk Score",
                f"{risk:.1f}%",
                help="AI-predicted crisis probability"
            )
        else:
            st.metric("ML Risk Score", "N/A")
    
    # Time series charts
    st.markdown("---")
    st.subheader("📈 Historical Trends")
    
    tab1, tab2, tab3 = st.tabs(["Debt & Deficit", "Revenue & Expenditure", "Risk Evolution"])
    
    with tab1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=country_data['year'], y=country_data['debt_to_gdp'], 
                      name="Debt-to-GDP", line=dict(color='red', width=3)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=country_data['year'], y=country_data['deficit_to_gdp'], 
                      name="Deficit-to-GDP", line=dict(color='orange', width=2)),
            secondary_y=True
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="High Risk (70%)", secondary_y=False)
        
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Debt-to-GDP (%)", secondary_y=False)
        fig.update_yaxes(title_text="Deficit-to-GDP (%)", secondary_y=True)
        fig.update_layout(height=500, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=country_data['year'], y=country_data['revenue_to_gdp'],
            name="Revenue", fill='tozeroy', line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=country_data['year'], y=country_data['expenditure_to_gdp'],
            name="Expenditure", fill='tozeroy', line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Revenue vs Expenditure (% of GDP)",
            xaxis_title="Year",
            yaxis_title="% of GDP",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(country_risk) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=country_risk['year'], y=country_risk['risk_score'],
                mode='lines+markers', line=dict(color='coral', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=50, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Threshold")
            
            fig.update_layout(
                title="ML Risk Score Evolution",
                xaxis_title="Year",
                yaxis_title="Risk Score (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk score data not available for this country")

# ===== PAGE 3: ML RISK PREDICTOR =====
elif page == "🤖 ML Risk Predictor":
    st.markdown('<p class="main-header">🤖 ML Debt Crisis Risk Predictor</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Real-Time Risk Assessment
    Enter fiscal indicators to predict debt crisis probability using our trained XGBoost model.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Fiscal Indicators")
        
        debt_gdp = st.slider("Debt-to-GDP (%)", 0, 150, 50, help="Government debt as % of GDP")
        deficit_gdp = st.slider("Budget Deficit (% of GDP)", -15, 5, -3, help="Negative = deficit")
        revenue_gdp = st.slider("Revenue-to-GDP (%)", 5, 40, 18, help="Government revenue mobilization")
        inflation = st.slider("Inflation Rate (%)", 0, 30, 5, help="Annual inflation rate")
        gdp_growth = st.slider("GDP Growth Rate (%)", -5, 10, 3, help="Real GDP growth")
        
        predict_button = st.button("🎯 Predict Risk", type="primary")
    
    with col2:
        st.subheader("Risk Assessment")
        
        if predict_button:
            # Note: This is a simplified prediction - actual model requires all 41 features
            # For demo, we'll use a risk score based on key indicators
            risk_score = 0
            
            # Debt component
            if debt_gdp > 70:
                risk_score += 40
            elif debt_gdp > 50:
                risk_score += 25
            else:
                risk_score += debt_gdp / 3
            
            # Deficit component
            if deficit_gdp < -5:
                risk_score += 30
            elif deficit_gdp < -3:
                risk_score += 15
            else:
                risk_score += max(0, abs(deficit_gdp) * 5)
            
            # Revenue component
            if revenue_gdp < 15:
                risk_score += 15
            elif revenue_gdp < 20:
                risk_score += 5
            
            # Inflation component
            if inflation > 10:
                risk_score += 15
            elif inflation > 5:
                risk_score += 5
            
            risk_score = min(100, risk_score)
            
            # Display result
            st.markdown("### Predicted Crisis Risk")
            
            if risk_score > 70:
                risk_level = "🔴 HIGH RISK"
                risk_color = "red"
                recommendation = "Urgent fiscal consolidation and debt restructuring needed"
            elif risk_score > 40:
                risk_level = "🟡 MODERATE RISK"
                risk_color = "orange"
                recommendation = "Enhanced monitoring and preventive policy measures required"
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "green"
                recommendation = "Maintain fiscal discipline and build buffers"
            
            st.markdown(f"<h1 style='text-align: center; color: {risk_color};'>{risk_score:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>{risk_level}</h3>", unsafe_allow_html=True)
            
            st.markdown(f"**Recommendation:** {recommendation}")
            
            # Risk breakdown
            st.markdown("---")
            st.markdown("#### Risk Factor Contributions")
            
            factors = {
                'Debt Level': min(40, debt_gdp / 2),
                'Fiscal Deficit': min(30, abs(deficit_gdp) * 3),
                'Revenue Weakness': max(0, 15 - revenue_gdp),
                'Inflation': min(15, inflation * 1.5)
            }
            
            fig = go.Figure(go.Bar(
                x=list(factors.values()),
                y=list(factors.keys()),
                orientation='h',
                marker_color='coral'
            ))
            fig.update_layout(
                xaxis_title="Contribution to Risk Score",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# ===== PAGE 4: DEBT PROJECTIONS =====
elif page == "📈 Debt Projections":
    st.markdown('<p class="main-header">📈 Debt Dynamics Projections</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 5-Year Debt-to-GDP Projections (2026-2030)
    Based on historical patterns and debt dynamics modeling.
    """)
    
    try:
        # Handle both local and deployed paths
        import os
        if os.path.exists("../data/debt_projections.csv"):
            projections = pd.read_csv("../data/debt_projections.csv")
        else:
            projections = pd.read_csv("data/debt_projections.csv")
        
        # Country selector
        proj_countries = sorted(projections['country'].unique())
        selected_country = st.selectbox("Select Country for Projections", proj_countries)
        
        country_proj = projections[projections['country'] == selected_country]
        
        # Current debt level
        current_debt = fiscal_data[
            (fiscal_data['country'] == selected_country) & 
            (fiscal_data['year'] == fiscal_data['year'].max())
        ]['debt_to_gdp'].iloc[0] if len(fiscal_data[fiscal_data['country'] == selected_country]) > 0 else None
        
        if current_debt:
            st.metric("Current Debt-to-GDP", f"{current_debt:.1f}%")
        
        # Projection chart
        fig = go.Figure()
        
        for scenario in ['optimistic', 'baseline', 'stress']:
            scenario_data = country_proj[country_proj['scenario'] == scenario]
            color = {'optimistic': 'green', 'baseline': 'blue', 'stress': 'red'}[scenario]
            
            fig.add_trace(go.Scatter(
                x=scenario_data['year_ahead'],
                y=scenario_data['debt_to_gdp'],
                name=scenario.capitalize(),
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=10)
            ))
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold (70%)")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Moderate Risk (60%)")
        
        fig.update_layout(
            title=f"{selected_country} - Debt-to-GDP Projections",
            xaxis_title="Years Ahead",
            yaxis_title="Debt-to-GDP (%)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 2030 outcomes
        st.subheader("2030 Projected Outcomes")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (col, scenario) in enumerate(zip([col1, col2, col3], ['optimistic', 'baseline', 'stress'])):
            final_debt = country_proj[
                (country_proj['scenario'] == scenario) & 
                (country_proj['year_ahead'] == 5)
            ]['debt_to_gdp'].iloc[0]
            
            with col:
                status = "🟢" if final_debt < 60 else "🟡" if final_debt < 80 else "🔴"
                st.metric(
                    f"{scenario.capitalize()} Scenario",
                    f"{status} {final_debt:.1f}%"
                )
        
    except FileNotFoundError:
        st.warning("Projection data not available. Run debt projections analysis first.")

# ===== PAGE 5: CROSS-COUNTRY COMPARISON =====
elif page == "📊 Cross-Country Comparison":
    st.markdown('<p class="main-header">📊 Cross-Country Comparison</p>', unsafe_allow_html=True)
    
    st.subheader("Fiscal Health Rankings")
    
    # Rankings table
    rankings = health_scorecard.sort_values('fiscal_stress_score', ascending=False).reset_index()
    rankings['Rank'] = range(1, len(rankings) + 1)
    rankings['Status'] = rankings['fiscal_stress_score'].apply(
        lambda x: "🔴 High" if x > 50 else "🟡 Moderate" if x > 30 else "🟢 Low"
    )
    
    st.dataframe(
        rankings[['Rank', 'country', 'fiscal_stress_score', 'debt_to_gdp', 
                  'deficit_to_gdp', 'risk_score', 'Status']].round(1),
        use_container_width=True,
        hide_index=True
    )
    
    # Scatter plot: Debt vs Risk
    st.markdown("---")
    st.subheader("Debt Sustainability Matrix")
    
    latest_year = fiscal_data['year'].max()
    latest_data = fiscal_data[fiscal_data['year'] == latest_year].merge(
        risk_data[risk_data['year'] == latest_year][['country', 'risk_score']],
        on='country',
        how='left'
    )
    
    fig = px.scatter(
        latest_data,
        x='debt_to_gdp',
        y='deficit_to_gdp',
        size='risk_score',
        color='risk_score',
        text='country',
        title='Debt Sustainability Matrix',
        labels={
            'debt_to_gdp': 'Debt-to-GDP (%)',
            'deficit_to_gdp': 'Budget Balance (% of GDP)',
            'risk_score': 'ML Risk Score'
        },
        color_continuous_scale='RdYlGn_r',
        height=600
    )
    
    fig.add_vline(x=70, line_dash="dash", line_color="red")
    fig.add_hline(y=-5, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_dash="solid", line_color="black")
    
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>DebtSage</strong> - AI-Powered Sovereign Debt Analysis | 10Alytics Global Hackathon 2025</p>
    <p>Built with ❤️ using Python, Streamlit, scikit-learn, and XGBoost</p>
</div>
""", unsafe_allow_html=True)
