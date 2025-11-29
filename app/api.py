"""
DebtSage API - REST Endpoints for ML Model
10Alytics Global Hackathon 2025

FastAPI application providing REST endpoints for debt crisis prediction model.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import json

# Initialize FastAPI app
app = FastAPI(
    title="DebtSage API",
    description="AI-Powered African Sovereign Debt Crisis Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data at startup
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Global variables to store loaded models and data
models = {}
scaler = None
data_cache = {}

@app.on_event("startup")
async def load_models_and_data():
    """Load ML models and data on application startup"""
    global models, scaler, data_cache
    
    try:
        # Load ML models
        models['xgboost'] = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        models['random_forest'] = joblib.load(MODELS_DIR / "random_forest_model.pkl")
        models['logistic_regression'] = joblib.load(MODELS_DIR / "logistic_regression_model.pkl")
        scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        
        # Load data files
        data_cache['panel_data'] = pd.read_csv(DATA_DIR / "cleaned_panel_data.csv")
        data_cache['risk_scores'] = pd.read_csv(DATA_DIR / "country_risk_scores.csv")
        data_cache['model_performance'] = pd.read_csv(DATA_DIR / "model_performance.csv")
        data_cache['feature_importance'] = pd.read_csv(DATA_DIR / "feature_importance.csv")
        data_cache['fiscal_metrics'] = pd.read_csv(DATA_DIR / "fiscal_metrics_complete.csv")
        data_cache['fiscal_scorecard'] = pd.read_csv(DATA_DIR / "fiscal_health_scorecard.csv")
        data_cache['debt_projections'] = pd.read_csv(DATA_DIR / "debt_projections.csv")
        
        print("✅ Models and data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models/data: {e}")
        raise

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    debt_to_gdp: float = Field(..., description="Debt-to-GDP ratio (%)", ge=0, le=500)
    deficit_to_gdp: float = Field(..., description="Budget deficit as % of GDP", ge=-50, le=50)
    revenue_to_gdp: float = Field(..., description="Revenue-to-GDP ratio (%)", ge=0, le=100)
    inflation_rate: float = Field(..., description="Inflation rate (%)", ge=-10, le=100)
    gdp_growth: float = Field(..., description="GDP growth rate (%)", ge=-20, le=20)
    external_debt_ratio: float = Field(..., description="External debt as % of total (%)", ge=0, le=100)
    debt_service_to_revenue: float = Field(..., description="Debt service to revenue ratio (%)", ge=0, le=200)
    reserves_months: float = Field(..., description="Import cover in months", ge=0, le=24)
    primary_balance: float = Field(..., description="Primary balance as % of GDP", ge=-30, le=30)
    exchange_rate_change: float = Field(..., description="Exchange rate % change", ge=-100, le=100)
    model: str = Field(default="xgboost", description="Model to use: xgboost, random_forest, or logistic_regression")

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    risk_score: float = Field(..., description="Crisis risk probability (0-100%)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, or High")
    risk_prediction: int = Field(..., description="Binary prediction: 0=No Crisis, 1=Crisis")
    confidence: float = Field(..., description="Model confidence (%)")
    model_used: str = Field(..., description="Model used for prediction")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    predictions: List[PredictionRequest]

class CountryDataRequest(BaseModel):
    """Request for country-specific historical data"""
    country: str = Field(..., description="Country name")
    start_year: Optional[int] = Field(None, description="Start year for data")
    end_year: Optional[int] = Field(None, description="End year for data")

# API Endpoints

@app.get("/")
async def root():
    """API root endpoint with welcome message"""
    return {
        "message": "Welcome to DebtSage API - AI-Powered Debt Crisis Prediction",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "countries": "/countries",
            "country_data": "/country/{country}",
            "risk_scores": "/risk-scores",
            "model_performance": "/model/performance",
            "feature_importance": "/model/feature-importance",
            "fiscal_metrics": "/fiscal/metrics",
            "fiscal_scorecard": "/fiscal/scorecard",
            "projections": "/projections/{country}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "data_files_loaded": len(data_cache),
        "available_models": list(models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_crisis_risk(request: PredictionRequest):
    """
    Predict debt crisis risk for given economic indicators
    
    Returns risk score (0-100%), risk level, and binary prediction.
    """
    try:
        # Select model
        if request.model not in models:
            raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found. Available: {list(models.keys())}")
        
        model = models[request.model]
        
        # Prepare features (order matters - must match training order)
        features = np.array([[
            request.debt_to_gdp,
            request.deficit_to_gdp,
            request.revenue_to_gdp,
            request.inflation_rate,
            request.gdp_growth,
            request.external_debt_ratio,
            request.debt_service_to_revenue,
            request.reserves_months,
            request.primary_balance,
            request.exchange_rate_change
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get prediction
        prediction = model.predict(features_scaled)[0]
        risk_probability = model.predict_proba(features_scaled)[0][1]
        
        # Calculate risk score and level
        risk_score = float(risk_probability * 100)
        
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Confidence (distance from decision boundary)
        confidence = float(abs(risk_probability - 0.5) * 200)
        
        return PredictionResponse(
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            risk_prediction=int(prediction),
            confidence=round(confidence, 2),
            model_used=request.model
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction endpoint for multiple predictions at once
    
    Returns list of predictions for all input records.
    """
    try:
        results = []
        for pred_request in request.predictions:
            result = await predict_crisis_risk(pred_request)
            results.append(result.dict())
        
        return {
            "predictions": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/countries")
async def get_countries():
    """Get list of all available countries in dataset"""
    try:
        countries = sorted(data_cache['panel_data']['country'].unique().tolist())
        
        return {
            "countries": countries,
            "count": len(countries)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching countries: {str(e)}")

@app.get("/country/{country}")
async def get_country_data(
    country: str,
    start_year: Optional[int] = Query(None, description="Start year"),
    end_year: Optional[int] = Query(None, description="End year")
):
    """
    Get historical data for a specific country
    
    Returns time series data for all available indicators.
    """
    try:
        df = data_cache['panel_data']
        
        # Filter by country
        country_data = df[df['country'].str.lower() == country.lower()].copy()
        
        if country_data.empty:
            raise HTTPException(status_code=404, detail=f"Country '{country}' not found")
        
        # Filter by year range if provided
        if start_year:
            country_data = country_data[country_data['year'] >= start_year]
        if end_year:
            country_data = country_data[country_data['year'] <= end_year]
        
        # Convert to dictionary format (replace NaN with None for JSON compliance)
        country_data = country_data.replace({np.nan: None})
        records = country_data.to_dict('records')
        
        return {
            "country": country,
            "records": records,
            "count": len(records),
            "year_range": {
                "min": int(country_data['year'].min()) if not country_data.empty else None,
                "max": int(country_data['year'].max()) if not country_data.empty else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching country data: {str(e)}")

@app.get("/risk-scores")
async def get_risk_scores(
    country: Optional[str] = Query(None, description="Filter by country"),
    min_risk: Optional[float] = Query(None, description="Minimum risk score", ge=0, le=100)
):
    """
    Get ML risk scores for all countries
    
    Can filter by country name or minimum risk threshold.
    """
    try:
        df = data_cache['risk_scores'].copy()
        
        # Apply filters
        if country:
            df = df[df['country'].str.lower() == country.lower()]
        
        if min_risk is not None:
            df = df[df['risk_score'] >= min_risk]
        
        # Sort by risk score descending
        df = df.sort_values('risk_score', ascending=False)
        
        # Replace NaN with None for JSON compliance
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        
        return {
            "risk_scores": records,
            "count": len(records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching risk scores: {str(e)}")

@app.get("/model/performance")
async def get_model_performance():
    """
    Get performance metrics for all trained models
    
    Returns AUC-ROC, precision, recall, F1-score for each model.
    """
    try:
        df = data_cache['model_performance']
        
        # Group by model for cleaner output
        models_list = []
        for model_name in df['model'].unique():
            model_subset = df[df['model'] == model_name].copy()
            model_subset = model_subset.replace({np.nan: None})
            model_data = model_subset.to_dict('records')
            models_list.append({
                "model": model_name,
                "metrics": model_data
            })
        
        return {
            "models": models_list,
            "count": len(models_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model performance: {str(e)}")

@app.get("/model/feature-importance")
async def get_feature_importance(
    top_n: Optional[int] = Query(None, description="Return only top N features", ge=1, le=50)
):
    """
    Get feature importance scores from XGBoost model
    
    Returns ranked list of features by importance.
    """
    try:
        df = data_cache['feature_importance'].copy()
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False)
        
        # Limit to top N if requested
        if top_n:
            df = df.head(top_n)
        
        # Replace NaN with None for JSON compliance
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        
        return {
            "features": records,
            "count": len(records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feature importance: {str(e)}")

@app.get("/fiscal/metrics")
async def get_fiscal_metrics(
    country: Optional[str] = Query(None, description="Filter by country")
):
    """
    Get comprehensive fiscal sustainability metrics
    
    Includes debt ratios, revenue efficiency, expenditure composition.
    """
    try:
        df = data_cache['fiscal_metrics'].copy()
        
        if country:
            df = df[df['country'].str.lower() == country.lower()]
            
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No fiscal data found for '{country}'")
        
        # Replace NaN with None for JSON compliance
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        
        return {
            "fiscal_metrics": records,
            "count": len(records)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching fiscal metrics: {str(e)}")

@app.get("/fiscal/scorecard")
async def get_fiscal_scorecard():
    """
    Get fiscal health scorecard with composite stress scores
    
    Returns rankings and fiscal health assessment for focus countries.
    """
    try:
        df = data_cache['fiscal_scorecard'].copy()
        
        # Sort by fiscal stress score descending (highest stress first)
        df = df.sort_values('fiscal_stress_score', ascending=False)
        
        # Replace NaN with None for JSON compliance
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        
        return {
            "scorecard": records,
            "count": len(records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching fiscal scorecard: {str(e)}")

@app.get("/projections/{country}")
async def get_debt_projections(country: str):
    """
    Get 5-year debt projections for a specific country
    
    Returns baseline, optimistic, and stress scenarios.
    """
    try:
        df = data_cache['debt_projections'].copy()
        
        # Filter by country
        country_projections = df[df['country'].str.lower() == country.lower()]
        
        if country_projections.empty:
            # Get available countries for helpful error message
            available = sorted(df['country'].unique().tolist())
            raise HTTPException(
                status_code=404, 
                detail=f"No projections found for '{country}'. Available countries: {', '.join(available)}"
            )
        
        # Organize by scenario
        scenarios = {}
        for scenario in country_projections['scenario'].unique():
            scenario_data = country_projections[country_projections['scenario'] == scenario].copy()
            scenario_data = scenario_data.replace({np.nan: None})
            scenarios[scenario] = scenario_data.to_dict('records')
        
        # Handle both 'year' and 'year_ahead' column names
        year_col = 'year_ahead' if 'year_ahead' in country_projections.columns else 'year'
        
        return {
            "country": country,
            "scenarios": scenarios,
            "years_ahead": sorted(country_projections[year_col].unique().tolist())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projections: {str(e)}")

@app.get("/stats/summary")
async def get_summary_statistics():
    """
    Get summary statistics across all data
    
    Returns overview metrics for the entire dataset.
    """
    try:
        panel_data = data_cache['panel_data']
        risk_scores = data_cache['risk_scores']
        
        # Calculate means with NaN handling
        avg_risk = risk_scores['risk_score'].mean()
        avg_debt = panel_data.get('debt_to_gdp', pd.Series([0])).mean()
        avg_deficit = panel_data.get('deficit_to_gdp', pd.Series([0])).mean()
        avg_revenue = panel_data.get('revenue_to_gdp', pd.Series([0])).mean()
        
        return {
            "dataset": {
                "total_observations": len(panel_data),
                "countries": len(panel_data['country'].unique()),
                "year_range": {
                    "min": int(panel_data['year'].min()),
                    "max": int(panel_data['year'].max())
                }
            },
            "risk_distribution": {
                "high_risk": int((risk_scores['risk_score'] >= 60).sum()),
                "medium_risk": int(((risk_scores['risk_score'] >= 30) & (risk_scores['risk_score'] < 60)).sum()),
                "low_risk": int((risk_scores['risk_score'] < 30).sum()),
                "average_risk": float(avg_risk) if not pd.isna(avg_risk) else None
            },
            "fiscal_indicators": {
                "avg_debt_to_gdp": float(avg_debt) if not pd.isna(avg_debt) else None,
                "avg_deficit_to_gdp": float(avg_deficit) if not pd.isna(avg_deficit) else None,
                "avg_revenue_to_gdp": float(avg_revenue) if not pd.isna(avg_revenue) else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating summary statistics: {str(e)}")

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "path": str(request.url)
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render sets this)
    port = int(os.getenv("PORT", 8000))
    
    print("="*70)
    print("🚀 Starting DebtSage API Server")
    print("="*70)
    print(f"\n📊 Port: {port}")
    print(f"📊 API Endpoints:")
    print(f"  • Documentation: http://0.0.0.0:{port}/docs")
    print(f"  • Alternative Docs: http://0.0.0.0:{port}/redoc")
    print(f"  • Health Check: http://0.0.0.0:{port}/health")
    print("\n💡 Example Request:")
    print(f"""
    curl -X POST http://0.0.0.0:{port}/predict \\
      -H "Content-Type: application/json" \\
      -d '{{
        "debt_to_gdp": 65.0,
        "deficit_to_gdp": -3.5,
        "revenue_to_gdp": 18.5,
        "inflation_rate": 5.2,
        "gdp_growth": 3.8,
        "external_debt_ratio": 45.0,
        "debt_service_to_revenue": 25.0,
        "reserves_months": 4.5,
        "primary_balance": -1.2,
        "exchange_rate_change": 2.1,
        "model": "xgboost"
      }}'
    """)
    print("="*70)
    
    # Disable reload in production (Render)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
