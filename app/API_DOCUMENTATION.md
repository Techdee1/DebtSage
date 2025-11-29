# DebtSage API Documentation

## Overview

The DebtSage API provides REST endpoints for AI-powered debt crisis prediction. Your software engineer can integrate these endpoints into any frontend application.

**Base URL:** `https://debtsage-api.onrender.com` (production)  
**API Version:** 1.0.0  
**Documentation:** https://debtsage-api.onrender.com/docs (Interactive Swagger UI)

---

## Quick Start

### 1. Installation

```bash
# Install API dependencies
pip install fastapi uvicorn[standard] pydantic python-multipart

# Or install all dependencies
pip install -r requirements.txt
pip install -r app/api_requirements.txt
```

### 2. Start API Server

```bash
# From project root
cd /workspaces/10Analytics
python app/api.py

# Or using uvicorn directly
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access Documentation

- **Interactive Docs:** https://debtsage-api.onrender.com/docs (Swagger UI)
- **Alternative Docs:** https://debtsage-api.onrender.com/redoc (ReDoc)
- **Health Check:** https://debtsage-api.onrender.com/health

---

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API root with endpoint list |
| GET | `/health` | Health check and status |
| POST | `/predict` | Single debt crisis prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/countries` | List all countries |
| GET | `/country/{country}` | Get country historical data |
| GET | `/risk-scores` | Get ML risk scores |
| GET | `/model/performance` | Model performance metrics |
| GET | `/model/feature-importance` | Feature importance rankings |
| GET | `/fiscal/metrics` | Fiscal sustainability metrics |
| GET | `/fiscal/scorecard` | Fiscal health scorecard |
| GET | `/projections/{country}` | 5-year debt projections |
| GET | `/stats/summary` | Summary statistics |

---

## Detailed Endpoint Reference

### 1. Predict Debt Crisis Risk

**Endpoint:** `POST /predict`

**Description:** Predict debt crisis risk for given economic indicators.

**Request Body:**
```json
{
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
}
```

**Response:**
```json
{
  "risk_score": 42.35,
  "risk_level": "Medium",
  "risk_prediction": 0,
  "confidence": 84.70,
  "model_used": "xgboost"
}
```

**cURL Example:**
```bash
curl -X POST https://debtsage-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**JavaScript Example:**
```javascript
const response = await fetch('https://debtsage-api.onrender.com/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    debt_to_gdp: 65.0,
    deficit_to_gdp: -3.5,
    revenue_to_gdp: 18.5,
    inflation_rate: 5.2,
    gdp_growth: 3.8,
    external_debt_ratio: 45.0,
    debt_service_to_revenue: 25.0,
    reserves_months: 4.5,
    primary_balance: -1.2,
    exchange_rate_change: 2.1,
    model: 'xgboost'
  })
});

const data = await response.json();
console.log(data);
// Output: { risk_score: 42.35, risk_level: "Medium", ... }
```

**Python Example:**
```python
import requests

url = "https://debtsage-api.onrender.com/predict"
payload = {
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
}

response = requests.post(url, json=payload)
print(response.json())
```

---

### 2. Batch Predictions

**Endpoint:** `POST /predict/batch`

**Description:** Submit multiple predictions in one request.

**Request Body:**
```json
{
  "predictions": [
    {
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
    },
    {
      "debt_to_gdp": 85.0,
      "deficit_to_gdp": -5.5,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "risk_score": 42.35,
      "risk_level": "Medium",
      ...
    },
    {
      "risk_score": 78.92,
      "risk_level": "High",
      ...
    }
  ],
  "count": 2
}
```

---

### 3. Get Countries List

**Endpoint:** `GET /countries`

**Response:**
```json
{
  "countries": [
    "Benin",
    "Cameroon",
    "Egypt",
    "Ghana",
    "Ivory Coast",
    "Kenya",
    "Nigeria",
    "Senegal",
    "South Africa",
    "Tanzania",
    "Togo",
    "Uganda",
    "Zambia",
    "Zimbabwe"
  ],
  "count": 14
}
```

---

### 4. Get Country Historical Data

**Endpoint:** `GET /country/{country}`

**Query Parameters:**
- `start_year` (optional): Filter from this year
- `end_year` (optional): Filter to this year

**Example:** `GET https://debtsage-api.onrender.com/country/Togo?start_year=2015&end_year=2023`

**Response:**
```json
{
  "country": "Togo",
  "records": [
    {
      "country": "Togo",
      "year": 2015,
      "debt_to_gdp": 62.3,
      "deficit_to_gdp": -2.1,
      "revenue_to_gdp": 19.4,
      ...
    },
    ...
  ],
  "count": 9,
  "year_range": {
    "min": 2015,
    "max": 2023
  }
}
```

---

### 5. Get Risk Scores

**Endpoint:** `GET /risk-scores`

**Query Parameters:**
- `country` (optional): Filter by country name
- `min_risk` (optional): Minimum risk score (0-100)

**Example:** `GET https://debtsage-api.onrender.com/risk-scores?min_risk=50`

**Response:**
```json
{
  "risk_scores": [
    {
      "country": "Nigeria",
      "year": 2023,
      "risk_score": 98.5,
      "risk_prediction": 1
    },
    {
      "country": "Egypt",
      "year": 2023,
      "risk_score": 98.2,
      "risk_prediction": 1
    },
    ...
  ],
  "count": 5
}
```

---

### 6. Get Model Performance

**Endpoint:** `GET /model/performance`

**Response:**
```json
{
  "models": [
    {
      "model": "XGBoost",
      "metrics": [
        {
          "model": "XGBoost",
          "dataset": "Test",
          "auc_roc": 0.934,
          "precision": 0.906,
          "recall": 0.784,
          "f1_score": 0.841
        },
        ...
      ]
    },
    ...
  ],
  "count": 3
}
```

---

### 7. Get Feature Importance

**Endpoint:** `GET /model/feature-importance`

**Query Parameters:**
- `top_n` (optional): Return only top N features

**Example:** `GET https://debtsage-api.onrender.com/model/feature-importance?top_n=5`

**Response:**
```json
{
  "features": [
    {
      "feature": "revenue_to_gdp_mean",
      "importance": 0.484
    },
    {
      "feature": "inflation_rate",
      "importance": 0.061
    },
    ...
  ],
  "count": 5
}
```

---

### 8. Get Fiscal Metrics

**Endpoint:** `GET /fiscal/metrics`

**Query Parameters:**
- `country` (optional): Filter by country

**Example:** `GET https://debtsage-api.onrender.com/fiscal/metrics?country=Togo`

**Response:**
```json
{
  "fiscal_metrics": [
    {
      "country": "Togo",
      "year": 2023,
      "debt_to_gdp": 66.5,
      "deficit_to_gdp": 0.0,
      "revenue_to_gdp": 21.3,
      "expenditure_to_gdp": 21.3,
      ...
    },
    ...
  ],
  "count": 45
}
```

---

### 9. Get Fiscal Scorecard

**Endpoint:** `GET /fiscal/scorecard`

**Response:**
```json
{
  "scorecard": [
    {
      "country": "Nigeria",
      "debt_to_gdp": 28557.18,
      "deficit_to_gdp": -3572.24,
      "risk_score": 98.25,
      "fiscal_stress_score": 84.7
    },
    ...
  ],
  "count": 6
}
```

---

### 10. Get Debt Projections

**Endpoint:** `GET /projections/{country}`

**Example:** `GET https://debtsage-api.onrender.com/projections/Togo`

**Response:**
```json
{
  "country": "Togo",
  "scenarios": {
    "baseline": [
      {
        "country": "Togo",
        "scenario": "baseline",
        "year": 2026,
        "debt_to_gdp": 50.4
      },
      ...
    ],
    "optimistic": [...],
    "stress": [...]
  },
  "years": [2026, 2027, 2028, 2029, 2030]
}
```

---

### 11. Get Summary Statistics

**Endpoint:** `GET /stats/summary`

**Response:**
```json
{
  "dataset": {
    "total_observations": 623,
    "countries": 14,
    "year_range": {
      "min": 1960,
      "max": 2025
    }
  },
  "risk_distribution": {
    "high_risk": 5,
    "medium_risk": 4,
    "low_risk": 5,
    "average_risk": 42.8
  },
  "fiscal_indicators": {
    "avg_debt_to_gdp": 52.3,
    "avg_deficit_to_gdp": -3.2,
    "avg_revenue_to_gdp": 18.7
  }
}
```

---

## Integration Examples

### React/Next.js Integration

```javascript
// lib/debtsageApi.js
const API_BASE_URL = 'https://debtsage-api.onrender.com';

export async function predictRisk(indicators) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(indicators),
  });
  
  if (!response.ok) {
    throw new Error('Prediction failed');
  }
  
  return response.json();
}

export async function getCountries() {
  const response = await fetch(`${API_BASE_URL}/countries`);
  return response.json();
}

export async function getCountryData(country, startYear, endYear) {
  const params = new URLSearchParams();
  if (startYear) params.append('start_year', startYear);
  if (endYear) params.append('end_year', endYear);
  
  const response = await fetch(
    `${API_BASE_URL}/country/${country}?${params}`
  );
  return response.json();
}

// Usage in component
import { predictRisk } from '@/lib/debtsageApi';

function PredictionForm() {
  const [result, setResult] = useState(null);
  
  const handleSubmit = async (formData) => {
    const prediction = await predictRisk({
      debt_to_gdp: formData.debt,
      deficit_to_gdp: formData.deficit,
      // ... other fields
      model: 'xgboost'
    });
    
    setResult(prediction);
  };
  
  return (
    <div>
      {result && (
        <div className={`risk-${result.risk_level.toLowerCase()}`}>
          <h3>Risk Score: {result.risk_score}%</h3>
          <p>Level: {result.risk_level}</p>
          <p>Confidence: {result.confidence}%</p>
        </div>
      )}
    </div>
  );
}
```

### Vue.js Integration

```javascript
// services/debtsageApi.js
import axios from 'axios';

const API_BASE_URL = 'https://debtsage-api.onrender.com';

export default {
  async predictRisk(indicators) {
    const { data } = await axios.post(`${API_BASE_URL}/predict`, indicators);
    return data;
  },
  
  async getCountries() {
    const { data } = await axios.get(`${API_BASE_URL}/countries`);
    return data;
  },
  
  async getRiskScores(minRisk = null) {
    const params = minRisk ? { min_risk: minRisk } : {};
    const { data } = await axios.get(`${API_BASE_URL}/risk-scores`, { params });
    return data;
  }
};

// Usage in component
<script setup>
import { ref } from 'vue';
import debtsageApi from '@/services/debtsageApi';

const riskScore = ref(null);

async function calculateRisk(formData) {
  riskScore.value = await debtsageApi.predictRisk({
    debt_to_gdp: formData.debt,
    deficit_to_gdp: formData.deficit,
    // ... other fields
    model: 'xgboost'
  });
}
</script>

<template>
  <div v-if="riskScore" :class="`risk-${riskScore.risk_level.toLowerCase()}`">
    <h3>Risk Score: {{ riskScore.risk_score }}%</h3>
    <p>Level: {{ riskScore.risk_level }}</p>
  </div>
</template>
```

### Angular Integration

```typescript
// services/debtsage-api.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

interface PredictionRequest {
  debt_to_gdp: number;
  deficit_to_gdp: number;
  // ... other fields
  model: string;
}

interface PredictionResponse {
  risk_score: number;
  risk_level: string;
  risk_prediction: number;
  confidence: number;
  model_used: string;
}

@Injectable({
  providedIn: 'root'
})
export class DebtsageApiService {
  private apiUrl = 'https://debtsage-api.onrender.com';

  constructor(private http: HttpClient) {}

  predictRisk(indicators: PredictionRequest): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(
      `${this.apiUrl}/predict`,
      indicators
    );
  }

  getCountries(): Observable<any> {
    return this.http.get(`${this.apiUrl}/countries`);
  }

  getCountryData(country: string, startYear?: number, endYear?: number): Observable<any> {
    let params: any = {};
    if (startYear) params.start_year = startYear;
    if (endYear) params.end_year = endYear;
    
    return this.http.get(`${this.apiUrl}/country/${country}`, { params });
  }
}

// Usage in component
export class PredictionComponent {
  constructor(private api: DebtsageApiService) {}

  calculateRisk(formData: any) {
    this.api.predictRisk({
      debt_to_gdp: formData.debt,
      deficit_to_gdp: formData.deficit,
      // ... other fields
      model: 'xgboost'
    }).subscribe(result => {
      console.log('Risk Score:', result.risk_score);
      console.log('Risk Level:', result.risk_level);
    });
  }
}
```

---

## Error Handling

All endpoints return standard HTTP status codes:

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found (country/resource not found) |
| 500 | Internal Server Error |

**Error Response Format:**
```json
{
  "detail": "Error message here"
}
```

**Example Error Handling:**
```javascript
try {
  const response = await fetch('https://debtsage-api.onrender.com/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(indicators)
  });
  
  if (!response.ok) {
    const error = await response.json();
    console.error('API Error:', error.detail);
    throw new Error(error.detail);
  }
  
  const data = await response.json();
  return data;
  
} catch (error) {
  console.error('Network Error:', error);
  throw error;
}
```

---

## CORS Configuration

The API is configured to allow requests from any origin during development:

```python
allow_origins=["*"]  # Accept all origins
```

**For production**, update CORS settings in `app/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://dashboard.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Testing the API

### Using cURL

```bash
# Health check
curl https://debtsage-api.onrender.com/health

# Get countries
curl https://debtsage-api.onrender.com/countries

# Make prediction
curl -X POST https://debtsage-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"debt_to_gdp": 65, "deficit_to_gdp": -3.5, ..., "model": "xgboost"}'

# Get country data
curl "https://debtsage-api.onrender.com/country/Togo?start_year=2020"
```

### Using Python

```python
import requests

# Get risk scores
response = requests.get('https://debtsage-api.onrender.com/risk-scores')
data = response.json()
print(data)

# Make prediction
response = requests.post(
    'https://debtsage-api.onrender.com/predict',
    json={
        'debt_to_gdp': 65.0,
        'deficit_to_gdp': -3.5,
        'revenue_to_gdp': 18.5,
        'inflation_rate': 5.2,
        'gdp_growth': 3.8,
        'external_debt_ratio': 45.0,
        'debt_service_to_revenue': 25.0,
        'reserves_months': 4.5,
        'primary_balance': -1.2,
        'exchange_rate_change': 2.1,
        'model': 'xgboost'
    }
)
print(response.json())
```

### Using Postman

1. Import API collection: Use OpenAPI spec from https://debtsage-api.onrender.com/openapi.json
2. Set base URL: https://debtsage-api.onrender.com
3. Test endpoints with sample requests

---

## Performance Notes

- **Startup Time:** ~2-3 seconds (models loaded on startup)
- **Prediction Latency:** <100ms for single prediction
- **Batch Processing:** <500ms for 10 predictions
- **Concurrent Requests:** Supports multiple simultaneous requests
- **Caching:** Models and data cached in memory for fast access

---

## Production Deployment

### Using Docker

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt app/api_requirements.txt ./
RUN pip install -r requirements.txt -r api_requirements.txt

COPY . .

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t debtsage-api .
docker run -p 8000:8000 debtsage-api
```

### Using Gunicorn (Production Server)

```bash
# Install gunicorn
pip install gunicorn

# Run with workers
gunicorn app.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Support & Contact

- **Documentation:** https://debtsage-api.onrender.com/docs
- **GitHub:** github.com/Techdee1/10Analytics
- **Production API:** https://debtsage-api.onrender.com
- **Issues:** Create issue on GitHub repository

---

**Last Updated:** November 29, 2025  
**Version:** 1.0.0
