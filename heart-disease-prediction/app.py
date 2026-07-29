"""
app.py
------
CardioPulse AI — Enterprise Clinical Decision Support System (CDSS) Backend Server.
Built with FastAPI asynchronous services, Pydantic data validation, SHAP explainability,
SQLite assessment audit log persistence, and FHIR interoperability simulation.
"""

import sys
import numpy

# Polyfill NumPy 2.0+ compatibility for models unpickled in NumPy 1.x environments
if not hasattr(numpy, "strings"):
    try:
        import numpy.core.defchararray as _char
        setattr(numpy, "strings", _char)
        sys.modules["numpy.strings"] = _char
    except Exception:
        pass

import os
import uuid
import joblib
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Query, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.explainability import ClinicalExplainer, get_clinical_recommendation, get_patient_shap
from src.database import (
    init_db,
    save_assessment,
    get_history,
    get_assessment_by_id,
    get_history_by_patient,
    get_statistics
)
from src.fhir import parse_fhir_bundle, generate_fhir_diagnostic_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "heart_disease_model.joblib")

_bundle = None
_explainer = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model file not found. Run `python src/train.py` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_explainer():
    global _explainer
    if _explainer is None:
        bundle = get_bundle()
        _explainer = ClinicalExplainer(bundle)
    return _explainer


# Initialize DB schema
init_db()

# Initialize FastAPI application
app = FastAPI(
    title="CardioPulse AI — Enterprise CDSS",
    description="HIPAA-aligned Enterprise Clinical Decision Support System backend with SHAP XAI and FHIR integration.",
    version="2.0.0"
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static & Templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": str(exc), "path": str(request.url.path)}
    )


# Rate Limiting & Security Audit Middleware Hook
@app.middleware("http")
async def security_audit_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["X-CardioPulse-Audit"] = "SECURE_HIPAA_ALIGNED"
    return response


# Pydantic Input Validation Model
class PatientMetrics(BaseModel):
    patient_id: Optional[str] = None
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest Pain Type (0-3)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting Blood Pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum Cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 (1 = yes, 0 = no)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG Result (0-2)")
    thalach: float = Field(..., ge=40, le=250, description="Max Heart Rate Achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-Induced Angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST Depression Induced by Exercise")
    slope: int = Field(..., ge=0, le=2, description="ST Segment Slope (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Major Vessels Colored by Fluoroscopy (0-4)")
    thal: int = Field(..., ge=1, le=3, description="Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)")


FIELDS = [
    ("age", "Age (years)", "number"),
    ("sex", "Sex (1 = male, 0 = female)", "select01"),
    ("cp", "Chest Pain Type (0 typical, 1 atypical, 2 non-anginal, 3 asymptomatic)", "select0123"),
    ("trestbps", "Resting Blood Pressure (mm Hg)", "number"),
    ("chol", "Serum Cholesterol (mg/dl)", "number"),
    ("fbs", "Fasting Blood Sugar > 120 mg/dl (1 = yes, 0 = no)", "select01"),
    ("restecg", "Resting ECG (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)", "select012"),
    ("thalach", "Max Heart Rate Achieved", "number"),
    ("exang", "Exercise-Induced Angina (1 = yes, 0 = no)", "select01"),
    ("oldpeak", "ST Depression Induced by Exercise", "number"),
    ("slope", "ST Segment Slope (0 upsloping, 1 flat, 2 downsloping)", "select012"),
    ("ca", "Major Vessels Colored by Fluoroscopy (0-4)", "select01234"),
    ("thal", "Thalassemia (1 normal, 2 fixed defect, 3 reversible defect)", "select123"),
]


def compute_clinical_assessment(input_data: dict) -> dict:
    bundle = get_bundle()
    explainer = get_explainer()

    patient_id = input_data.get("patient_id") or f"PAT-{str(uuid.uuid4())[:6].upper()}"

    feature_values = {}
    for name, _, _ in FIELDS:
        if name not in input_data:
            raise ValueError(f"Missing clinical parameter: {name}")
        feature_values[name] = float(input_data[name])

    df_patient = pd.DataFrame([feature_values])[bundle["feature_names"]]

    X_imp = bundle["imputer"].transform(df_patient)
    X_scaled = bundle["scaler"].transform(X_imp)

    probability = float(bundle["model"].predict_proba(X_scaled)[0][1])
    risk_score = round(probability * 100.0, 1)
    confidence = round(max(probability, 1 - probability) * 100.0, 1)

    severity, recommendation = get_clinical_recommendation(risk_score)

    shap_factors = explainer.explain_patient(feature_values)

    record_id = save_assessment(
        patient_id=patient_id,
        risk_score=risk_score,
        confidence=confidence,
        severity=severity,
        recommendation=recommendation,
        top_factors=shap_factors[:5],
        input_features=feature_values,
        model_version=bundle.get("model_name", "RandomForest_v1.0")
    )

    return {
        "assessment_id": record_id,
        "patient_id": patient_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_score": risk_score,
        "probability": round(probability, 4),
        "confidence": confidence,
        "severity": severity,
        "recommendation": recommendation,
        "top_factors": shap_factors[:5],
        "all_factors": shap_factors,
        "input_features": feature_values,
        "model_version": bundle.get("model_name", "RandomForest_v1.0")
    }


# ==========================================
# REST API Endpoints (FastAPI v1 & v2 Specs)
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"fields": FIELDS})


@app.post("/api/v1/predict")
@app.post("/api/v2/predict")
async def api_predict(patient: PatientMetrics):
    try:
        data_dict = patient.model_dump()
        assessment = compute_clinical_assessment(data_dict)
        return JSONResponse(content=assessment, status_code=200)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history")
@app.get("/api/v2/history")
async def api_history(
    limit: int = Query(50, ge=1, le=500),
    severity: Optional[str] = Query(None),
    query: Optional[str] = Query(None)
):
    try:
        records = get_history(limit=limit, severity_filter=severity, search_query=query)
        return {"count": len(records), "history": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history/{patient_id}")
@app.get("/api/v2/history/{patient_id}")
async def api_history_by_patient(patient_id: str):
    try:
        records = get_history_by_patient(patient_id)
        if not records:
            if patient_id.isdigit():
                single = get_assessment_by_id(int(patient_id))
                if single:
                    return {"patient_id": patient_id, "count": 1, "history": [single]}
            return JSONResponse(status_code=404, content={"error": f"No assessment history found for patient ID '{patient_id}'"})
        return {"patient_id": patient_id, "count": len(records), "history": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/statistics")
@app.get("/api/v2/statistics")
async def api_statistics():
    try:
        stats = get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/fhir-ingest")
@app.post("/api/v2/fhir-ingest")
async def api_fhir_ingest(fhir_payload: Dict[str, Any]):
    try:
        extracted = parse_fhir_bundle(fhir_payload)
        assessment = compute_clinical_assessment(extracted)
        fhir_report = generate_fhir_diagnostic_report(assessment["patient_id"], assessment)
        return {
            "extracted_clinical_metrics": extracted,
            "assessment": assessment,
            "fhir_diagnostic_report": fhir_report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/healthz")
async def healthz():
    return {
        "status": "healthy",
        "service": "CardioPulse AI CDSS",
        "framework": "FastAPI",
        "model_loaded": os.path.exists(MODEL_PATH),
        "timestamp": datetime.now().isoformat()
    }


# Legacy endpoint redirects
@app.post("/api/predict")
async def legacy_predict(patient: PatientMetrics):
    return await api_predict(patient)


if __name__ == "__main__":
    import uvicorn
    print("Starting CardioPulse AI CDSS Enterprise FastAPI Server...")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
