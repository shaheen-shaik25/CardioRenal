"""
tests/test_app.py
------------------
Automated test suite for CardioPulse AI Enterprise System (Standard Unittest).
Verifies explainability calculations, database persistence, FHIR parser, and REST API endpoints.
"""

import unittest
import json
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient
from app import app, get_bundle
from src.database import init_db, save_assessment, get_history, get_statistics, get_history_by_patient
from src.fhir import parse_fhir_bundle, generate_fhir_diagnostic_report
from src.explainability import ClinicalExplainer, get_clinical_recommendation


class TestCardioPulseAI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "CardioPulse AI CDSS")
        self.assertEqual(data["framework"], "FastAPI")

    def test_explainability(self):
        bundle = get_bundle()
        explainer = ClinicalExplainer(bundle)
        sample_patient = {
            "age": 60, "sex": 1, "cp": 0, "trestbps": 140, "chol": 240,
            "fbs": 0, "restecg": 0, "thalach": 145, "exang": 1, "oldpeak": 2.0,
            "slope": 1, "ca": 1, "thal": 2
        }
        factors = explainer.explain_patient(sample_patient)
        self.assertEqual(len(factors), 13)
        self.assertIn("feature", factors[0])
        self.assertIn("impact", factors[0])
        self.assertIn("interpretation", factors[0])

        sev_crit, rec_crit = get_clinical_recommendation(80.0)
        self.assertEqual(sev_crit, "CRITICAL")
        self.assertIn("Immediate cardiology consultation", rec_crit)

        sev_mod, rec_mod = get_clinical_recommendation(60.0)
        self.assertEqual(sev_mod, "MODERATE")
        self.assertIn("Follow-up diagnostic testing", rec_mod)

        sev_low, rec_low = get_clinical_recommendation(30.0)
        self.assertEqual(sev_low, "LOW")
        self.assertIn("Routine preventive follow-up", rec_low)

    def test_database_operations(self):
        init_db()
        rec_id = save_assessment(
            patient_id="TEST-PAT-001",
            risk_score=78.5,
            confidence=92.0,
            severity="CRITICAL",
            recommendation="Immediate cardiology consultation required.",
            top_factors=[{"feature": "oldpeak", "impact": 0.45}],
            input_features={"age": 60}
        )
        self.assertGreater(rec_id, 0)
        history = get_history(limit=5)
        self.assertGreater(len(history), 0)
        patient_history = get_history_by_patient("TEST-PAT-001")
        self.assertGreater(len(patient_history), 0)
        stats = get_statistics()
        self.assertGreater(stats["total_assessments"], 0)

    def test_fhir_parser(self):
        sample_bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "12345",
                        "gender": "male",
                        "birthDate": "1970-01-01"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "85354-9"}]},
                        "valueQuantity": {"value": 142}
                    }
                }
            ]
        }
        extracted = parse_fhir_bundle(sample_bundle)
        self.assertIn("patient_id", extracted)
        self.assertEqual(extracted["trestbps"], 142.0)

        report = generate_fhir_diagnostic_report("PAT-12345", {"risk_score": 75, "severity": "CRITICAL", "recommendation": "Urgent ECG"})
        self.assertEqual(report["resourceType"], "DiagnosticReport")

    def test_predict_api(self):
        payload = {
            "patient_id": "PAT-TEST-API",
            "age": 58, "sex": 1, "cp": 0, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
            "slope": 0, "ca": 0, "thal": 1
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("risk_score", data)
        self.assertIn("severity", data)
        self.assertIn("top_factors", data)
        self.assertGreater(len(data["top_factors"]), 0)

    def test_statistics_api(self):
        response = self.client.get("/api/v1/statistics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_assessments", data)

    def test_history_api(self):
        response = self.client.get("/api/v1/history?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("history", data)

    def test_patient_history_api(self):
        response = self.client.get("/api/v1/history/PAT-TEST-API")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("history", data)

    def test_fhir_ingest_api(self):
        fhir_sample = {
            "resourceType": "Patient",
            "id": "FHIR-PAT-009",
            "gender": "female",
            "birthDate": "1980-03-25"
        }
        response = self.client.post("/api/v1/fhir-ingest", json=fhir_sample)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("extracted_clinical_metrics", data)
        self.assertIn("assessment", data)
        self.assertIn("fhir_diagnostic_report", data)


if __name__ == "__main__":
    unittest.main()
