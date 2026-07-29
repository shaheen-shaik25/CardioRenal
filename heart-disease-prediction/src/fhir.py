"""
src/fhir.py
-----------
HL7 / FHIR (Fast Healthcare Interoperability Resources) data parsing and bundle generation module.
Simulates clinical data exchange with hospital Electronic Health Record (EHR) systems like Epic & Cerner.
"""

import uuid
from datetime import datetime


def parse_fhir_bundle(bundle_json: dict) -> dict:
    """
    Parses an incoming FHIR Resource Bundle (Patient, Observation, Condition)
    and extracts standard 13 clinical parameters required by CardioPulse AI.
    """
    extracted = {
        "patient_id": "PAT-" + str(uuid.uuid4())[:8].upper(),
        "age": 55.0,
        "sex": 1,
        "cp": 0,
        "trestbps": 120.0,
        "chol": 200.0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    resource_type = bundle_json.get("resourceType")

    if resource_type == "Bundle":
        entries = bundle_json.get("entry", [])
        for entry in entries:
            res = entry.get("resource", {})
            _extract_from_resource(res, extracted)
    elif resource_type in ["Patient", "Observation", "Condition", "DiagnosticReport"]:
        _extract_from_resource(bundle_json, extracted)

    return extracted


def _extract_from_resource(res: dict, extracted: dict):
    res_type = res.get("resourceType")
    
    if res_type == "Patient":
        if "id" in res:
            extracted["patient_id"] = f"PAT-{res['id']}"
        if "gender" in res:
            extracted["sex"] = 1 if res["gender"].lower() == "male" else 0
        if "birthDate" in res:
            try:
                bdate = datetime.strptime(res["birthDate"], "%Y-%m-%d")
                age = (datetime.now() - bdate).days // 365
                extracted["age"] = float(age)
            except Exception:
                pass

    elif res_type == "Observation":
        code_coding = res.get("code", {}).get("coding", [])
        for coding in code_coding:
            code = coding.get("code")
            # Blood Pressure (LOINC 85354-9 or 8480-6)
            if code in ["85354-9", "8480-6"]:
                val = res.get("valueQuantity", {}).get("value")
                if val:
                    extracted["trestbps"] = float(val)
            # Cholesterol (LOINC 2093-3)
            elif code == "2093-3":
                val = res.get("valueQuantity", {}).get("value")
                if val:
                    extracted["chol"] = float(val)
            # Heart Rate (LOINC 8867-4)
            elif code == "8867-4":
                val = res.get("valueQuantity", {}).get("value")
                if val:
                    extracted["thalach"] = float(val)


def generate_fhir_diagnostic_report(patient_id: str, assessment: dict) -> dict:
    """
    Generates a FHIR R4 DiagnosticReport resource JSON output representing
    the AI assessment for EHR export.
    """
    return {
        "resourceType": "DiagnosticReport",
        "id": f"cardiopulse-report-{assessment.get('id', uuid.uuid4())}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "CARD",
                        "display": "Cardiology"
                    }
                ]
            }
        ],
        "code": {
            "text": "AI Cardiac Risk Assessment Report"
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "effectiveDateTime": datetime.now().isoformat(),
        "conclusion": f"Risk Score: {assessment.get('risk_score')}%, Severity: {assessment.get('severity')}. Recommendation: {assessment.get('recommendation')}",
        "result": [
            {
                "display": f"Confidence: {assessment.get('confidence')}%"
            }
        ]
    }
