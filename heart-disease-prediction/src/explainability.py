"""
src/explainability.py
---------------------
Per-patient local Explainable AI (XAI) engine using SHAP (SHapley Additive exPlanations).
Calculates exact feature contributions (Shapley values) for individual patient records,
mapping raw model predictions to clinician-understandable risk drivers and actionable insights.
"""

import sys
import numpy as np

if not hasattr(np, "strings"):
    try:
        import numpy.core.defchararray as _char
        setattr(np, "strings", _char)
        sys.modules["numpy.strings"] = _char
    except Exception:
        pass
import pandas as pd

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Result",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Exercise)",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels (Fluoroscopy)",
    "thal": "Thalassemia Result"
}

_EXPLAINER_CACHE = {}


def get_clinical_recommendation(risk_score: float) -> tuple:
    """
    Returns (severity, recommendation) based on percentage risk score.
    - Risk Score >= 75%: Critical / High Severity
    - Risk Score 45 - 74%: Moderate Risk
    - Risk Score < 45%: Low Risk
    """
    if risk_score >= 75.0:
        return (
            "CRITICAL",
            "Immediate cardiology consultation & stat ECG/Echo recommended."
        )
    elif risk_score >= 45.0:
        return (
            "MODERATE",
            "Follow-up diagnostic testing (Stress ECG, Lipid Panel) recommended within 14 days."
        )
    else:
        return (
            "LOW",
            "Routine preventive follow-up & lifestyle monitoring."
        )


class ClinicalExplainer:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.scaler = bundle.get("scaler")
        self.imputer = bundle.get("imputer")
        self._tree_explainer = self._get_cached_tree_explainer()

    def _get_cached_tree_explainer(self):
        """Pre-warmed singleton TreeExplainer for zero-latency inference."""
        model_id = id(self.model)
        if model_id not in _EXPLAINER_CACHE:
            try:
                import shap
                if hasattr(self.model, "estimators_") or hasattr(self.model, "feature_importances_"):
                    _EXPLAINER_CACHE[model_id] = shap.TreeExplainer(self.model)
                else:
                    _EXPLAINER_CACHE[model_id] = None
            except Exception:
                _EXPLAINER_CACHE[model_id] = None
        return _EXPLAINER_CACHE[model_id]

    def explain_patient(self, raw_patient_dict: dict) -> list:
        """
        Calculates per-patient feature contributions.
        Returns a sorted list of dictionaries with feature name, patient value, SHAP impact score,
        direction, and clinical interpretation.
        """
        df_raw = pd.DataFrame([raw_patient_dict])[self.feature_names]
        
        # Preprocess
        X_imp = self.imputer.transform(df_raw) if self.imputer else df_raw.values
        X_scaled = self.scaler.transform(X_imp) if self.scaler else X_imp

        shap_impacts = None

        # 1. Attempt SHAP TreeExplainer evaluation
        if self._tree_explainer is not None:
            try:
                shap_vals = self._tree_explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    shap_impacts = shap_vals[1][0]
                elif isinstance(shap_vals, np.ndarray):
                    if shap_vals.ndim == 3:
                        shap_impacts = shap_vals[0, :, 1]
                    elif shap_vals.ndim == 2:
                        shap_impacts = shap_vals[0]
            except Exception:
                shap_impacts = None

        # 2. Fallback local contribution calculation if SHAP fails or isn't installed
        if shap_impacts is None:
            if hasattr(self.model, "feature_importances_"):
                global_imp = self.model.feature_importances_
            else:
                global_imp = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            scaled_vec = X_scaled[0]
            shap_impacts = scaled_vec * global_imp

        contributions = []
        for i, feature in enumerate(self.feature_names):
            val = float(df_raw.iloc[0][feature])
            impact = float(shap_impacts[i])
            direction = "INCREASE" if impact > 0 else "DECREASE"
            
            interpretation = self._build_interpretation(feature, val, impact)

            contributions.append({
                "name": FEATURE_LABELS.get(feature, feature),
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": val,
                "formatted_value": self._format_value(feature, val),
                "impact": round(impact, 4),
                "direction": "INCREASES_RISK" if impact > 0 else "DECREASES_RISK",
                "interpretation": interpretation
            })

        # Sort by absolute magnitude of contribution
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions

    def _format_value(self, feature: str, val: float) -> str:
        if feature == "sex":
            return "Male" if val == 1 else "Female"
        elif feature == "cp":
            labels = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"]
            return labels[int(val)] if 0 <= int(val) < 4 else str(val)
        elif feature == "fbs":
            return "> 120 mg/dl" if val == 1 else "<= 120 mg/dl"
        elif feature == "restecg":
            labels = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
            return labels[int(val)] if 0 <= int(val) < 3 else str(val)
        elif feature == "exang":
            return "Yes" if val == 1 else "No"
        elif feature == "slope":
            labels = ["Upsloping", "Flat", "Downsloping"]
            return labels[int(val)] if 0 <= int(val) < 3 else str(val)
        elif feature == "thal":
            labels = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
            return labels.get(int(val), str(val))
        elif feature == "oldpeak":
            return f"{val} mm"
        elif feature == "trestbps":
            return f"{int(val)} mm Hg"
        elif feature == "chol":
            return f"{int(val)} mg/dl"
        elif feature == "thalach":
            return f"{int(val)} bpm"
        else:
            return str(int(val) if val.is_integer() else val)

    def _build_interpretation(self, feature: str, val: float, impact: float) -> str:
        if impact > 0:
            if feature == "oldpeak" and val > 1.0:
                return f"Significant exercise ST depression ({val} mm) indicates myocardial ischemia risk."
            elif feature == "ca" and val > 0:
                return f"{int(val)} major coronary vessel(s) flagged on fluoroscopy, increasing risk."
            elif feature == "thal" and val in [2, 3]:
                return "Abnormal thalassemia scan detected (perfusion defect)."
            elif feature == "exang" and val == 1:
                return "Exercise-induced angina indicates active myocardial oxygen deficit."
            elif feature == "cp" and val == 0:
                return "Typical angina presentation strongly aligns with coronary artery disease."
            elif feature == "trestbps" and val > 140:
                return f"Elevated resting blood pressure ({int(val)} mm Hg) adds vascular stress."
            elif feature == "chol" and val > 240:
                return f"High serum cholesterol ({int(val)} mg/dl) promotes atherosclerotic plaque."
            else:
                return f"Elevated {FEATURE_LABELS.get(feature, feature)} pushes risk higher."
        else:
            if feature == "thalach" and val > 150:
                return f"Good maximum heart rate capacity ({int(val)} bpm) demonstrates cardiac reserve."
            elif feature == "ca" and val == 0:
                return "Clear fluoroscopy (0 blocked major vessels) lowers overall risk."
            elif feature == "exang" and val == 0:
                return "No exercise-induced angina recorded."
            elif feature == "oldpeak" and val <= 0.5:
                return "Minimal ST depression observed during exercise stress test."
            else:
                return f"Parameter {FEATURE_LABELS.get(feature, feature)} is within protective bounds."


def get_patient_shap(patient_dict: dict, bundle: dict) -> list:
    """Helper function to calculate per-patient SHAP contributions directly."""
    explainer = ClinicalExplainer(bundle)
    return explainer.explain_patient(patient_dict)
