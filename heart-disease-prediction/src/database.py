"""
src/database.py
---------------
SQLite database manager for patient risk assessment persistence, clinical audit logging,
history search, patient specific queries, and aggregated system statistics.
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cardiopulse_audit.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite table schema if it does not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                risk_score REAL NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                top_factors_json TEXT NOT NULL,
                input_features_json TEXT NOT NULL,
                model_version TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_id ON patient_assessments(patient_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON patient_assessments(severity);")
        conn.commit()


def save_assessment(patient_id: str, risk_score: float, confidence: float, severity: str,
                    recommendation: str, top_factors: list, input_features: dict,
                    model_version: str = "RandomForest_v1.0") -> int:
    """Saves a completed patient assessment record to the database."""
    init_db()
    now_iso = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patient_assessments 
            (patient_id, timestamp, risk_score, confidence, severity, recommendation, top_factors_json, input_features_json, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            now_iso,
            round(risk_score, 2),
            round(confidence, 2),
            severity,
            recommendation,
            json.dumps(top_factors),
            json.dumps(input_features),
            model_version
        ))
        conn.commit()
        return cursor.lastrowid


def get_history(limit: int = 50, severity_filter: str = None, search_query: str = None) -> list:
    """Retrieves patient assessment history records with optional filtering."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM patient_assessments WHERE 1=1"
        params = []

        if severity_filter and severity_filter != "ALL":
            query += " AND severity = ?"
            params.append(severity_filter)

        if search_query:
            query += " AND (patient_id LIKE ? OR recommendation LIKE ?)"
            params.extend([f"%{search_query}%", f"%{search_query}%"])

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        result = []
        for r in rows:
            record = dict(r)
            record["top_factors"] = json.loads(record["top_factors_json"])
            record["input_features"] = json.loads(record["input_features_json"])
            result.append(record)
        return result


def get_assessment_by_id(assessment_id: int) -> dict:
    """Fetches a single assessment record by primary key ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patient_assessments WHERE id = ?", (assessment_id,))
        row = cursor.fetchone()
        if not row:
            return None
        record = dict(row)
        record["top_factors"] = json.loads(record["top_factors_json"])
        record["input_features"] = json.loads(record["input_features_json"])
        return record


def get_history_by_patient(patient_id: str) -> list:
    """Retrieves all assessment history records for a given patient_id."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patient_assessments WHERE patient_id = ? ORDER BY id DESC", (patient_id,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            record = dict(r)
            record["top_factors"] = json.loads(record["top_factors_json"])
            record["input_features"] = json.loads(record["input_features_json"])
            result.append(record)
        return result


def get_statistics() -> dict:
    """Computes high-level clinical statistics for the system dashboard."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM patient_assessments")
        total = cursor.fetchone()[0]

        if total == 0:
            return {
                "total_assessments": 0,
                "avg_risk_score": 0.0,
                "severity_counts": {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0},
                "high_risk_ratio": 0.0
            }

        cursor.execute("SELECT AVG(risk_score) FROM patient_assessments")
        avg_risk = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT severity, COUNT(*) FROM patient_assessments GROUP BY severity")
        sev_rows = cursor.fetchall()
        sev_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
        for sev, count in sev_rows:
            if sev in sev_counts:
                sev_counts[sev] = count

        high_risk_total = sev_counts["CRITICAL"] + sev_counts["HIGH"]
        high_risk_ratio = round((high_risk_total / total) * 100, 1)

        return {
            "total_assessments": total,
            "avg_risk_score": round(avg_risk, 1),
            "severity_counts": sev_counts,
            "high_risk_ratio": high_risk_ratio
        }
