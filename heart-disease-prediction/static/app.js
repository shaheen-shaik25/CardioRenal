/**
 * static/app.js
 * CardioPulse AI -- Enterprise System Dashboard Frontend Interactivity
 */

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    const form = document.getElementById("risk-form");
    if (form) {
        form.addEventListener("submit", handleRiskFormSubmit);
    }

    fetchStatistics();
    fetchHistory();
}

// Preset Patient Data
const PRESETS = {
    high: {
        patient_id: "PAT-6284-HIGH",
        age: 62, sex: 1, cp: 0, trestbps: 155, chol: 280, fbs: 1,
        restecg: 1, thalach: 122, exang: 1, oldpeak: 2.8, slope: 1, ca: 2, thal: 3
    },
    low: {
        patient_id: "PAT-1049-LOW",
        age: 45, sex: 0, cp: 2, trestbps: 118, chol: 195, fbs: 0,
        restecg: 0, thalach: 172, exang: 0, oldpeak: 0.2, slope: 0, ca: 0, thal: 1
    },
    moderate: {
        patient_id: "PAT-3391-MOD",
        age: 55, sex: 1, cp: 1, trestbps: 138, chol: 245, fbs: 0,
        restecg: 0, thalach: 142, exang: 0, oldpeak: 1.2, slope: 1, ca: 1, thal: 2
    }
};

function loadPreset(type) {
    const data = PRESETS[type];
    if (!data) return;

    for (const key in data) {
        const input = document.getElementById(`input-${key}`);
        if (input) {
            input.value = data[key];
        }
    }
}

async function handleRiskFormSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const payload = {};

    formData.forEach((val, key) => {
        if (key === "patient_id") {
            payload[key] = val;
        } else {
            payload[key] = isNaN(val) ? val : parseFloat(val);
        }
    });

    const submitBtn = document.getElementById("btn-assess");
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerText = "Processing AI Clinical Inference...";
    }

    try {
        const response = await fetch("/api/v1/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || err.error || "Failed to calculate prediction");
        }

        const data = await response.json();
        renderAssessmentResults(data);
        fetchStatistics();
        fetchHistory();

    } catch (err) {
        alert("Error running assessment: " + err.message);
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerText = "⚡ Assess Cardiac Risk & Explain (SHAP)";
        }
    }
}

function renderAssessmentResults(data) {
    const resultsCard = document.getElementById("results-card");
    if (resultsCard) {
        resultsCard.style.display = "block";
    }

    // Gauge Score & Color
    const scoreElem = document.getElementById("res-score");
    const gaugeCircle = document.getElementById("gauge-circle");
    const severityBadge = document.getElementById("res-severity");
    const confElem = document.getElementById("res-confidence");
    const recElem = document.getElementById("res-recommendation");
    const pidElem = document.getElementById("res-patient-id");

    if (scoreElem) scoreElem.innerText = `${data.risk_score}%`;
    if (pidElem) pidElem.innerText = `Patient ID: ${data.patient_id}`;
    if (confElem) confElem.innerText = `Model Certainty: ${data.confidence}% (${data.model_version})`;
    if (recElem) recElem.innerText = data.recommendation;

    // Severity Colors
    let colorHex = "#10b981";
    if (data.severity === "CRITICAL") colorHex = "#ef4444";
    else if (data.severity === "HIGH") colorHex = "#f97316";
    else if (data.severity === "MODERATE") colorHex = "#eab308";

    if (gaugeCircle) {
        gaugeCircle.style.setProperty("--gauge-color", colorHex);
        gaugeCircle.style.setProperty("--gauge-value", data.risk_score);
    }

    if (severityBadge) {
        severityBadge.innerText = `${data.severity} RISK`;
        severityBadge.className = `badge-severity badge-${data.severity}`;
    }

    // Render SHAP Factor Bars
    const shapContainer = document.getElementById("shap-factors-list");
    if (shapContainer && data.top_factors) {
        shapContainer.innerHTML = "";
        data.top_factors.forEach(factor => {
            const pctWidth = Math.min(Math.abs(factor.impact) * 200, 100);
            const isIncrease = factor.direction === "INCREASE";

            const item = document.createElement("div");
            item.className = "shap-item";
            item.innerHTML = `
                <div class="shap-header">
                    <span><strong>${factor.label}</strong> (${factor.formatted_value})</span>
                    <span style="color: ${isIncrease ? '#dc2626' : '#059669'}; font-weight: 600;">
                        ${isIncrease ? '+' : '-'}${Math.abs(factor.impact).toFixed(3)} SHAP
                    </span>
                </div>
                <div class="shap-bar-bg">
                    <div class="shap-bar-fill shap-bar-${factor.direction}" style="width: ${pctWidth}%;"></div>
                </div>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 0.35rem;">
                    ${factor.interpretation}
                </div>
            `;
            shapContainer.appendChild(item);
        });
    }

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

async function fetchStatistics() {
    try {
        const resp = await fetch("/api/v1/statistics");
        if (!resp.ok) return;
        const data = await resp.json();

        const statTotal = document.getElementById("stat-total");
        const statAvg = document.getElementById("stat-avg");
        const statHigh = document.getElementById("stat-high");

        if (statTotal) statTotal.innerText = data.total_assessments;
        if (statAvg) statAvg.innerText = `${data.avg_risk_score}%`;
        if (statHigh) statHigh.innerText = `${data.high_risk_ratio}%`;
    } catch (e) {
        console.error("Failed to fetch statistics", e);
    }
}

async function fetchHistory() {
    try {
        const resp = await fetch("/api/v1/history?limit=20");
        if (!resp.ok) return;
        const data = await resp.json();

        const tbody = document.getElementById("history-tbody");
        if (!tbody) return;

        tbody.innerHTML = "";
        if (data.history.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #9ca3af;">No clinical assessments logged yet.</td></tr>`;
            return;
        }

        data.history.forEach(item => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${item.patient_id}</strong></td>
                <td>${item.timestamp.replace("T", " ").substring(0, 16)}</td>
                <td><strong>${item.risk_score}%</strong></td>
                <td><span class="badge-severity badge-${item.severity}" style="padding: 0.2rem 0.6rem; font-size: 0.75rem;">${item.severity}</span></td>
                <td>${item.confidence}%</td>
                <td style="max-width: 250px; font-size: 0.8rem; color: #d1d5db;">${item.recommendation}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (e) {
        console.error("Failed to fetch history", e);
    }
}

function filterHistoryTable() {
    const input = document.getElementById("search-history-input");
    if (!input) return;
    const filter = input.value.toLowerCase();
    const tbody = document.getElementById("history-tbody");
    if (!tbody) return;
    const rows = tbody.getElementsByTagName("tr");
    for (let i = 0; i < rows.length; i++) {
        const text = rows[i].textContent || rows[i].innerText;
        if (text.toLowerCase().indexOf(filter) > -1) {
            rows[i].style.display = "";
        } else {
            rows[i].style.display = "none";
        }
    }
}

async function simulateFhirIngest() {
    const sampleFhirBundle = {
        resourceType: "Bundle",
        entry: [
            {
                resource: {
                    resourceType: "Patient",
                    id: "PAT-FHIR-8842",
                    gender: "male",
                    birthDate: "1966-05-12"
                }
            },
            {
                resource: {
                    resourceType: "Observation",
                    code: { coding: [{ code: "85354-9", display: "Blood Pressure" }] },
                    valueQuantity: { value: 160 }
                }
            },
            {
                resource: {
                    resourceType: "Observation",
                    code: { coding: [{ code: "2093-3", display: "Cholesterol" }] },
                    valueQuantity: { value: 265 }
                }
            }
        ]
    };

    try {
        const resp = await fetch("/api/v1/fhir-ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(sampleFhirBundle)
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || err.error || "FHIR Ingest failed");
        }
        const data = await resp.json();
        alert("FHIR Bundle Ingested Successfully!\nPatient ID: " + data.assessment.patient_id + "\nExtracted BP: " + data.extracted_clinical_metrics.trestbps + " mm Hg");
        
        // Auto-fill form and render results
        for (const key in data.extracted_clinical_metrics) {
            const input = document.getElementById(`input-${key}`);
            if (input) input.value = data.extracted_clinical_metrics[key];
        }
        renderAssessmentResults(data.assessment);
        fetchHistory();
        fetchStatistics();
    } catch (e) {
        alert("FHIR Ingest failed: " + e.message);
    }
}

function printReport() {
    window.print();
}
