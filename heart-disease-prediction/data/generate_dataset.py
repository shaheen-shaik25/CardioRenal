"""
generate_dataset.py
--------------------
Generates a clinically-plausible heart disease dataset that follows the exact
schema of the UCI Cleveland Heart Disease dataset (14 attributes) described in
the project report.

WHY A GENERATED DATASET?
This sandbox has no internet access, so the live UCI file cannot be
downloaded. Real medical feature distributions, correlations, and class
balance are reproduced from published Cleveland dataset statistics so the
pipeline behaves the way it would on the real data.

FOR PRODUCTION USE:
Replace data/heart.csv with the real dataset before deploying, e.g.:
  https://archive.ics.uci.edu/dataset/45/heart+disease
The rest of the pipeline (preprocessing, training, evaluation, app) does not
need to change -- it only depends on the column names below.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 920  # similar scale to the combined Cleveland+Hungarian+VA+Swiss sets

rng = np.random.default_rng(RANDOM_SEED)


def generate():
    n = N_SAMPLES

    # ---- Target (heart disease present = 1) ----
    # Base prevalence roughly matches the real Cleveland dataset (~46% positive)
    target = rng.binomial(1, 0.46, size=n)

    # ---- Age: older patients skew towards disease ----
    age = np.where(
        target == 1,
        rng.normal(56, 8, n),
        rng.normal(50, 9, n),
    ).clip(29, 80).round().astype(int)

    # ---- Sex: 1 = male, 0 = female. Disease more prevalent in males in this dataset ----
    sex = np.where(
        target == 1,
        rng.binomial(1, 0.75, n),
        rng.binomial(1, 0.55, n),
    )

    # ---- Chest pain type (cp): 0=typical angina,1=atypical,2=non-anginal,3=asymptomatic ----
    # Asymptomatic (3) is strongly associated with disease
    cp = np.where(
        target == 1,
        rng.choice([0, 1, 2, 3], size=n, p=[0.08, 0.15, 0.20, 0.57]),
        rng.choice([0, 1, 2, 3], size=n, p=[0.20, 0.30, 0.35, 0.15]),
    )

    # ---- Resting blood pressure (trestbps) ----
    trestbps = np.where(
        target == 1,
        rng.normal(134, 18, n),
        rng.normal(129, 16, n),
    ).clip(90, 200).round().astype(int)

    # ---- Serum cholesterol (chol) ----
    chol = np.where(
        target == 1,
        rng.normal(251, 52, n),
        rng.normal(239, 45, n),
    ).clip(120, 570).round().astype(int)

    # ---- Fasting blood sugar > 120 mg/dl (fbs): 1 true, 0 false ----
    fbs = rng.binomial(1, 0.15, n)

    # ---- Resting ECG results (restecg): 0=normal,1=ST-T abnormality,2=LV hypertrophy ----
    restecg = np.where(
        target == 1,
        rng.choice([0, 1, 2], size=n, p=[0.42, 0.05, 0.53]),
        rng.choice([0, 1, 2], size=n, p=[0.60, 0.03, 0.37]),
    )

    # ---- Max heart rate achieved (thalach): lower in disease group ----
    thalach = np.where(
        target == 1,
        rng.normal(139, 22, n),
        rng.normal(158, 19, n),
    ).clip(70, 202).round().astype(int)

    # ---- Exercise induced angina (exang): 1 yes, 0 no ----
    exang = np.where(
        target == 1,
        rng.binomial(1, 0.55, n),
        rng.binomial(1, 0.14, n),
    )

    # ---- ST depression induced by exercise (oldpeak) ----
    oldpeak = np.where(
        target == 1,
        rng.gamma(2.0, 0.9, n),
        rng.gamma(1.1, 0.6, n),
    ).clip(0, 6.2).round(1)

    # ---- Slope of peak exercise ST segment: 0=up,1=flat,2=down ----
    slope = np.where(
        target == 1,
        rng.choice([0, 1, 2], size=n, p=[0.20, 0.60, 0.20]),
        rng.choice([0, 1, 2], size=n, p=[0.60, 0.32, 0.08]),
    )

    # ---- Number of major vessels colored by fluoroscopy (ca): 0-4 ----
    ca = np.where(
        target == 1,
        rng.choice([0, 1, 2, 3, 4], size=n, p=[0.30, 0.28, 0.20, 0.14, 0.08]),
        rng.choice([0, 1, 2, 3, 4], size=n, p=[0.75, 0.15, 0.06, 0.03, 0.01]),
    )

    # ---- Thalassemia (thal): 1=normal,2=fixed defect,3=reversible defect ----
    thal = np.where(
        target == 1,
        rng.choice([1, 2, 3], size=n, p=[0.25, 0.10, 0.65]),
        rng.choice([1, 2, 3], size=n, p=[0.75, 0.08, 0.17]),
    )

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target,
    })

    # Inject a small, realistic proportion of missing values (as real clinical
    # data almost always has), so the preprocessing pipeline has real work to do.
    for col, frac in [("ca", 0.02), ("thal", 0.01), ("chol", 0.01), ("trestbps", 0.005)]:
        idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Shuffle row order
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/heart.csv", index=False)
    print(f"Generated data/heart.csv with {len(df)} rows and {df.shape[1]} columns")
    print(df["target"].value_counts(normalize=True))
