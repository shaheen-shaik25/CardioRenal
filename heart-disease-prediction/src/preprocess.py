"""
preprocess.py
-------------
Loads the raw heart disease CSV and turns it into clean, model-ready arrays.

Steps performed (matching the "Proposed System" section of the project report):
  1. Load data
  2. Basic quality checks (missing values, duplicates)
  3. Impute missing values (median for numeric clinical measurements)
  4. Split into train/test sets (stratified, so class balance is preserved)
  5. Scale numeric features (StandardScaler) -- fit on train only, applied to test
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "target"

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler
    imputer: SimpleImputer


def load_raw(path: str = "data/heart.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_expected = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing_expected:
        raise ValueError(f"Dataset is missing expected columns: {missing_expected}")
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Returns a small data-quality summary, echoing the checks described
    in the report's 'Data Pre-Processing' section (accuracy, completeness,
    consistency)."""
    return {
        "n_rows": len(df),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_values_per_column": df.isna().sum().to_dict(),
        "target_balance": df[TARGET_COL].value_counts(normalize=True).to_dict(),
    }


def prepare(path: str = "data/heart.csv", test_size: float = 0.2, random_state: int = 42) -> PreparedData:
    df = load_raw(path)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Impute missing values using statistics learned ONLY from the training set
    # to avoid data leakage from test set into training.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    # Scale features. Tree ensembles don't strictly need this, but Logistic
    # Regression / SVM do, and a shared scaled representation keeps the
    # comparison across models fair.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return PreparedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=FEATURE_COLS,
        scaler=scaler,
        imputer=imputer,
    )


if __name__ == "__main__":
    df = load_raw()
    report = quality_report(df)
    print("Data quality report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
