"""
predict.py
----------
Loads the trained model bundle and predicts heart disease risk for one or
more patients supplied either as a CSV file or as command-line arguments.

Usage:
    # Predict for every row in a CSV of new patients
    python src/predict.py --csv new_patients.csv

    # Predict for a single patient via flags
    python src/predict.py --age 58 --sex 1 --cp 0 --trestbps 145 --chol 233 \\
        --fbs 1 --restecg 0 --thalach 150 --exang 0 --oldpeak 2.3 \\
        --slope 0 --ca 0 --thal 1
"""

import argparse
import sys
import numpy

if not hasattr(numpy, "strings"):
    try:
        import numpy.core.defchararray as _char
        setattr(numpy, "strings", _char)
        sys.modules["numpy.strings"] = _char
    except Exception:
        pass

import joblib
import pandas as pd

MODEL_PATH = "models/heart_disease_model.joblib"


def load_bundle(path: str = MODEL_PATH):
    return joblib.load(path)


def predict_dataframe(bundle, df: pd.DataFrame) -> pd.DataFrame:
    feats = df[bundle["feature_names"]]
    X = bundle["imputer"].transform(feats)
    X = bundle["scaler"].transform(X)
    proba = bundle["model"].predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["predicted_disease_probability"] = proba.round(4)
    out["predicted_disease"] = pred
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(description="Predict heart disease risk.")
    p.add_argument("--csv", help="Path to a CSV of patients with the 13 feature columns.")
    p.add_argument("--age", type=float)
    p.add_argument("--sex", type=int, choices=[0, 1])
    p.add_argument("--cp", type=int, choices=[0, 1, 2, 3])
    p.add_argument("--trestbps", type=float)
    p.add_argument("--chol", type=float)
    p.add_argument("--fbs", type=int, choices=[0, 1])
    p.add_argument("--restecg", type=int, choices=[0, 1, 2])
    p.add_argument("--thalach", type=float)
    p.add_argument("--exang", type=int, choices=[0, 1])
    p.add_argument("--oldpeak", type=float)
    p.add_argument("--slope", type=int, choices=[0, 1, 2])
    p.add_argument("--ca", type=int, choices=[0, 1, 2, 3, 4])
    p.add_argument("--thal", type=int, choices=[1, 2, 3])
    return p


def main():
    args = build_arg_parser().parse_args()
    bundle = load_bundle()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        required = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        values = {k: getattr(args, k) for k in required}
        if any(v is None for v in values.values()):
            print("Error: either --csv or all patient flags must be provided.", file=sys.stderr)
            sys.exit(1)
        df = pd.DataFrame([values])

    result = predict_dataframe(bundle, df)
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(result)


if __name__ == "__main__":
    main()
