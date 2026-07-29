"""
train.py
--------
Trains and compares several supervised ML models for heart disease
prediction, exactly as outlined in the project's "Proposed System" and
"Project Objectives" sections:

  - Logistic Regression
  - Random Forest  (primary model, per the report's abstract)
  - Support Vector Machine
  - K-Nearest Neighbors
  - Gradient Boosting

For each model we run cross-validated hyperparameter tuning, then evaluate
on a held-out test set using accuracy, precision, recall, F1, and ROC-AUC.
The best model (by ROC-AUC) is saved to disk along with the scaler/imputer
needed to run it on new patient data, plus explainability artifacts
(feature importances) and evaluation plots.
"""

import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "strings"):
    try:
        import numpy.core.defchararray as _char
        setattr(np, "strings", _char)
        sys.modules["numpy.strings"] = _char
    except Exception:
        pass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from src.preprocess import prepare
except ImportError:
    from preprocess import prepare

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def get_model_grid():
    """Returns model estimators paired with a small hyperparameter grid each.
    Grids are kept modest so the whole thing trains in seconds, but cover the
    parameters that matter most for each algorithm."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6, 10, None],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "svm": (
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 7, 9, 11, 15]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    data = prepare("data/heart.csv")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    fitted_models = {}

    for name, (estimator, grid) in get_model_grid().items():
        print(f"\nTraining {name} ...")
        t0 = time.time()
        search = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=1)
        search.fit(data.X_train, data.y_train)
        best_model = search.best_estimator_
        metrics = evaluate(best_model, data.X_test, data.y_test)
        metrics["best_params"] = search.best_params_
        metrics["cv_roc_auc_mean"] = search.best_score_
        metrics["train_time_sec"] = round(time.time() - t0, 2)
        results[name] = metrics
        fitted_models[name] = best_model
        print(
            f"  best_params={search.best_params_}  "
            f"test_accuracy={metrics['accuracy']:.3f}  "
            f"test_roc_auc={metrics['roc_auc']:.3f}"
        )

    # Pick the best model by test-set ROC-AUC (robust to class imbalance,
    # matches the evaluation metrics called out in the report's objectives)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}  (ROC-AUC={results[best_name]['roc_auc']:.3f})")

    # Persist model + preprocessing artifacts together so predict.py / the web
    # app can reproduce the exact same feature pipeline on new patients.
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "scaler": data.scaler,
            "imputer": data.imputer,
            "feature_names": data.feature_names,
        },
        os.path.join(MODELS_DIR, "heart_disease_model.joblib"),
    )

    with open(os.path.join(REPORTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Plots ----
    plot_model_comparison(results)
    plot_confusion_matrix(results[best_name]["confusion_matrix"], best_name)
    plot_roc_curves(fitted_models, data.X_test, data.y_test)
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model.feature_importances_, data.feature_names, best_name)
    elif hasattr(best_model, "coef_"):
        plot_feature_importance(np.abs(best_model.coef_[0]), data.feature_names, best_name)

    print(f"\nAll artifacts saved to '{MODELS_DIR}/' and '{REPORTS_DIR}/'")
    print(f"Full comparison table: {REPORTS_DIR}/model_comparison.json")


def plot_model_comparison(results):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    aucs = [results[n]["roc_auc"] for n in names]
    f1s = [results[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, accs, width, label="Accuracy")
    ax.bar(x, f1s, width, label="F1-score")
    ax.bar(x + width, aucs, width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, model_name):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No disease", "Disease"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No disease", "Disease"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({model_name})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_roc_curves(fitted_models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC curves - all models")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, model_name):
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[order][::-1], np.array(importances)[order][::-1], color="#2f6fed")
    ax.set_title(f"Feature importance ({model_name})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
