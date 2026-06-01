"""Train all models, write metrics + figures + the best model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import FIGURES_DIR, METRICS_PATH, MODELS_DIR, REPORTS_DIR
from .data import load_raw, make_splits
from .model import build_models


def _ensure_dirs() -> None:
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _metric_row(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _save_confusion_matrix(y_true, y_pred, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Parkinson's"],
        yticklabels=["Healthy", "Parkinson's"],
        cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def _save_roc_curve(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, payload in results.items():
        fpr, tpr, _ = roc_curve(y_test, payload["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={payload['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — model comparison")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def _save_metric_bar(results: dict) -> None:
    names = list(results.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.16
    for i, k in enumerate(metric_keys):
        vals = [results[n]["metrics"][k] for n in names]
        ax.bar(x + i * width, vals, width, label=k)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison")
    ax.legend(fontsize=9, ncol=5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    _ensure_dirs()

    df = load_raw()
    X_train, X_test, y_train, y_test, scaler = make_splits(df)

    models = build_models()
    results: dict[str, dict] = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        results[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": _metric_row(y_test, y_pred, y_proba),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
        _save_confusion_matrix(y_test, y_pred, name)
        print(f"  {name:22s} acc={results[name]['metrics']['accuracy']:.4f}  "
              f"f1={results[name]['metrics']['f1']:.4f}  "
              f"auc={results[name]['metrics']['roc_auc']:.4f}")

    _save_roc_curve(results, y_test)
    _save_metric_bar(results)

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_name}")

    joblib.dump(models[best_name], MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    Path(MODELS_DIR / "best_model_name.txt").write_text(best_name)

    metrics_out = {
        "best_model": best_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "models": {n: r["metrics"] for n, r in results.items()},
    }
    METRICS_PATH.write_text(json.dumps(metrics_out, indent=2))
    print(f"Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
