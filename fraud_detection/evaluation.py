from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    name: str
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, float | list[list[int]]]


def evaluate(name: str, model, X_test, y_test) -> EvaluationResult:
    """Hitung metrik evaluasi pada test set yang tidak pernah dilihat model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return EvaluationResult(
        name=name,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
    )


def _save_confusion_matrices(
    results: list[EvaluationResult],
    output_dir: Path,
) -> None:
    for result in results:
        cm = np.asarray(result.metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No Fraud", "Fraud"],
            yticklabels=["No Fraud", "Fraud"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {result.name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        slug = result.name.lower().replace(" ", "_")
        fig.savefig(
            output_dir / f"confusion_matrix_{slug}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_precision_recall_curve(
    results: list[EvaluationResult],
    y_test: np.ndarray | pd.Series,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        precision, recall, _ = precision_recall_curve(y_test, result.y_proba)
        auprc = result.metrics["average_precision"]
        ax.plot(recall, precision, label=f"{result.name} (AUPRC = {auprc:.4f})")

    prevalence = float(np.mean(y_test))
    ax.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"Prevalensi fraud = {prevalence:.4f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_metric_comparison(
    results: list[EvaluationResult],
    output_dir: Path,
) -> None:
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AUPRC"]

    x = np.arange(len(metric_keys))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    for index, result in enumerate(results):
        values = [result.metrics[key] for key in metric_keys]
        offset = index * width - (len(results) - 1) * width / 2
        bars = ax.bar(x + offset, values, width, label=result.name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Metrik: Baseline vs SMOTE")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "metric_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_outputs(
    results: list[EvaluationResult],
    y_test: np.ndarray | pd.Series,
    output_dir: str | Path,
) -> dict:
    """Simpan metrik, classification report, dan seluruh plot ke disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": {
            "n_test": int(len(y_test)),
            "fraud_test": int(np.sum(y_test)),
            "fraud_rate_test": float(np.mean(y_test)),
        },
        "results": {result.name: result.metrics for result in results},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report_lines = []
    for result in results:
        report_lines.append(f"=== {result.name} ===")
        report_lines.append(
            classification_report(
                y_test,
                result.y_pred,
                target_names=["No Fraud", "Fraud"],
                digits=4,
                zero_division=0,
            )
        )
        report_lines.append("")
    (output_dir / "classification_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    table_lines = [
        "| Skenario | Accuracy | Precision | Recall | F1 | ROC-AUC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result.metrics
        table_lines.append(
            f"| {result.name} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} | "
            f"{metrics['average_precision']:.4f} |"
        )
    (output_dir / "results_table.md").write_text(
        "\n".join(table_lines) + "\n",
        encoding="utf-8",
    )

    _save_confusion_matrices(results, output_dir)
    _save_precision_recall_curve(results, y_test, output_dir)
    _save_metric_comparison(results, output_dir)

    return payload
