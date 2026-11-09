from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_data
from .evaluation import evaluate, save_outputs
from .modeling import train_baseline, train_with_smote
from .preprocessing import prepare_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Klasifikasi fraud transaksi kartu kredit menggunakan Random Forest, "
            "dengan perbandingan baseline vs SMOTE."
        )
    )
    parser.add_argument(
        "--data",
        default="creditcard.csv",
        help="Path ke dataset creditcard.csv (default: creditcard.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Direktori untuk menyimpan metrik dan plot (default: outputs)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data test (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed untuk split, SMOTE, dan model (default: 42)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    df = load_data(args.data)
    print(f"Dataset : {args.data}")
    print(f"Shape   : {df.shape[0]:,} baris x {df.shape[1]} kolom")

    counts = df["Class"].value_counts().sort_index()
    print("Distribusi kelas:")
    for label, count in counts.items():
        name = "Fraud" if int(label) == 1 else "No Fraud"
        print(f"  {name:<8}: {count:>7,} ({count / len(df):.4%})")

    split = prepare_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(
        f"\nTrain   : {len(split.X_train):,} baris | "
        f"Test: {len(split.X_test):,} baris"
    )
    print(
        f"Fraud di test: {int(split.y_test.sum()):,} "
        f"({split.y_test.mean():.4%})"
    )

    print("\nMelatih baseline (tanpa SMOTE)...")
    baseline = train_baseline(
        split.X_train,
        split.y_train,
        random_state=args.random_state,
    )
    print(f"  Distribusi train: {baseline.class_distribution_before}")

    print("Melatih model dengan SMOTE...")
    smote = train_with_smote(
        split.X_train,
        split.y_train,
        random_state=args.random_state,
    )
    print(f"  Sebelum SMOTE   : {smote.class_distribution_before}")
    print(f"  Sesudah SMOTE   : {smote.class_distribution_after}")

    results = [
        evaluate("Baseline", baseline.model, split.X_test, split.y_test),
        evaluate("SMOTE", smote.model, split.X_test, split.y_test),
    ]

    save_outputs(results, split.y_test, args.output_dir)

    header = (
        f"{'Skenario':<10} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'AUPRC':>8}"
    )
    print("\n=== Ringkasan evaluasi pada test set ===")
    print(header)
    print("-" * len(header))
    for result in results:
        metrics = result.metrics
        print(
            f"{result.name:<10} {metrics['accuracy']:>9.4f} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f} {metrics['roc_auc']:>9.4f} "
            f"{metrics['average_precision']:>8.4f}"
        )

    print(f"\nOutput disimpan di: {Path(args.output_dir).resolve()}")
    print(
        "Metrik utama: AUPRC (average precision), karena kelas fraud sangat imbalanced."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
