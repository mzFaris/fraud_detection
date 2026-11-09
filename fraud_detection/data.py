from __future__ import annotations

from pathlib import Path

import pandas as pd

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time", *V_COLUMNS, "Amount", "Class"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Muat dan validasi dataset credit card fraud."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Letakkan creditcard.csv di root repo atau gunakan opsi --data PATH."
        )

    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Dataset mengandung missing value:\n{null_counts}")

    classes = set(df["Class"].unique().tolist())
    if not classes.issubset({0, 1}):
        raise ValueError(f"Kolom Class harus biner 0/1, ditemukan: {sorted(classes)}")
    if len(classes) < 2:
        raise ValueError("Kolom Class hanya berisi satu kelas.")

    return df
