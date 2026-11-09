from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class TrainingResult:
    model: RandomForestClassifier
    n_train_before: int
    n_train_after: int
    class_distribution_before: dict[int, int]
    class_distribution_after: dict[int, int]


def _build_model(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )


def _class_distribution(y: np.ndarray | pd.Series) -> dict[int, int]:
    values, counts = np.unique(y, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def train_baseline(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int = 42,
) -> TrainingResult:
    """Latih Random Forest tanpa penanganan imbalanced data."""
    model = _build_model(random_state)
    model.fit(X_train, y_train)
    distribution = _class_distribution(y_train)
    return TrainingResult(
        model=model,
        n_train_before=len(y_train),
        n_train_after=len(y_train),
        class_distribution_before=distribution,
        class_distribution_after=distribution,
    )


def train_with_smote(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int = 42,
) -> TrainingResult:
    """Latih Random Forest setelah menyeimbangkan data train dengan SMOTE."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = _build_model(random_state)
    model.fit(X_resampled, y_resampled)

    return TrainingResult(
        model=model,
        n_train_before=len(y_train),
        n_train_after=len(y_resampled),
        class_distribution_before=_class_distribution(y_train),
        class_distribution_after=_class_distribution(y_resampled),
    )
