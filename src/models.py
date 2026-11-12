from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def make_logistic_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    class_weight: str | None = "balanced",
    seed: int = 20261112,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)]
    )
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_lightgbm_categories(
    train: pd.DataFrame, categorical_columns: list[str]
) -> dict[str, list[str]]:
    return {
        column: sorted(train[column].fillna("__MISSING__").astype(str).unique().tolist())
        for column in categorical_columns
    }


def transform_lightgbm_features(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    category_maps: dict[str, list[str]],
) -> pd.DataFrame:
    result = frame[numeric_columns].copy()
    for column in categorical_columns:
        values = frame[column].fillna("__MISSING__").astype(str)
        result[f"{column}_is_unknown"] = (~values.isin(category_maps[column])).astype("int8")
        result[column] = pd.Categorical(values, categories=category_maps[column])
    return result


def make_lightgbm(
    scale_pos_weight: float = 1.0,
    seed: int = 20261112,
    n_threads: int = 4,
    **parameters: Any,
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=parameters.pop("n_estimators", 400),
        learning_rate=parameters.pop("learning_rate", 0.04),
        num_leaves=parameters.pop("num_leaves", 31),
        min_child_samples=parameters.pop("min_child_samples", 30),
        subsample=parameters.pop("subsample", 0.9),
        colsample_bytree=parameters.pop("colsample_bytree", 0.9),
        reg_lambda=parameters.pop("reg_lambda", 1.0),
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=n_threads,
        verbosity=-1,
        **parameters,
    )


def calibrate_frozen_model(
    model: Any,
    calibration_features: pd.DataFrame | np.ndarray,
    calibration_target: pd.Series | np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be sigmoid or isotonic")
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrator.fit(calibration_features, calibration_target)
    return calibrator
