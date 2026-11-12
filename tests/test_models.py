from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain import attach_alert_reason_codes, lightgbm_raw_explanations, local_reason_card
from src.models import (
    calibrate_frozen_model,
    fit_lightgbm_categories,
    make_lightgbm,
    make_logistic_pipeline,
    transform_lightgbm_features,
)


def _frame(n: int, offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(42 + offset)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "category": np.where(x > 0, "A", "B"),
            "target": (x + rng.normal(scale=0.2, size=n) > 0).astype(int),
        }
    )


def test_logistic_and_lightgbm_calibration_and_raw_shap() -> None:
    train = _frame(200)
    calibration = _frame(100, 1)
    logistic = make_logistic_pipeline(["x"], ["category"])
    logistic.fit(train[["x", "category"]], train["target"])
    assert logistic.predict_proba(calibration[["x", "category"]]).shape == (100, 2)

    maps = fit_lightgbm_categories(train, ["category"])
    train_features = transform_lightgbm_features(train, ["x"], ["category"], maps)
    calibration_features = transform_lightgbm_features(calibration, ["x"], ["category"], maps)
    model = make_lightgbm(n_estimators=30, n_threads=1)
    model.fit(train_features, train["target"], categorical_feature=["category"])
    calibrator = calibrate_frozen_model(model, calibration_features, calibration["target"])
    risk = calibrator.predict_proba(calibration_features)[:, 1]
    assert np.all((risk >= 0) & (risk <= 1))

    explanation = lightgbm_raw_explanations(model, train_features.iloc[:50], calibration_features.iloc[:5])
    assert explanation["additivity_max_error"] < 1e-5
    card = local_reason_card(
        calibration_features.iloc[0],
        explanation["values"][0],
        calibration_features.columns.tolist(),
        risk[0],
        0.5,
        "C1",
    )
    assert len(card["top_reasons"]) == 3
    assert "log-odds" in card["raw_score_semantics"]

    decisions = calibration.iloc[:2].copy()
    decisions["final_alert"] = True
    decisions["risk_score"] = risk[:2]
    decisions["component_id"] = ["C1", "C2"]
    bundle = {
        "model_family": "lightgbm",
        "model": model,
        "numeric_columns": ["x"],
        "categorical_columns": ["category"],
        "category_maps": maps,
    }
    with_codes = attach_alert_reason_codes(decisions, bundle, 0.5)
    assert with_codes["reason_codes"].str.len().gt(0).all()
