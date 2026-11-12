from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def lightgbm_raw_explanations(
    model: Any,
    background: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(samples)
    if isinstance(values, list):
        values = values[-1]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.asarray(expected).reshape(-1)[-1]
    raw_prediction = model.predict(samples, raw_score=True)
    reconstructed = float(expected) + values.sum(axis=1)
    if not np.allclose(reconstructed, raw_prediction, atol=1e-5):
        raise ValueError("SHAP raw-score additivity check failed")
    return {
        "explainer": explainer,
        "values": values,
        "expected_raw_score": float(expected),
        "raw_score_type": "lightgbm_log_odds",
        "background_mode": "tree_path_dependent_node_cover",
        "supplied_background_rows": len(background),
        "additivity_max_error": float(np.max(np.abs(reconstructed - raw_prediction))),
    }


def save_global_shap_summary(values: np.ndarray, samples: pd.DataFrame, output: str | Path) -> None:
    shap.summary_plot(values, samples, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()


def local_reason_card(
    sample: pd.Series,
    shap_values: np.ndarray,
    feature_names: list[str],
    calibrated_risk: float,
    threshold: float,
    component_id: str,
    top_n: int = 3,
) -> dict[str, Any]:
    contributions = pd.Series(shap_values, index=feature_names)
    ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(top_n)
    reasons = []
    for feature, contribution in ordered.items():
        reasons.append(
            {
                "feature": feature,
                "value": sample[feature],
                "direction": "INCREASES_RISK" if contribution > 0 else "DECREASES_RISK",
                "raw_score_contribution": float(contribution),
                "reason_code": f"{'HIGH' if contribution > 0 else 'LOW'}_{feature.upper()}",
                "suggested_action": "Inspect the related subsystem and confirm the sensor/fault evidence before maintenance release.",
            }
        )
    return {
        "component_id": component_id,
        "risk_score": float(calibrated_risk),
        "threshold": float(threshold),
        "raw_score_semantics": "Contributions apply to LightGBM log-odds, not calibrated probability",
        "top_reasons": reasons,
    }
