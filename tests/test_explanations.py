from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain import local_reason_card


def test_reason_card_groups_related_windows() -> None:
    sample = pd.Series(
        {
            "vibration_rms": 2.0,
            "vibration_rms_mean_last_15_cycles": 1.8,
            "sensor_temperature": 90.0,
            "fault_code_count_last_10_cycles": 2,
        }
    )
    card = local_reason_card(
        sample,
        np.array([1.0, 2.0, 2.0, 1.0]),
        sample.index.tolist(),
        0.9,
        0.5,
        "C1",
    )
    assert card["top_reasons"][0]["reason_code"] == "HIGH_VIBRATION"
    assert card["top_reasons"][0]["raw_score_contribution"] == 3.0
    assert len(card["top_reasons"][0]["member_features"]) == 2


def test_non_engine_card_excludes_not_applicable_egt() -> None:
    sample = pd.Series(
        {
            "component_family": "HYDRAULIC_PUMP",
            "egt_margin": np.nan,
            "sensor_pressure": 150.0,
            "vibration_rms": 2.0,
            "fault_code_count_last_10_cycles": 2,
        }
    )
    card = local_reason_card(
        sample,
        np.array([0.0, 100.0, 2.0, 1.0, 3.0]),
        sample.index.tolist(),
        0.9,
        0.5,
        "C2",
    )
    assert all(not reason["feature"].startswith("egt_margin") for reason in card["top_reasons"])
