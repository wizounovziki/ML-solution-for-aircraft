from __future__ import annotations

import numpy as np
import pandas as pd

from src.production import (
    apply_cold_start_fallback,
    fit_part_number_baselines,
    population_stability_index,
)


def test_cold_start_blends_baseline_and_model_score() -> None:
    frame = pd.DataFrame(
        {
            "part_number": ["P1", "P1"],
            "aircraft_age_cycles": [8000, 8000],
            "cycles_since_install": [2, 5],
            "vibration_rms": [1.0, 1.0],
            "sensor_temperature": [70.0, 70.0],
        }
    )
    baselines = fit_part_number_baselines(
        pd.DataFrame({"part_number": ["P1"] * 10, "unscheduled_removal_within_30_cycles": [0, 1] * 5}),
        prior_strength=1.0,
    )
    result = apply_cold_start_fallback(frame, np.array([0.8, 0.8]), baselines)
    assert result.loc[0, "scoring_method"] == "COLD_START_MODEL_BASELINE_BLEND"
    assert result.loc[0, "risk_score"] < 0.8
    assert result.loc[1, "scoring_method"] == "MODEL"
    assert result.loc[1, "risk_score"] == 0.8


def test_psi_detects_distribution_shift_and_ignores_stable_input() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=1000)
    assert population_stability_index(reference, reference) == 0.0
    assert population_stability_index(reference, reference + 2.0) > 0.5
