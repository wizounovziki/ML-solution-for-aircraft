from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import add_causal_rolling_features
from src.splitter import (
    assert_entity_disjoint,
    build_entity_groups,
    temporal_group_purged_split,
)


def test_label_availability_purges_cross_boundary_training_labels() -> None:
    cycles = np.arange(20)
    frame = pd.DataFrame(
        {
            "flight_date": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(cycles, unit="D"),
            "component_id": "A",
            "aircraft_id": "AC1",
            "component_total_cycles": cycles,
            "label_available_at": pd.Timestamp("2025-01-01", tz="UTC")
            + pd.to_timedelta(cycles + 5, unit="D"),
        }
    )
    train, test = temporal_group_purged_split(
        frame, cutoff_date="2025-01-11", horizon_cycles=30
    )
    assert train["component_total_cycles"].tolist() == [0, 1, 2, 3, 4]
    assert test["component_total_cycles"].min() == 10


def test_cycle_fallback_purges_cycles_not_row_count() -> None:
    cycles = np.arange(0, 41, 2)
    frame = pd.DataFrame(
        {
            "flight_date": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(cycles, unit="D"),
            "component_id": "A",
            "aircraft_id": "AC1",
            "component_total_cycles": cycles,
        }
    )
    train, test = temporal_group_purged_split(
        frame, cutoff_date="2025-01-21", horizon_cycles=10
    )
    assert train["component_total_cycles"].max() == 8
    assert test["component_total_cycles"].min() == 20


def test_entity_disjoint_rejects_component_overlap() -> None:
    left = pd.DataFrame({"component_id": ["A"], "aircraft_id": ["AC1"]})
    right = pd.DataFrame({"component_id": ["A"], "aircraft_id": ["AC2"]})
    with pytest.raises(ValueError, match="Entity leakage"):
        assert_entity_disjoint({"train": left, "test": right})


def test_shared_component_connects_aircraft_groups() -> None:
    frame = pd.DataFrame(
        {
            "component_id": ["A", "B", "A", "C"],
            "aircraft_id": ["AC1", "AC2", "AC2", "AC3"],
        }
    )
    groups = build_entity_groups(frame)
    assert groups.loc[0, "entity_group_id"] == groups.loc[2, "entity_group_id"]
    assert groups.loc[1, "entity_group_id"] == groups.loc[2, "entity_group_id"]
    assert groups.loc[3, "entity_group_id"] != groups.loc[2, "entity_group_id"]


def test_delayed_sensor_is_not_used_before_available() -> None:
    times = pd.to_datetime(["2025-01-01", "2025-01-06", "2025-01-11"], utc=True)
    frame = pd.DataFrame(
        {
            "component_id": ["A"] * 3,
            "component_total_cycles": [5, 10, 15],
            "cycles_since_install": [5, 10, 15],
            "reading_time": times,
            "flight_date": times,
            "available_at": [times[0], times[2] + pd.Timedelta(days=1), times[2]],
            "vibration_rms": [10.0, 20.0, 100.0],
        }
    )
    result = add_causal_rolling_features(
        frame, sensor_columns=["vibration_rms"], windows=[15]
    )
    assert result.loc[2, "vibration_rms_mean_last_15_cycles"] == 55.0
    assert result.loc[2, "vibration_rms_is_missing"] == 0
