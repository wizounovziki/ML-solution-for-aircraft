from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import ROLLING_WINDOWS_CYCLES, SENSOR_COLUMNS


def add_missing_indicators(
    frame: pd.DataFrame, columns: Sequence[str] = SENSOR_COLUMNS
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[f"{column}_is_missing"] = result[column].isna().astype("int8")
    return result


def _window_statistics(cycles: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
    valid = ~np.isnan(values)
    count = valid.sum(axis=0)
    safe_count = np.maximum(count, 1)
    value_sum = np.where(valid, values, 0.0).sum(axis=0)
    mean = value_sum / safe_count
    centered = np.where(valid, values - mean, 0.0)
    variance = (centered**2).sum(axis=0) / safe_count
    x_mean = (cycles[:, None] * valid).sum(axis=0) / safe_count
    x_centered = np.where(valid, cycles[:, None] - x_mean, 0.0)
    numerator = (x_centered * centered).sum(axis=0)
    denominator = (x_centered**2).sum(axis=0)
    slope = np.divide(numerator, denominator, out=np.full(values.shape[1], np.nan), where=denominator > 0)
    mean[count == 0] = np.nan
    std = np.sqrt(variance)
    std[count == 0] = np.nan
    slope[count < 2] = np.nan
    return mean, std, slope


def add_causal_rolling_features(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str] = SENSOR_COLUMNS,
    windows: Sequence[int] = ROLLING_WINDOWS_CYCLES,
    component_col: str = "component_id",
    cycle_col: str = "component_total_cycles",
    event_time_col: str = "reading_time",
    available_col: str = "available_at",
    snapshot_time_col: str = "flight_date",
) -> pd.DataFrame:
    required = {
        component_col,
        cycle_col,
        event_time_col,
        available_col,
        snapshot_time_col,
        *sensor_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing causal feature columns: {sorted(missing)}")
    work = frame.copy().reset_index(drop=True)
    for column in [event_time_col, available_col, snapshot_time_col]:
        work[column] = pd.to_datetime(work[column], errors="coerce", utc=True)
    if work[[event_time_col, available_col, snapshot_time_col]].isna().any().any():
        raise ValueError("Causal features require valid event, availability, and snapshot times")

    feature_arrays = {
        f"{sensor}_{stat}_last_{window}_cycles": np.full(len(work), np.nan)
        for sensor in sensor_columns
        for window in windows
        for stat in ("mean", "std", "slope_per_cycle")
    }
    for _, group in work.groupby(component_col, sort=False):
        group = group.sort_values([cycle_col, snapshot_time_col])
        output_index = group.index.to_numpy()
        cycles = group[cycle_col].to_numpy(dtype=float)
        event_ns = group[event_time_col].astype("int64").to_numpy()
        available_ns = group[available_col].astype("int64").to_numpy()
        snapshot_ns = group[snapshot_time_col].astype("int64").to_numpy()
        values = group[list(sensor_columns)].to_numpy(dtype=float)
        for row_index in range(len(group)):
            for window in windows:
                start = np.searchsorted(cycles, cycles[row_index] - window, side="right")
                candidate = np.arange(start, row_index + 1)
                eligible = candidate[
                    (event_ns[candidate] <= snapshot_ns[row_index])
                    & (available_ns[candidate] <= snapshot_ns[row_index])
                ]
                if not len(eligible):
                    continue
                mean, std, slope = _window_statistics(cycles[eligible], values[eligible])
                for sensor_index, sensor in enumerate(sensor_columns):
                    for stat, statistics in zip(("mean", "std", "slope_per_cycle"), (mean, std, slope)):
                        name = f"{sensor}_{stat}_last_{window}_cycles"
                        feature_arrays[name][output_index[row_index]] = statistics[sensor_index]
    for name, values in feature_arrays.items():
        work[name] = values
    work["history_coverage_cycles"] = np.minimum(work["cycles_since_install"], max(windows))
    work["is_cold_start"] = work["cycles_since_install"].lt(5).astype("int8")
    return add_missing_indicators(work, sensor_columns)


def model_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {
        "snapshot_id",
        "flight_date",
        "snapshot_time",
        "available_at",
        "snapshot_available_at",
        "label_available_at",
        "reading_time",
        "aircraft_id",
        "component_id",
        "component_serial",
        "installation_id",
        "scoring_batch_id",
        "removal_event_id",
        "target_unscheduled_removal",
        "unscheduled_removal_within_30_cycles",
        "is_censored",
        "component_model",
    }
    categorical = [
        column
        for column in [
            "aircraft_model",
            "component_family",
            "part_number",
            "manufacture_batch",
            "operating_environment",
            "sensor_quality",
        ]
        if column in frame.columns
    ]
    numeric = [
        column
        for column in frame.columns
        if column not in excluded | set(categorical)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return numeric, categorical
