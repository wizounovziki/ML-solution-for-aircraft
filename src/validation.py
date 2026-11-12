from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import TARGET_POSITIVE_RATE_RANGE

TABLE_NAMES = [
    "aircraft",
    "components",
    "component_snapshots",
    "sensor_readings",
    "fault_events",
    "maintenance_events",
    "removal_events",
    "labels",
    "mro_dataset",
]


def read_dataset(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    return {name: pd.read_parquet(root / f"{name}.parquet") for name in TABLE_NAMES}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_unique(frame: pd.DataFrame, column: str, table: str) -> None:
    _require(frame[column].notna().all(), f"{table}.{column} contains missing keys")
    _require(not frame[column].duplicated().any(), f"{table}.{column} is not unique")


def _timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def validate_dataset(
    frames: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, bool]:
    _require(set(TABLE_NAMES).issubset(frames), "One or more required tables are missing")
    for name in TABLE_NAMES:
        _require(not frames[name].empty, f"{name} is empty")

    aircraft = frames["aircraft"]
    components = frames["components"]
    snapshots = frames["component_snapshots"]
    sensors = frames["sensor_readings"]
    faults = frames["fault_events"]
    maintenance = frames["maintenance_events"]
    removals = frames["removal_events"]
    labels = frames["labels"]
    combined = frames["mro_dataset"]
    horizon = int(config["prediction_horizon_cycles"])
    total_cycles = int(config["cycles_per_aircraft"])
    days_per_cycle = float(config["days_per_cycle"])

    _check_unique(aircraft, "aircraft_id", "aircraft")
    _check_unique(components, "component_serial", "components")
    _check_unique(components, "installation_id", "components")
    _check_unique(snapshots, "snapshot_id", "component_snapshots")
    _check_unique(sensors, "snapshot_id", "sensor_readings")
    _check_unique(faults, "fault_event_id", "fault_events")
    _check_unique(maintenance, "maintenance_event_id", "maintenance_events")
    _check_unique(removals, "removal_event_id", "removal_events")
    _check_unique(removals, "installation_id", "removal_events")
    _check_unique(labels, "snapshot_id", "labels")
    _check_unique(combined, "snapshot_id", "mro_dataset")

    _require(
        snapshots["aircraft_id"].isin(aircraft["aircraft_id"]).all(),
        "Snapshot contains an unknown aircraft",
    )
    _require(
        snapshots["component_serial"].isin(components["component_serial"]).all(),
        "Snapshot contains an unknown component",
    )
    _require(
        sensors["snapshot_id"].isin(snapshots["snapshot_id"]).all(),
        "Sensor reading contains an unknown snapshot",
    )
    _require(
        labels["snapshot_id"].isin(snapshots["snapshot_id"]).all(),
        "Label contains an unknown snapshot",
    )
    _require(len(labels) == len(snapshots), "Every snapshot must have exactly one label row")
    _require(len(combined) == len(snapshots), "Combined dataset must contain every snapshot")

    snapshot_history = snapshots.merge(
        components[["installation_id", "install_cycle", "cycles_before_install"]],
        on="installation_id",
        how="left",
        validate="many_to_one",
    ).merge(
        removals[["installation_id", "removal_cycle"]],
        on="installation_id",
        how="left",
        validate="many_to_one",
    )
    _require(
        snapshot_history["snapshot_cycle"].ge(snapshot_history["install_cycle"]).all(),
        "Snapshot predates its component installation",
    )
    expected_total_cycles = (
        snapshot_history["snapshot_cycle"]
        - snapshot_history["install_cycle"]
        + snapshot_history["cycles_before_install"]
    )
    _require(
        snapshot_history["component_total_cycles"].eq(expected_total_cycles).all(),
        "Component lifetime cycles are not traceable to installation intervals",
    )
    has_removal = snapshot_history["removal_cycle"].notna()
    _require(
        snapshot_history.loc[has_removal, "snapshot_cycle"]
        .lt(snapshot_history.loc[has_removal, "removal_cycle"])
        .all(),
        "An in-service snapshot occurs at or after removal",
    )

    sensor_time = sensors.merge(
        snapshots[["snapshot_id", "snapshot_cycle", "snapshot_time"]],
        on="snapshot_id",
        how="left",
        validate="one_to_one",
    )
    _require(
        sensor_time["reading_cycle"].eq(sensor_time["snapshot_cycle"]).all(),
        "Sensor reading time differs from its snapshot time",
    )
    _require(
        (_timestamps(sensors, "reading_time") <= _timestamps(sensors, "available_at")).all(),
        "Sensor availability predates the actual reading",
    )
    for name, events in [("fault", faults), ("maintenance", maintenance)]:
        event_time = events.merge(
            snapshots[["snapshot_id", "snapshot_cycle", "snapshot_time"]],
            on="snapshot_id",
            how="left",
            validate="many_to_one",
        )
        _require(
            event_time["event_cycle"].le(event_time["snapshot_cycle"]).all(),
            f"{name} event occurs after its associated snapshot",
        )
        _require(
            (_timestamps(events, "event_time") <= _timestamps(events, "available_at")).all(),
            f"{name} availability predates the actual event",
        )
    _require(
        (_timestamps(removals, "removal_time") <= _timestamps(removals, "available_at")).all(),
        "Removal availability predates the actual removal",
    )

    label_history = labels.merge(
        snapshots[["snapshot_id", "installation_id", "snapshot_cycle", "snapshot_time"]],
        on="snapshot_id",
        how="left",
        validate="one_to_one",
    ).merge(
        removals[
            ["installation_id", "removal_event_id", "removal_cycle", "removal_type", "available_at"]
        ],
        on="installation_id",
        how="left",
        suffixes=("", "_actual"),
        validate="many_to_one",
    )
    cycles_to_removal = label_history["removal_cycle"] - label_history["snapshot_cycle"]
    expected_positive = (
        label_history["removal_type"].eq("UNSCHEDULED")
        & cycles_to_removal.gt(0)
        & cycles_to_removal.le(horizon)
    )
    scheduled_competing_event = (
        label_history["removal_type"].eq("SCHEDULED")
        & cycles_to_removal.gt(0)
        & cycles_to_removal.le(horizon)
    )
    expected_censored = ~expected_positive & (
        scheduled_competing_event | label_history["snapshot_cycle"].add(horizon).gt(total_cycles)
    )
    target = label_history["target_unscheduled_removal"]
    _require(target.isin([0, 1, pd.NA]).all(), "Target contains values other than 0, 1, or null")
    _require(target.isna().eq(expected_censored).all(), "Censoring flags and null targets disagree")
    _require(target.eq(1).eq(expected_positive).all(), "Positive labels do not match removal events")
    _require(
        label_history["is_censored"].eq(expected_censored).all(),
        "Censoring does not match the observation or competing-event window",
    )
    _require(
        label_history.loc[expected_positive, "removal_event_id"].eq(
            label_history.loc[expected_positive, "removal_event_id_actual"]
        ).all(),
        "Positive labels reference the wrong removal event",
    )
    _require(
        label_history.loc[~expected_positive, "removal_event_id"].isna().all(),
        "Non-positive labels expose a removal event identifier",
    )

    negative_maturity = _timestamps(label_history, "snapshot_time") + pd.to_timedelta(
        horizon * days_per_cycle, unit="D"
    )
    expected_label_availability = negative_maturity.where(
        ~expected_positive, _timestamps(label_history, "available_at")
    )
    expected_label_availability = expected_label_availability.mask(expected_censored)
    actual_label_availability = _timestamps(label_history, "label_available_at")
    _require(
        (
            actual_label_availability.eq(expected_label_availability)
            | (actual_label_availability.isna() & expected_label_availability.isna())
        ).all(),
        "Label availability does not match event availability or horizon maturity",
    )

    positive_events = set(label_history.loc[expected_positive, "removal_event_id"])
    unscheduled_events = set(
        removals.loc[removals["removal_type"].eq("UNSCHEDULED"), "removal_event_id"]
    )
    _require(
        positive_events == unscheduled_events,
        "At least one unscheduled removal has no positive snapshot in its prediction window",
    )
    positive_rate = target.eq(1).sum() / target.notna().sum()
    lower, upper = TARGET_POSITIVE_RATE_RANGE
    _require(lower <= positive_rate <= upper, "Positive rate is outside the required 1.8%–2.2% range")
    _require(
        sensors[["vibration_rms", "egt_margin", "sensor_temperature", "sensor_pressure", "sensor_current"]]
        .isna()
        .any()
        .any(),
        "Missing-value injection did not produce missing sensor values",
    )
    _require(
        combined["target_unscheduled_removal"].equals(
            combined["unscheduled_removal_within_30_cycles"]
        ),
        "Combined target alias differs from the authoritative label",
    )
    _require(combined["scoring_batch_id"].notna().all(), "Scoring batch mapping is missing")

    return {
        "schema_and_keys": True,
        "referential_integrity": True,
        "installation_cycle_accounting": True,
        "event_time_alignment": True,
        "record_availability": True,
        "label_availability": True,
        "no_post_removal_snapshots": True,
        "label_horizon": True,
        "censoring": True,
        "all_unscheduled_events_covered": True,
        "target_positive_rate_1_8_to_2_2_percent": True,
        "missing_values_present": True,
        "combined_dataset_consistency": True,
    }
