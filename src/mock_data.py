from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

FAMILY_SPECS = {
    "ENGINE": {
        "models": ["ENG-A1", "ENG-A2"],
        "next_model": "ENG-NEXT",
        "temperature": 610.0,
        "pressure": 54.0,
        "vibration": 1.1,
        "current": 24.0,
        "fault_codes": ["ENG-VIB", "ENG-EGT", "ENG-OIL"],
    },
    "HYDRAULIC_PUMP": {
        "models": ["HYD-B1", "HYD-B2"],
        "next_model": "HYD-NEXT",
        "temperature": 74.0,
        "pressure": 190.0,
        "vibration": 0.7,
        "current": 16.0,
        "fault_codes": ["HYD-PRESS", "HYD-LEAK", "HYD-TEMP"],
    },
    "AVIONICS_COMPUTER": {
        "models": ["AVI-C1", "AVI-C2"],
        "next_model": "AVI-NEXT",
        "temperature": 42.0,
        "pressure": 28.0,
        "vibration": 0.2,
        "current": 8.0,
        "fault_codes": ["AVI-BIT", "AVI-BUS", "AVI-TEMP"],
    },
    "BRAKE_ACTUATOR": {
        "models": ["BRK-D1", "BRK-D2"],
        "next_model": "BRK-NEXT",
        "temperature": 64.0,
        "pressure": 158.0,
        "vibration": 0.5,
        "current": 12.0,
        "fault_codes": ["BRK-PRESS", "BRK-WEAR", "BRK-TEMP"],
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "seed",
        "start_date",
        "n_aircraft",
        "cycles_per_aircraft",
        "snapshot_interval_cycles",
        "prediction_horizon_cycles",
        "mean_life_cycles",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if config["n_aircraft"] < 4 or config["cycles_per_aircraft"] < 100:
        raise ValueError("Dataset is too small for the intended evaluation")
    if config["snapshot_interval_cycles"] > config["prediction_horizon_cycles"]:
        raise ValueError("Snapshot interval cannot exceed the prediction horizon")
    return config


def _event_time(start: pd.Timestamp, cycle: float, days_per_cycle: float) -> str:
    return (start + pd.to_timedelta(cycle * days_per_cycle, unit="D")).isoformat()


def _choose_model(
    family: str, install_cycle: int, total_cycles: int, rng: np.random.Generator
) -> str:
    spec = FAMILY_SPECS[family]
    if install_cycle >= total_cycles * 0.8 and rng.random() < 0.35:
        return spec["next_model"]
    return str(rng.choice(spec["models"]))


def _recent_available_counts(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    window_cycles: int,
) -> pd.Series:
    result = pd.Series(0, index=snapshots.index, dtype="int64")
    event_groups = {
        component: group["available_cycle"].sort_values().to_numpy()
        for component, group in events.groupby("component_serial")
    }
    for component, group in snapshots.groupby("component_serial"):
        event_cycles = event_groups.get(component, np.array([]))
        cycles = group["component_total_cycles"].to_numpy()
        right = np.searchsorted(event_cycles, cycles, side="right")
        left = np.searchsorted(event_cycles, cycles - window_cycles, side="right")
        result.loc[group.index] = right - left
    return result


def _build_combined_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    snapshots = frames["component_snapshots"].rename(
        columns={"available_at": "snapshot_available_at"}
    )
    sensors = frames["sensor_readings"].drop(columns=["component_serial"])
    labels = frames["labels"]
    combined = snapshots.merge(sensors, on="snapshot_id", validate="one_to_one")
    combined = combined.merge(labels, on="snapshot_id", validate="one_to_one")
    combined = combined.merge(
        frames["aircraft"][["aircraft_id", "aircraft_model", "is_legacy_aircraft"]],
        on="aircraft_id",
        validate="many_to_one",
    )
    combined["component_id"] = combined["component_serial"]
    combined["part_number"] = combined["component_model"]
    combined["flight_date"] = combined["snapshot_time"]
    combined["scoring_batch_id"] = "BATCH-" + combined["flight_date"].str[:10]
    combined["csn"] = combined["component_total_cycles"]
    combined["cso"] = combined["cycles_since_install"]
    combined["unscheduled_removal_within_30_cycles"] = combined[
        "target_unscheduled_removal"
    ]
    combined["fault_code_count_last_10_cycles"] = _recent_available_counts(
        combined, frames["fault_events"], 10
    )
    combined["maintenance_count_last_30_cycles"] = _recent_available_counts(
        combined, frames["maintenance_events"], 30
    )
    first_columns = [
        "snapshot_id",
        "flight_date",
        "aircraft_id",
        "component_id",
        "installation_id",
        "part_number",
        "unscheduled_removal_within_30_cycles",
        "is_censored",
        "label_available_at",
    ]
    remaining = [column for column in combined.columns if column not in first_columns]
    return combined[first_columns + remaining].sort_values(
        ["flight_date", "aircraft_id", "component_id"]
    ).reset_index(drop=True)


def build_dataset(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config["seed"])
    prefix = str(config.get("entity_namespace", ""))
    prefix = prefix + "-" if prefix else ""
    start = pd.Timestamp(config["start_date"])
    if start.tzinfo is None:
        start = start.tz_localize(config.get("time_zone", "UTC"))
    total_cycles = int(config["cycles_per_aircraft"])
    interval = int(config["snapshot_interval_cycles"])
    horizon = int(config["prediction_horizon_cycles"])
    days_per_cycle = float(config["days_per_cycle"])
    unscheduled_probability = float(config["unscheduled_removal_probability"])

    aircraft_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    maintenance_rows: list[dict[str, Any]] = []
    removal_rows: list[dict[str, Any]] = []

    environments = ["TEMPERATE", "HOT", "COLD", "HIGH_HUMIDITY", "HOT_DUSTY"]
    models = ["A320", "A321", "B737-800", "B737-CLASSIC"]

    for aircraft_index in range(1, int(config["n_aircraft"]) + 1):
        aircraft_id = f"{prefix}AC{aircraft_index:04d}"
        environment = str(rng.choice(environments))
        aircraft_model = str(rng.choice(models, p=[0.3, 0.25, 0.3, 0.15]))
        initial_aircraft_age = int(rng.integers(500, 4000))
        aircraft_rows.append(
            {
                "aircraft_id": aircraft_id,
                "aircraft_model": aircraft_model,
                "entry_service_date": _event_time(start, -initial_aircraft_age, 1.0)[:10],
                "age_cycles_at_simulation_start": initial_aircraft_age,
                "operating_environment": environment,
                "is_legacy_aircraft": aircraft_model == "B737-CLASSIC",
                "fleet_group": f"F{(aircraft_index - 1) % 6 + 1}",
            }
        )

        for family in FAMILY_SPECS:
            install_cycle = 0
            installation_number = 1
            while install_cycle <= total_cycles:
                component_serial = f"{prefix}C{len(component_rows) + 1:07d}"
                installation_id = f"INS-{component_serial}"
                spec = FAMILY_SPECS[family]
                harshness = 0.82 if environment in {"HOT", "HIGH_HUMIDITY", "HOT_DUSTY"} else 1.0
                mean_life = float(config["mean_life_cycles"][family]) * harshness
                lifetime = int(
                    np.clip(
                        rng.lognormal(np.log(mean_life) - 0.5 * 0.22**2, 0.22),
                        350,
                        mean_life * 2.2,
                    )
                )
                removal_cycle = install_cycle + lifetime
                removal_observed = removal_cycle <= total_cycles
                removal_type = (
                    str(
                        rng.choice(
                            ["UNSCHEDULED", "SCHEDULED"],
                            p=[unscheduled_probability, 1.0 - unscheduled_probability],
                        )
                    )
                    if removal_observed
                    else None
                )
                removal_event_id = f"{prefix}REM{len(removal_rows) + 1:07d}" if removal_observed else None
                component_model = _choose_model(family, install_cycle, total_cycles, rng)
                component_rows.append(
                    {
                        "component_serial": component_serial,
                        "installation_id": installation_id,
                        "component_family": family,
                        "component_model": component_model,
                        "manufacture_batch": f"B{int(rng.integers(2017, 2024))}-{int(rng.integers(1, 13)):02d}",
                        "manufacture_date": _event_time(start, -float(rng.integers(100, 2500)), 1.0)[:10],
                        "install_aircraft_id": aircraft_id,
                        "install_cycle": install_cycle,
                        "cycles_before_install": 0,
                        "installation_number": installation_number,
                        "remove_cycle": removal_cycle if removal_observed else np.nan,
                    }
                )

                if removal_observed:
                    removal_rows.append(
                        {
                            "removal_event_id": removal_event_id,
                            "component_serial": component_serial,
                            "installation_id": installation_id,
                            "aircraft_id": aircraft_id,
                            "removal_cycle": removal_cycle,
                            "removal_time": _event_time(start, removal_cycle, days_per_cycle),
                            "available_at": _event_time(
                                start, removal_cycle + 0.25 / days_per_cycle, days_per_cycle
                            ),
                            "removal_type": removal_type,
                            "reason_code": "DEGRADATION" if removal_type == "UNSCHEDULED" else "LIFE_LIMIT",
                            "confirmed_failure": removal_type == "UNSCHEDULED" and rng.random() > 0.08,
                        }
                    )

                first_snapshot = ((install_cycle + interval) // interval) * interval
                last_snapshot = min(removal_cycle - 1, total_cycles)
                for snapshot_cycle in range(first_snapshot, last_snapshot + 1, interval):
                    snapshot_id = f"{prefix}S{len(snapshot_rows) + 1:08d}"
                    age = snapshot_cycle - install_cycle
                    cycles_to_end = removal_cycle - snapshot_cycle
                    degradation = float(np.clip(age / lifetime, 0.0, 1.0))
                    terminal = float(np.clip((degradation - 0.78) / 0.22, 0.0, 1.0))
                    acute = float(np.clip((35.0 - cycles_to_end) / 35.0, 0.0, 1.0))
                    acute = float((np.exp(2.0 * acute) - 1.0) / (np.e**2 - 1.0))
                    environment_heat = {
                        "HOT": 9.0,
                        "HIGH_HUMIDITY": 4.0,
                        "HOT_DUSTY": 12.0,
                    }.get(environment, 0.0)
                    seasonal = 8.0 * np.sin(2.0 * np.pi * snapshot_cycle / 487.0)
                    ambient_temperature = 15.0 + environment_heat + seasonal + rng.normal(0, 2.0)
                    snapshot_time = _event_time(start, snapshot_cycle, days_per_cycle)
                    snapshot_rows.append(
                        {
                            "snapshot_id": snapshot_id,
                            "aircraft_id": aircraft_id,
                            "component_serial": component_serial,
                            "installation_id": installation_id,
                            "component_family": family,
                            "component_model": component_model,
                            "snapshot_cycle": snapshot_cycle,
                            "snapshot_time": snapshot_time,
                            "available_at": snapshot_time,
                            "aircraft_total_cycles": snapshot_cycle,
                            "aircraft_age_cycles": initial_aircraft_age + snapshot_cycle,
                            "component_total_cycles": age,
                            "component_flight_hours": age * float(rng.uniform(1.15, 1.35)),
                            "operating_environment": environment,
                            "ambient_temperature": ambient_temperature,
                            "cycles_since_install": age,
                        }
                    )

                    drift = max(snapshot_cycle / total_cycles - 0.8, 0.0) * 4.0
                    sensor_delay = 0.25 / days_per_cycle if rng.random() < 0.03 else 0.0
                    sensor_rows.append(
                        {
                            "snapshot_id": snapshot_id,
                            "component_serial": component_serial,
                            "reading_cycle": snapshot_cycle,
                            "reading_time": snapshot_time,
                            "available_at": _event_time(
                                start, snapshot_cycle + sensor_delay, days_per_cycle
                            ),
                            "vibration_rms": spec["vibration"]
                            + 1.2 * degradation**1.8
                            + 0.8 * terminal
                            + 2.2 * acute
                            + rng.normal(0, 0.12),
                            "egt_margin": (
                                62.0 - 12.0 * degradation - 24.0 * acute + rng.normal(0, 1.0)
                                if family == "ENGINE"
                                else np.nan
                            ),
                            "sensor_temperature": spec["temperature"]
                            + environment_heat
                            + 10.0 * degradation
                            + 10.0 * terminal
                            + 18.0 * acute
                            + rng.normal(0, 1.4),
                            "sensor_pressure": spec["pressure"]
                            - 0.04 * spec["pressure"] * degradation
                            - 0.04 * spec["pressure"] * terminal
                            - 0.06 * spec["pressure"] * acute
                            + rng.normal(0, 0.008 * spec["pressure"]),
                            "sensor_current": spec["current"]
                            + 0.10 * spec["current"] * degradation
                            + 0.14 * spec["current"] * terminal
                            + rng.normal(0, 0.025 * spec["current"])
                            + drift,
                            "sensor_quality": "DEGRADED" if drift > 0 and rng.random() < 0.2 else "VALID",
                        }
                    )

                    fault_probability = 0.004 + 0.02 * degradation + 0.16 * terminal + 0.38 * acute
                    if rng.random() < fault_probability:
                        code = (
                            spec["fault_codes"][0]
                            if acute > 0.05 and rng.random() < 0.7
                            else str(rng.choice(spec["fault_codes"]))
                        )
                        available_delay = float(rng.uniform(0, 2.0)) / days_per_cycle
                        available_cycle = age + available_delay
                        fault_rows.append(
                            {
                                "fault_event_id": f"{prefix}FLT{len(fault_rows) + 1:08d}",
                                "snapshot_id": snapshot_id,
                                "component_serial": component_serial,
                                "installation_id": installation_id,
                                "aircraft_id": aircraft_id,
                                "event_cycle": snapshot_cycle,
                                "event_time": snapshot_time,
                                "available_cycle": available_cycle,
                                "available_at": _event_time(
                                    start, snapshot_cycle + available_delay, days_per_cycle
                                ),
                                "fault_code": code,
                                "severity": str(rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.35, 0.5, 0.15])),
                            }
                        )

                    maintenance_type = None
                    if age > 0 and age % 250 == 0:
                        maintenance_type = "INSPECTION"
                    elif degradation > 0.45 and rng.random() < 0.018:
                        maintenance_type = "REPAIR"
                    if maintenance_type:
                        available_delay = float(rng.uniform(0, 1.5)) / days_per_cycle
                        available_cycle = age + available_delay
                        maintenance_rows.append(
                            {
                                "maintenance_event_id": f"{prefix}MNT{len(maintenance_rows) + 1:08d}",
                                "snapshot_id": snapshot_id,
                                "component_serial": component_serial,
                                "installation_id": installation_id,
                                "aircraft_id": aircraft_id,
                                "event_cycle": snapshot_cycle,
                                "event_time": snapshot_time,
                                "available_cycle": available_cycle,
                                "available_at": _event_time(
                                    start, snapshot_cycle + available_delay, days_per_cycle
                                ),
                                "event_type": maintenance_type,
                                "action_code": (
                                    "ROUTINE_CHECK"
                                    if maintenance_type == "INSPECTION"
                                    else "CORRECTIVE_ACTION"
                                ),
                            }
                        )

                if not removal_observed:
                    break
                install_cycle = removal_cycle + 1
                installation_number += 1

    frames = {
        "aircraft": pd.DataFrame(aircraft_rows),
        "components": pd.DataFrame(component_rows),
        "component_snapshots": pd.DataFrame(snapshot_rows),
        "sensor_readings": pd.DataFrame(sensor_rows),
        "fault_events": pd.DataFrame(fault_rows),
        "maintenance_events": pd.DataFrame(maintenance_rows),
        "removal_events": pd.DataFrame(removal_rows),
    }

    labels = frames["component_snapshots"][
        ["snapshot_id", "installation_id", "snapshot_cycle", "snapshot_time"]
    ].copy()
    labels = labels.merge(
        frames["removal_events"][
            ["installation_id", "removal_event_id", "removal_cycle", "removal_type", "available_at"]
        ],
        on="installation_id",
        how="left",
        validate="many_to_one",
    )
    cycles_to_removal = labels["removal_cycle"] - labels["snapshot_cycle"]
    positive = (
        labels["removal_type"].eq("UNSCHEDULED")
        & cycles_to_removal.gt(0)
        & cycles_to_removal.le(horizon)
    )
    scheduled_competing_event = (
        labels["removal_type"].eq("SCHEDULED")
        & cycles_to_removal.gt(0)
        & cycles_to_removal.le(horizon)
    )
    end_of_observation = labels["snapshot_cycle"] + horizon > total_cycles
    labels["is_censored"] = ~positive & (scheduled_competing_event | end_of_observation)
    labels["target_unscheduled_removal"] = pd.array(
        np.where(positive, 1, np.where(labels["is_censored"], None, 0)), dtype="Int64"
    )
    labels["removal_event_id"] = labels["removal_event_id"].where(positive)
    mature_time = pd.to_datetime(labels["snapshot_time"]) + pd.to_timedelta(
        horizon * days_per_cycle, unit="D"
    )
    labels["label_available_at"] = mature_time.where(~positive, pd.to_datetime(labels["available_at"]))
    labels.loc[labels["is_censored"], "label_available_at"] = pd.NaT
    frames["labels"] = labels[
        [
            "snapshot_id",
            "target_unscheduled_removal",
            "is_censored",
            "removal_event_id",
            "label_available_at",
        ]
    ]

    sensors = frames["sensor_readings"]
    numeric_columns = [
        "vibration_rms",
        "egt_margin",
        "sensor_temperature",
        "sensor_pressure",
        "sensor_current",
    ]
    missing_mask = rng.random((len(sensors), len(numeric_columns))) < float(config["missing_rate"])
    sensors.loc[:, numeric_columns] = sensors[numeric_columns].mask(missing_mask)
    legacy_by_snapshot = frames["component_snapshots"].set_index("snapshot_id")["aircraft_id"].map(
        frames["aircraft"].set_index("aircraft_id")["is_legacy_aircraft"]
    )
    legacy = sensors["snapshot_id"].map(legacy_by_snapshot).to_numpy(dtype=bool)
    structured_mask = (
        rng.random((len(sensors), len(numeric_columns))) < 0.10
    ) & legacy[:, None]
    sensors.loc[:, numeric_columns] = sensors[numeric_columns].mask(structured_mask)
    avionics = sensors["component_serial"].map(
        frames["components"].set_index("component_serial")["component_family"]
    ).eq("AVIONICS_COMPUTER")
    sensors.loc[avionics & (rng.random(len(sensors)) < 0.65), "sensor_pressure"] = np.nan
    outlier_mask = rng.random((len(sensors), len(numeric_columns))) < float(config["outlier_rate"])
    for column_index, column in enumerate(numeric_columns):
        multiplier = np.where(outlier_mask[:, column_index], rng.choice([-2.5, 3.5]), 1.0)
        sensors.loc[:, column] = sensors[column] * multiplier

    frames["mro_dataset"] = _build_combined_dataset(frames)
    return frames


def dataset_summary(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, Any]:
    labels = frames["labels"]
    labeled = labels.dropna(subset=["target_unscheduled_removal"])
    removals = frames["removal_events"]
    positives = labeled["target_unscheduled_removal"].eq(1)
    return {
        "dataset_version": config["dataset_version"],
        "seed": config["seed"],
        "prediction_horizon_cycles": config["prediction_horizon_cycles"],
        "snapshot_interval_cycles": config["snapshot_interval_cycles"],
        "rows": {name: len(frame) for name, frame in frames.items()},
        "aircraft": int(frames["aircraft"]["aircraft_id"].nunique()),
        "components": int(frames["components"]["component_serial"].nunique()),
        "removals_by_type": {
            str(key): int(value) for key, value in removals["removal_type"].value_counts().items()
        },
        "labeled_snapshots": len(labeled),
        "censored_snapshots": int(labels["is_censored"].sum()),
        "positive_snapshots": int(positives.sum()),
        "positive_snapshot_rate": float(positives.mean()),
        "unscheduled_events_with_positive_snapshot": int(
            labeled.loc[positives, "removal_event_id"].nunique()
        ),
    }


def write_dataset(
    frames: dict[str, pd.DataFrame], config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"summary": dataset_summary(frames, config), "files": {},
                                "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    for name, frame in frames.items():
        csv_path = output / f"{name}.csv"
        parquet_path = output / f"{name}.parquet"
        frame.to_csv(csv_path, index=False)
        frame.to_parquet(parquet_path, index=False)
        manifest["files"][name] = {
            "csv": csv_path.name,
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "parquet": parquet_path.name,
            "parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        }
    with (output / "generation_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
