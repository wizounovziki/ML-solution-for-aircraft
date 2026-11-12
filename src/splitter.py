from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_entity_groups(
    df: pd.DataFrame,
    component_col: str = "component_id",
    aircraft_col: str = "aircraft_id",
) -> pd.DataFrame:
    required = {component_col, aircraft_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing entity columns: {sorted(required.difference(df.columns))}")
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for component, aircraft in df[[component_col, aircraft_col]].drop_duplicates().itertuples(index=False):
        union(f"component:{component}", f"aircraft:{aircraft}")
    roots = [find(f"aircraft:{aircraft}") for aircraft in df[aircraft_col]]
    root_ids = {root: f"EG{index:05d}" for index, root in enumerate(sorted(set(roots)), start=1)}
    result = df[[component_col, aircraft_col]].copy()
    result["entity_group_id"] = [root_ids[root] for root in roots]
    return result


def entity_group_diagnostics(
    df: pd.DataFrame,
    group_col: str = "entity_group_id",
    event_col: str = "removal_event_id",
) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for group_id, group in df.groupby(group_col):
        rows.append(
            {
                "entity_group_id": group_id,
                "snapshots": len(group),
                "aircraft": group["aircraft_id"].nunique(),
                "components": group["component_id"].nunique(),
                "events": group[event_col].dropna().nunique() if event_col in group else 0,
                "snapshot_share": len(group) / total_rows,
            }
        )
    return pd.DataFrame(rows).sort_values("snapshots", ascending=False).reset_index(drop=True)


def assign_outer_partitions(
    df: pd.DataFrame,
    group_col: str = "entity_group_id",
    event_col: str = "removal_event_id",
    fractions: dict[str, float] | None = None,
    seed: int = 20261112,
) -> pd.Series:
    fractions = fractions or {"train": 0.60, "validation": 0.20, "test": 0.20}
    if set(fractions) != {"train", "validation", "test"} or not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Outer fractions must define train/validation/test and sum to one")
    stats = entity_group_diagnostics(df, group_col, event_col)
    if len(stats) < 3:
        raise ValueError("At least three connected entity groups are required")
    rng = np.random.default_rng(seed)
    tie_breaker = {group: value for group, value in zip(stats[group_col], rng.random(len(stats)))}
    stats = stats.sort_values(
        ["snapshots", group_col], ascending=[False, True], key=lambda col: col.map(tie_breaker) if col.name == group_col else col
    )
    total_rows = stats["snapshots"].sum()
    total_events = stats["events"].sum()
    current = {name: {"snapshots": 0, "events": 0} for name in fractions}
    assignments: dict[str, str] = {}
    for row in stats.itertuples(index=False):
        costs = {}
        for name, fraction in fractions.items():
            row_cost = (current[name]["snapshots"] + row.snapshots) / (total_rows * fraction)
            event_cost = (
                (current[name]["events"] + row.events) / (total_events * fraction)
                if total_events
                else 0.0
            )
            costs[name] = row_cost + 0.5 * event_cost
        selected = min(costs, key=lambda name: (costs[name], tie_breaker[name] if name in tie_breaker else 0.0))
        assignments[row.entity_group_id] = selected
        current[selected]["snapshots"] += row.snapshots
        current[selected]["events"] += row.events
    return df[group_col].map(assignments).rename("outer_partition")


def temporal_group_purged_split(
    df: pd.DataFrame,
    time_col: str = "flight_date",
    component_col: str = "component_id",
    aircraft_col: str = "aircraft_id",
    cutoff_date: str = "2025-09-01",
    horizon_cycles: int = 30,
    cycle_col: str = "component_total_cycles",
    label_available_col: str = "label_available_at",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")
    work = df.copy()
    work[time_col] = _utc(work[time_col])
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    if work[time_col].isna().any():
        raise ValueError(f"{time_col} contains invalid timestamps")
    work = work.sort_values([component_col, time_col, cycle_col]).reset_index(drop=True)
    pre_cutoff = work.loc[work[time_col] < cutoff].copy()
    test_df = work.loc[work[time_col] >= cutoff].copy()

    if label_available_col in work.columns:
        label_available = _utc(pre_cutoff[label_available_col])
        train_df = pre_cutoff.loc[label_available.notna() & (label_available < cutoff)].copy()
    else:
        active_across_cutoff = set(pre_cutoff[component_col]).intersection(test_df[component_col])
        last_pre_cycles = pre_cutoff.groupby(component_col)[cycle_col].max()
        keep = pd.Series(True, index=pre_cutoff.index)
        for component in active_across_cutoff:
            component_rows = pre_cutoff[component_col].eq(component)
            boundary_cycle = last_pre_cycles.loc[component] - horizon_cycles
            keep.loc[component_rows] = pre_cutoff.loc[component_rows, cycle_col] <= boundary_cycle
        train_df = pre_cutoff.loc[keep].copy()

    if not train_df.empty and not test_df.empty:
        assert train_df[time_col].max() < cutoff <= test_df[time_col].min()
    train_df.attrs["purged_snapshots"] = len(pre_cutoff) - len(train_df)
    train_df.attrs["cutoff_date"] = cutoff.isoformat()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def assert_entity_disjoint(
    partitions: dict[str, pd.DataFrame],
    component_col: str = "component_id",
    aircraft_col: str = "aircraft_id",
) -> None:
    names = list(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = partitions[left_name], partitions[right_name]
            for column in (component_col, aircraft_col):
                overlap = set(left[column]).intersection(right[column])
                if overlap:
                    raise ValueError(f"Entity leakage between {left_name} and {right_name} in {column}")


def create_joint_split_manifest(
    df: pd.DataFrame,
    train_end: str,
    validation_end: str,
    calibration_end: str | None = None,
    seed: int = 20261112,
) -> pd.DataFrame:
    required = {
        "snapshot_id",
        "flight_date",
        "component_id",
        "aircraft_id",
        "removal_event_id",
        "label_available_at",
        "unscheduled_removal_within_30_cycles",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    df = df.reset_index(drop=True)
    manifest = build_entity_groups(df)
    manifest["snapshot_id"] = df["snapshot_id"].to_numpy()
    manifest["outer_partition"] = assign_outer_partitions(
        pd.concat([manifest, df[["removal_event_id"]]], axis=1), seed=seed
    ).to_numpy()
    flight_date = _utc(df["flight_date"])
    label_available = _utc(df["label_available_at"])
    target = df["unscheduled_removal_within_30_cycles"]
    train_cutoff = pd.Timestamp(train_end, tz="UTC")
    validation_cutoff = pd.Timestamp(validation_end, tz="UTC")
    if not train_cutoff < validation_cutoff:
        raise ValueError("train_end must be earlier than validation_end")
    calibration_cutoff = (
        pd.Timestamp(calibration_end, tz="UTC")
        if calibration_end
        else train_cutoff + (validation_cutoff - train_cutoff) / 2
    )
    if not train_cutoff < calibration_cutoff < validation_cutoff:
        raise ValueError("calibration_end must lie between train_end and validation_end")

    manifest["flight_date"] = flight_date.to_numpy()
    manifest["label_available_at"] = label_available.to_numpy()
    manifest["final_use"] = "excluded"
    manifest["exclusion_reason"] = "outside_entity_time_assignment"

    train_mask = (
        manifest["outer_partition"].eq("train")
        & flight_date.lt(train_cutoff)
        & target.notna()
        & label_available.notna()
        & label_available.lt(train_cutoff)
    )
    manifest.loc[train_mask, "final_use"] = "train"
    manifest.loc[train_mask, "exclusion_reason"] = ""

    validation_entity = manifest["outer_partition"].eq("validation")
    validation_time = flight_date.ge(train_cutoff) & flight_date.lt(validation_cutoff)
    calibration_mask = (
        validation_entity
        & validation_time
        & flight_date.lt(calibration_cutoff)
        & target.notna()
        & label_available.notna()
        & label_available.lt(calibration_cutoff)
    )
    policy_mask = validation_entity & validation_time & flight_date.ge(calibration_cutoff)
    test_mask = manifest["outer_partition"].eq("test") & flight_date.ge(validation_cutoff)
    for name, mask in [("calibration", calibration_mask), ("policy", policy_mask), ("test", test_mask)]:
        manifest.loc[mask, "final_use"] = name
        manifest.loc[mask, "exclusion_reason"] = ""
    validate_joint_split(manifest, df, train_cutoff, calibration_cutoff, validation_cutoff)
    return manifest


def validate_joint_split(
    manifest: pd.DataFrame,
    source: pd.DataFrame,
    train_end: pd.Timestamp,
    calibration_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> None:
    uses = {"train", "calibration", "policy", "test"}
    selected = manifest[manifest["final_use"].isin(uses)]
    joined = selected.merge(
        source[["snapshot_id", "aircraft_id", "component_id", "removal_event_id"]],
        on="snapshot_id",
        suffixes=("", "_source"),
        validate="one_to_one",
    )
    partitions = {name: group for name, group in joined.groupby("outer_partition")}
    assert_entity_disjoint(partitions, "component_id_source", "aircraft_id_source")
    event_partitions = joined.dropna(subset=["removal_event_id"]).groupby("removal_event_id")["outer_partition"].nunique()
    if event_partitions.gt(1).any():
        raise ValueError("A removal event crosses outer partitions")
    times = manifest.set_index("final_use")["flight_date"]
    if "train" in times.index and "calibration" in times.index:
        assert times.loc["train"].max() < train_end <= times.loc["calibration"].min()
    if "calibration" in times.index and "policy" in times.index:
        assert times.loc["calibration"].max() < calibration_end <= times.loc["policy"].min()
    if "policy" in times.index and "test" in times.index:
        assert times.loc["policy"].max() < validation_end <= times.loc["test"].min()
    train_labels = manifest[manifest["final_use"].eq("train")]
    calibration_labels = manifest[manifest["final_use"].eq("calibration")]
    assert train_labels["label_available_at"].max() < train_end
    assert calibration_labels["label_available_at"].max() < calibration_end
