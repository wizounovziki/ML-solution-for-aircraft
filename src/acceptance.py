from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.provenance import sha256
from src.splitter import build_entity_groups
from src.validation import read_dataset, validate_dataset


def make_acceptance_split(features_path: str | Path, output_path: str | Path) -> None:
    features = pd.read_parquet(features_path)
    manifest = build_entity_groups(features)
    manifest["snapshot_id"] = features["snapshot_id"].to_numpy()
    manifest["outer_partition"] = "test"
    manifest["final_use"] = "test"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)


def validate_acceptance_inputs(
    freeze: dict[str, Any], protocol_path: str | Path,
    features_path: str | Path, split_path: str | Path, data_dir: str | Path,
) -> dict[str, Any]:
    if freeze.get("acceptance_protocol_sha256") != sha256(protocol_path):
        raise ValueError("Acceptance protocol differs from the preregistered protocol")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol != freeze["acceptance_protocol"]:
        raise ValueError("Acceptance protocol content differs from the frozen protocol")
    if protocol["target_event_recall"] != 0.80 or protocol["max_alert_rate"] != freeze["max_alert_rate"]:
        raise ValueError("Acceptance criteria differ from the frozen policy")
    for name in ("features", "split"):
        if sha256(freeze[f"{name}_path"]) != freeze[f"{name}_sha256"]:
            raise ValueError(f"Original {name} artifact has changed")
    root = Path(data_dir)
    config = yaml.safe_load((root / "generation_config.yaml").read_text(encoding="utf-8"))
    if config != protocol["dataset_config"]:
        raise ValueError("Acceptance generator config differs from the frozen config")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if pd.Timestamp(manifest["generated_at_utc"]) <= pd.Timestamp(freeze["frozen_at_utc"]):
        raise ValueError("Acceptance data must be generated after model freeze")
    for name, record in manifest["files"].items():
        if sha256(root / record["parquet"]) != record["parquet_sha256"]:
            raise ValueError(f"Acceptance raw artifact changed: {name}")
    validate_dataset(read_dataset(root), config)
    feature_manifest = json.loads((Path(features_path).parent / "feature_manifest.json").read_text(encoding="utf-8"))
    if (feature_manifest["output_sha256"] != sha256(features_path)
            or feature_manifest["input_sha256"] != sha256(root / "mro_dataset.parquet")
            or feature_manifest["source_fingerprint"] != freeze["source_fingerprint"]):
        raise ValueError("Acceptance feature provenance differs from the frozen implementation")
    features = pd.read_parquet(features_path)
    previous = pd.read_parquet(freeze["features_path"])
    splits = pd.read_parquet(split_path)
    if (splits["snapshot_id"].duplicated().any()
            or set(splits["snapshot_id"]) != set(features["snapshot_id"])
            or not splits["final_use"].eq("test").all()):
        raise ValueError("Acceptance split must score every snapshot exactly once as test")
    expected = build_entity_groups(features).assign(snapshot_id=features["snapshot_id"].to_numpy())
    actual = splits.set_index("snapshot_id")["entity_group_id"]
    if not expected["snapshot_id"].map(actual).eq(expected["entity_group_id"]).all():
        raise ValueError("Acceptance entity groups differ from connected components")
    times = pd.to_datetime(features["flight_date"], utc=True)
    if not (times.ge(pd.Timestamp(protocol["start_inclusive"]))
            & times.lt(pd.Timestamp(protocol["end_exclusive"]))).all():
        raise ValueError("Acceptance rows fall outside the preregistered interval")
    previous_end = pd.to_datetime(previous["flight_date"], utc=True).max()
    if times.min() <= previous_end:
        raise ValueError("Acceptance interval overlaps previously inspected data")
    intersections = {}
    for column in ("aircraft_id", "component_id", "installation_id", "snapshot_id", "removal_event_id"):
        overlap = set(features[column].dropna()) & set(previous[column].dropna())
        intersections[column] = len(overlap)
        if overlap:
            raise ValueError(f"Acceptance entities overlap prior data: {column}")
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "data_manifest_sha256": sha256(root / "manifest.json"),
        "data_version": config["dataset_version"],
        "seed": config["seed"],
        "generated_at_utc": manifest["generated_at_utc"],
        "frozen_at_utc": freeze["frozen_at_utc"],
        "previous_latest_snapshot": previous_end.isoformat(),
        "first_snapshot": times.min().isoformat(),
        "last_snapshot": times.max().isoformat(),
        "entity_intersections": intersections,
        "aircraft": int(features["aircraft_id"].nunique()),
        "components": int(features["component_id"].nunique()),
        "rows": len(features),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create independent acceptance inputs")
    commands = parser.add_subparsers(dest="command", required=True)
    protocol = commands.add_parser("protocol")
    protocol.add_argument("--config", required=True)
    protocol.add_argument("--output", required=True)
    split = commands.add_parser("split")
    split.add_argument("--features", required=True)
    split.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "split":
        make_acceptance_split(args.features, args.output)
        return
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Acceptance protocol already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "protocol_id": "independent_2027_v1",
        "start_inclusive": "2027-01-01T00:00:00+00:00",
        "end_exclusive": "2032-01-01T00:00:00+00:00",
        "minimum_events": 100,
        "target_event_recall": 0.80,
        "max_alert_rate": 0.05,
        "bootstrap_iterations": 1000,
        "dataset_config": config,
        "dataset_config_sha256": sha256(args.config),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
