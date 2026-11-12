from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.splitter import create_joint_split_manifest, entity_group_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the frozen MRO split manifest")
    parser.add_argument("--data-dir", default="data/generated/mock_v0_2_0")
    parser.add_argument("--output", default="artifacts/splits/mock_v0_2_0")
    parser.add_argument("--train-end", default="2024-09-01")
    parser.add_argument("--calibration-end", default="2025-03-01")
    parser.add_argument("--validation-end", default="2025-09-01")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()

    frame = pd.read_parquet(Path(args.data_dir) / "mro_dataset.parquet")
    manifest = create_joint_split_manifest(
        frame,
        train_end=args.train_end,
        calibration_end=args.calibration_end,
        validation_end=args.validation_end,
        seed=args.seed,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output / "split_manifest.parquet", index=False)
    joined = manifest.merge(
        frame[["snapshot_id", "removal_event_id", "unscheduled_removal_within_30_cycles"]],
        on="snapshot_id",
        validate="one_to_one",
    )
    diagnostics = entity_group_diagnostics(joined)
    diagnostics.to_csv(output / "entity_group_diagnostics.csv", index=False)
    summary = {
        "data_dir": args.data_dir,
        "train_end": args.train_end,
        "calibration_end": args.calibration_end,
        "validation_end": args.validation_end,
        "seed": args.seed,
        "entity_groups": len(diagnostics),
        "max_group_snapshot_share": float(diagnostics["snapshot_share"].max()),
        "uses": {
            name: {
                "snapshots": int(len(group)),
                "positive_snapshots": int(group["unscheduled_removal_within_30_cycles"].sum()),
                "events": int(group["removal_event_id"].nunique()),
            }
            for name, group in joined.groupby("final_use")
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
