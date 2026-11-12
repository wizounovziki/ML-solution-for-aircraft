from __future__ import annotations

import argparse
import json

from src.training import run_validation_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, calibrate, and select MRO models")
    parser.add_argument("--features", default="artifacts/features/mock_v0_2_0/features.parquet")
    parser.add_argument("--split", default="artifacts/splits/mock_v0_2_0/split_manifest.parquet")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = run_validation_selection(args.features, args.split, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
