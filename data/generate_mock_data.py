from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mock_data import build_dataset, load_config, write_dataset
from src.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the MRO mock dataset")
    parser.add_argument("--config", default="configs/mock_data.yaml")
    parser.add_argument("--output", default="data/generated/mock_v0_2_0")
    parser.add_argument("--combined-output", default="data/mro_dataset.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    frames = build_dataset(config)
    checks = validate_dataset(frames, config)
    manifest = write_dataset(frames, config, args.output)
    combined_output = Path(args.combined_output)
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    frames["mro_dataset"].to_csv(combined_output, index=False)
    print(json.dumps({"checks": checks, **manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
