from __future__ import annotations

import argparse
import json

from src.mock_data import dataset_summary, load_config
from src.validation import read_dataset, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated MRO dataset")
    parser.add_argument("--data-dir", default="data/generated/mock_v0_2_0")
    args = parser.parse_args()

    frames = read_dataset(args.data_dir)
    config = load_config(f"{args.data_dir}/generation_config.yaml")
    checks = validate_dataset(frames, config)
    print(json.dumps({"checks": checks, "summary": dataset_summary(frames, config)}, indent=2))


if __name__ == "__main__":
    main()
