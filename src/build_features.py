from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from src.features import add_causal_rolling_features, model_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal snapshot features")
    parser.add_argument("--data-dir", default="data/generated/mock_v0_2_0")
    parser.add_argument("--output", default="artifacts/features/mock_v0_2_0/features.parquet")
    args = parser.parse_args()

    input_path = Path(args.data_dir) / "mro_dataset.parquet"
    started = time.perf_counter()
    frame = pd.read_parquet(input_path)
    features = add_causal_rolling_features(frame)
    numeric, categorical = model_feature_columns(features)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    metadata = {
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output),
        "rows": len(features),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (output.parent / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps({**metadata, "numeric_features": len(numeric)}, indent=2))


if __name__ == "__main__":
    main()
