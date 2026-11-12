from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def package_dataset(data_dir: str | Path, output: str | Path) -> dict[str, object]:
    source = Path(data_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in source.iterdir()
        if path.suffix in {".csv", ".parquet", ".yaml", ".json"}
    )
    if not files:
        raise ValueError("No dataset files were found")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=f"mro_mock_dataset_v0_2_0/{path.name}")
        archive.write("DATASET.md", arcname="mro_mock_dataset_v0_2_0/DATASET.md")
    with zipfile.ZipFile(output_path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC validation failed: {corrupt}")
        members = len(archive.infolist())
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_version": "0.2.0",
        "archive": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": digest,
        "members": members,
        "crc_validated": True,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the MRO mock dataset")
    parser.add_argument("--data-dir", default="data/generated/mock_v0_2_0")
    parser.add_argument("--output", default="artifacts/packages/mro_mock_dataset_v0_2_0.zip")
    args = parser.parse_args()
    print(json.dumps(package_dataset(args.data_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
