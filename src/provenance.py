from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def verify_frozen_experiment(experiment: str | Path) -> dict[str, Any]:
    root = Path(experiment)
    freeze = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    if freeze.get("model_sha256") != sha256(root / "frozen_model.joblib"):
        raise ValueError("Model artifact differs from its frozen hash")
    if freeze.get("source_fingerprint") != source_fingerprint():
        raise ValueError("Source fingerprint differs from the frozen implementation")
    return freeze
