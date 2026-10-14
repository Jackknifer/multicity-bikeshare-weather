import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

WORK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = WORK_ROOT / "configs"
DEFAULT_PATHS_FILE = CONFIG_DIR / "paths.local.yaml"
EXAMPLE_PATHS_FILE = CONFIG_DIR / "paths.example.yaml"


def load_paths(path=None):
    candidate = Path(path) if path else Path(os.environ.get("BIKESHARE_PATHS", DEFAULT_PATHS_FILE))
    if not candidate.is_file():
        raise FileNotFoundError(f"paths configuration not found: {candidate}")
    config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    required = ["project_root", "midterm_root", "work_root", "data_csv", "data_sha256"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"paths configuration missing keys: {missing}")
    return {key: (str(value) if not isinstance(value, (int, float)) else value) for key, value in config.items()}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_input(paths):
    data_path = Path(paths["data_csv"])
    if not data_path.is_file():
        raise FileNotFoundError(f"frozen input missing: {data_path}")
    actual = sha256_file(data_path)
    expected = paths["data_sha256"]
    if actual != expected:
        raise ValueError(f"frozen input hash mismatch: expected {expected}, found {actual}")
    return {"path": str(data_path), "sha256": actual, "bytes": data_path.stat().st_size}


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return target


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_revision(path):
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def environment_snapshot():
    import arviz
    import numpy
    import pandas
    import pymc
    import pytensor
    import scipy

    snapshot = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "pymc": pymc.__version__,
        "pytensor": pytensor.__version__,
        "arviz": arviz.__version__,
    }
    try:
        import nutpie

        snapshot["nutpie"] = nutpie.__version__
    except ModuleNotFoundError:
        snapshot["nutpie"] = None
    try:
        import netCDF4

        snapshot["netCDF4"] = netCDF4.__version__
    except ModuleNotFoundError:
        snapshot["netCDF4"] = None
    return snapshot
