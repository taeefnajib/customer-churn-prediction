from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_joblib(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


