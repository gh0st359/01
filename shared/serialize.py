"""Schema-versioned serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import orjson


SCHEMA_VERSION = 1


def _intish(value: int) -> Any:
    if value > 2**63 - 1 or value < -(2**63):
        return {"__bigint__": str(value)}
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "data": to_jsonable(value.tolist())}
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return _intish(int(value))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, int):
        return _intish(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value
    return value


def from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            return np.array(from_jsonable(value["data"]), dtype=value.get("dtype", "float64"))
        if "__bigint__" in value and len(value) == 1:
            return int(value["__bigint__"])
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_bytes(orjson.dumps(to_jsonable(payload), option=orjson.OPT_INDENT_2))


def read_json(path: str | Path) -> Any:
    return from_jsonable(orjson.loads(Path(path).read_bytes()))


def write_text_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
