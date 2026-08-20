#!/usr/bin/env python3
"""Stable, fail-closed fingerprints for RACER initialization observations."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


class ObservationFingerprintError(ValueError):
    pass


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def fingerprint(value: Any, *, path: str = "root", ancestors: set[int] | None = None):
    if ancestors is None:
        ancestors = set()
    if value is None or isinstance(value, (bool, str, int)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ObservationFingerprintError(f"non-finite float at {path}")
        return {"type": "float", "value_hex": value.hex()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "type": "bytes",
            "length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise ObservationFingerprintError(f"object scalar is unsupported at {path}")
        return fingerprint(value.item(), path=path, ancestors=ancestors)

    tensor_like = value
    if all(hasattr(tensor_like, name) for name in ("detach", "cpu", "numpy")):
        value = tensor_like.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ObservationFingerprintError(f"object array is unsupported at {path}")
        contiguous = np.ascontiguousarray(value)
        if np.issubdtype(contiguous.dtype, np.number) and not np.isfinite(contiguous).all():
            raise ObservationFingerprintError(f"non-finite numeric array at {path}")
        return {
            "type": "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }

    identity = id(value)
    if identity in ancestors:
        raise ObservationFingerprintError(f"cycle detected at {path}")
    nested_ancestors = ancestors | {identity}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ObservationFingerprintError(f"non-string mapping key at {path}")
        return {
            "type": "dict",
            "items": {
                key: fingerprint(
                    value[key], path=f"{path}.{key}", ancestors=nested_ancestors
                )
                for key in sorted(value)
            },
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "items": [
                fingerprint(item, path=f"{path}[{index}]", ancestors=nested_ancestors)
                for index, item in enumerate(value)
            ],
        }
    if hasattr(value, "__dict__"):
        attributes = vars(value)
        return {
            "type": "object",
            "class": _qualified_type(value),
            "attributes": fingerprint(
                attributes, path=f"{path}.__dict__", ancestors=nested_ancestors
            ),
        }
    raise ObservationFingerprintError(
        f"unsupported observation value {_qualified_type(value)} at {path}"
    )


def snapshot(obs_dict: Any, observation: Any):
    return {
        "obs_dict": fingerprint(obs_dict, path="obs_dict"),
        "observation": fingerprint(observation, path="observation"),
    }
