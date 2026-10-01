"""Helpers for golden-output (regression) tests of the perception pipeline.

A golden file stores, for a fixed list of real frames, everything the pipeline
produces that is *perception* (grids, hazards, tracks, classes, profiles,
accuracy), so a later change that alters any of it is caught. Files are npz
(arrays) plus a JSON tree, not pickles, so they load on any numpy version.

Comparison is exact for structure, integers, booleans, strings and NaN
positions, and uses a tight tolerance (rtol 1e-5) for floating point values so
the test does not fail on another CPU's rounding. The shortcuts it exists to
catch change grids, counts and metadata by whole cells/objects, far outside
that tolerance.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

# Fields that are measurements or display-only data, not perception output.
IGNORED_FIELDS = ("metrics", "display_points", "schema_version")


def plain(state) -> dict:
    """FrameState -> nested plain structure (arrays copied, dataclasses to dicts),
    without measurement/display fields and without the raw cloud (its checksum is kept)."""
    d = _to_plain(state)
    pts = d.pop("points")
    for k in IGNORED_FIELDS:
        d.pop(k, None)
    d["n_points"] = int(len(pts))
    d["points_sum"] = float(np.asarray(pts, dtype=np.float64).sum())
    return normalize(d)


def _to_plain(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _to_plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.copy()
    return x


def normalize(x):
    """Tuples -> lists, numpy scalars -> python scalars (dict keys kept)."""
    if isinstance(x, dict):
        return {k: normalize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [normalize(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


# ------------------------------------------------------------------ storage
def _encode(x, arrays: dict):
    if isinstance(x, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = x
        return {"__nd__": key}
    if isinstance(x, dict):
        return {"__dict__": [[k, _encode(v, arrays)] for k, v in x.items()]}
    if isinstance(x, list):
        return [_encode(v, arrays) for v in x]
    return x


def _decode(x, arrays):
    if isinstance(x, dict):
        if "__nd__" in x:
            return arrays[x["__nd__"]]
        return {k: _decode(v, arrays) for k, v in x["__dict__"]}
    if isinstance(x, list):
        return [_decode(v, arrays) for v in x]
    return x


def save_golden(path: Path, frames: list[dict]) -> None:
    arrays: dict = {}
    tree = _encode(frames, arrays)
    blob = np.frombuffer(json.dumps(tree).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, __tree__=blob, **arrays)


def load_golden(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as z:
        tree = json.loads(bytes(z["__tree__"]).decode("utf-8"))
        arrays = {k: z[k] for k in z.files if k != "__tree__"}
    return _decode(tree, arrays)


# --------------------------------------------------------------- comparison
def assert_same(actual, expected, path="frame", rtol=1e-5, atol=1e-9) -> None:
    """Raise AssertionError naming the first differing field."""
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        a, e = np.asarray(actual), np.asarray(expected)
        assert a.shape == e.shape, f"{path}: shape {a.shape} != {e.shape}"
        if a.dtype.kind == "f" or e.dtype.kind == "f":
            same_nan = np.array_equal(np.isnan(a), np.isnan(e))
            assert same_nan, f"{path}: NaN positions differ ({int((np.isnan(a) != np.isnan(e)).sum())} cells)"
            m = ~np.isnan(a)
            ok = np.allclose(a[m], e[m], rtol=rtol, atol=atol)
            assert ok, f"{path}: float values differ (max abs diff {np.max(np.abs(a[m] - e[m])):.3g})"
        else:
            nd = int((a != e).sum())
            assert nd == 0, f"{path}: {nd} of {a.size} values differ"
    elif isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), (
            f"{path}: keys differ: {sorted(set(actual) ^ set(expected), key=str)}")
        for k in actual:
            assert_same(actual[k], expected[k], f"{path}.{k}", rtol, atol)
    elif isinstance(actual, list):
        assert isinstance(expected, list) and len(actual) == len(expected), (
            f"{path}: length {len(actual)} != {len(expected) if isinstance(expected, list) else expected!r}")
        for i, (x, y) in enumerate(zip(actual, expected)):
            assert_same(x, y, f"{path}[{i}]", rtol, atol)
    elif isinstance(actual, float) or isinstance(expected, float):
        if isinstance(actual, float) and isinstance(expected, float) and math.isnan(actual) and math.isnan(expected):
            return
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and math.isclose(
            actual, expected, rel_tol=rtol, abs_tol=atol), f"{path}: {actual!r} != {expected!r}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
