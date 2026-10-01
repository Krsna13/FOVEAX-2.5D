"""Readers for common LiDAR point-cloud and label file formats, plus the
geometric conversion into FOVEAX's vehicle-frame convention."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SUPPORTED_POINT_FORMATS = ("bin", "npy", "npz", "ply", "pcd", "csv", "txt", "xyz")
_SUFFIX_TO_FORMAT = {
    ".bin": "bin", ".npy": "npy", ".npz": "npz", ".ply": "ply", ".pcd": "pcd",
    ".csv": "csv", ".txt": "txt", ".xyz": "xyz",
}


def detect_format(path: Path) -> str:
    fmt = _SUFFIX_TO_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"Cannot tell the point-cloud format of {path.name!r} from its extension. "
            f"Set 'format' to one of {SUPPORTED_POINT_FORMATS}."
        )
    return fmt


def natural_key(p: Path):
    """Sort 2.bin before 10.bin (plain string sort would not)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def read_points(path: Path, fmt: str | None = None, n_cols: int = 4,
                skip_header: int | None = None) -> np.ndarray:
    """Read a point cloud as a 2-D float array with at least 3 columns."""
    fmt = fmt or detect_format(path)
    if fmt == "bin":
        raw = np.fromfile(path, dtype=np.float32)
        if n_cols < 3 or raw.size % n_cols != 0:
            raise ValueError(
                f"{path.name}: {raw.size} float32 values do not divide into rows of "
                f"{n_cols} columns. Set 'n_cols' to the number of values stored per "
                "point (KITTI = 4, nuScenes = 5, Ouster/Velodyne dumps = 4)."
            )
        arr = raw.reshape(-1, n_cols)
    elif fmt == "npy":
        arr = np.load(path)
    elif fmt == "npz":
        with np.load(path) as z:
            key = "points" if "points" in z.files else z.files[0]
            arr = z[key]
    elif fmt in ("ply", "pcd"):
        arr = _read_open3d(path)
    elif fmt in ("csv", "txt", "xyz"):
        arr = _read_text(path, skip_header)
    else:
        raise ValueError(f"Unsupported point format {fmt!r}; use one of {SUPPORTED_POINT_FORMATS}")
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path.name}: expected an (N, >=3) point array, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _read_open3d(path: Path) -> np.ndarray:
    import open3d as o3d  # lazy: only needed for .ply/.pcd

    pcd = o3d.t.io.read_point_cloud(str(path))
    xyz = pcd.point.positions.numpy()
    if "intensity" in pcd.point:
        inten = pcd.point.intensity.numpy().reshape(-1, 1)
    else:
        inten = np.zeros((xyz.shape[0], 1), dtype=xyz.dtype)
    return np.hstack([xyz, inten])


def _read_text(path: Path, skip_header: int | None) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                first = line
                break
    delim = "," if "," in first else None
    if skip_header is None:  # a non-numeric first line is a header row
        try:
            [float(t) for t in first.replace(",", " ").split()]
            skip_header = 0
        except ValueError:
            skip_header = 1
    return np.loadtxt(path, delimiter=delim, skiprows=skip_header, ndmin=2, comments="#")


# ------------------------------------------------------------------ geometry
_AXIS_RE = re.compile(r"^-?[xyz]$")


def parse_axes(spec: str) -> tuple[tuple[int, float], ...]:
    """'x,y,z' / '-y,x,z' -> ((source_col, sign), ...) for the output x, y, z.

    Reads as: output x = (sign) * that source axis, and so on. Use it when a
    sensor's own frame is not x-forward, y-left, z-up.
    """
    toks = [t.strip().lower() for t in spec.split(",")]
    if len(toks) != 3 or not all(_AXIS_RE.match(t) for t in toks):
        raise ValueError(f"axes must look like 'x,y,z' or '-y,x,z' (got {spec!r})")
    if len({t.lstrip("-") for t in toks}) != 3:
        raise ValueError(f"axes must use each of x, y, z exactly once (got {spec!r})")
    return tuple(("xyz".index(t.lstrip("-")), -1.0 if t.startswith("-") else 1.0) for t in toks)


def to_pipeline_points(
    raw: np.ndarray,
    *,
    columns: tuple = (0, 1, 2, 3),
    axes: str = "x,y,z",
    z_offset: float = 0.0,
    intensity_scale: float = 1.0,
    drop_nonfinite: bool = True,
    drop_zero_returns: bool = False,
    min_range_m: float = 0.0,
    max_range_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw file columns into the pipeline's (N, 4) float32 array.

    Returns (points, keep_mask) where keep_mask has one entry per *raw* row so
    the caller can filter labels the same way.
    """
    n = raw.shape[0]
    xi, yi, zi, ii = columns
    for c in (xi, yi, zi) + ((ii,) if ii is not None else ()):
        if not 0 <= c < raw.shape[1]:
            raise ValueError(
                f"column index {c} is outside the file's {raw.shape[1]} columns; check 'columns'/'n_cols'"
            )
    src = np.stack([raw[:, xi], raw[:, yi], raw[:, zi]], axis=1).astype(np.float32, copy=False)
    xyz = np.empty_like(src)
    for out_i, (col, sign) in enumerate(parse_axes(axes)):
        xyz[:, out_i] = sign * src[:, col]
    xyz[:, 2] += np.float32(z_offset)
    inten = (raw[:, ii].astype(np.float32) * np.float32(intensity_scale)
             if ii is not None else np.zeros(n, dtype=np.float32))

    keep = np.ones(n, dtype=bool)
    if drop_nonfinite:
        keep &= np.isfinite(xyz).all(axis=1)
    if drop_zero_returns:  # sensors that pad missing returns with (0, 0, 0)
        keep &= (src != 0).any(axis=1)
    if min_range_m > 0.0 or max_range_m is not None:
        with np.errstate(invalid="ignore"):
            r = np.hypot(xyz[:, 0], xyz[:, 1])
        if min_range_m > 0.0:
            keep &= r >= min_range_m
        if max_range_m is not None:
            keep &= r <= max_range_m
    pts = np.column_stack([xyz, np.nan_to_num(inten)]).astype(np.float32)
    return pts[keep], keep


# -------------------------------------------------------------------- labels
def read_labels(path: Path, fmt: str) -> np.ndarray:
    """Read per-point raw class IDs (before mapping to FOVEAX classes)."""
    if fmt == "semantickitti":  # uint32: lower 16 bits class, upper 16 instance
        return (np.fromfile(path, dtype=np.uint32) & 0xFFFF).astype(np.int64)
    if fmt in ("uint8", "uint16", "int32", "uint32"):
        return np.fromfile(path, dtype=np.dtype(fmt)).astype(np.int64)
    if fmt == "npy":
        return np.load(path).reshape(-1).astype(np.int64)
    raise ValueError(f"Unsupported label format {fmt!r}")


def map_labels(raw_ids: np.ndarray, mapping: dict, unknown_id: int = 7) -> np.ndarray:
    """Vectorized raw-ID -> FOVEAX-ID remap; unmapped IDs become UNKNOWN."""
    if raw_ids.size == 0:
        return raw_ids.astype(np.uint8)
    hi = int(max(raw_ids.max(), max(mapping) if mapping else 0))
    lut = np.full(max(hi, 0) + 1, unknown_id, dtype=np.uint8)
    for k, v in mapping.items():
        if k >= 0:
            lut[int(k)] = int(v)
    ids = np.where(raw_ids < 0, 0, raw_ids)
    out = lut[ids]
    out[raw_ids < 0] = unknown_id
    return out
