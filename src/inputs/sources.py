"""Concrete LiDAR sources and the registry that builds them from a config.

Three source types ship built in:

  folder         any directory of point-cloud files (bin/npy/npz/ply/pcd/csv/
                 txt), with optional per-frame label files. Column layout,
                 axis orientation, height offset, intensity scale, timestamps
                 and label maps are all configurable, so most datasets and
                 sensor dumps work without writing code.
  rellis3d       the RELLIS-3D layout (Rellis-3D/<seq>/os1_cloud_node_*).
  semantickitti  the SemanticKITTI layout (sequences/<seq>/velodyne|labels).

Add another dataset by subclassing LidarSource and calling
register_source_type("name", factory) (see docs/input_sources.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.inputs.base import DEFAULT_SENSOR_HZ, LidarFrame, LidarSource, check_index
from src.inputs.formats import (
    SUPPORTED_POINT_FORMATS,
    detect_format,
    map_labels,
    natural_key,
    read_labels,
    read_points,
    to_pipeline_points,
)

_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _check_keys(d: dict, allowed: set, where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {where}. Allowed keys: {sorted(allowed)}"
        )


def _builtin_label_map(name: str) -> dict:
    from src.perception.semantic_labels import RELLIS3D_TO_FOVEAX, SEMANTICKITTI_TO_FOVEAX

    maps = {"semantickitti": SEMANTICKITTI_TO_FOVEAX, "rellis3d": RELLIS3D_TO_FOVEAX}
    if name == "foveax":  # labels are already FOVEAX class IDs
        return {i: i for i in range(8)}
    if name not in maps:
        raise ValueError(f"Unknown built-in label map {name!r}; use one of {sorted(maps) + ['foveax']} or give a dict")
    return dict(maps[name])


# --------------------------------------------------------------------- folder
_FOLDER_KEYS = {
    "type", "path", "pattern", "recursive", "format", "n_cols", "skip_header", "columns", "axes",
    "z_offset", "intensity_scale", "drop_nonfinite", "drop_zero_returns", "min_range_m",
    "max_range_m", "timestamps", "sensor", "labels", "name",
}


class FolderSource(LidarSource):
    def __init__(self, cfg: dict, base_dir: Path | None = None):
        _check_keys(cfg, _FOLDER_KEYS, "the 'folder' source")
        if "path" not in cfg:
            raise ValueError("The 'folder' source needs a 'path' to the directory with the scans")
        root = Path(cfg["path"])
        if not root.is_absolute() and base_dir is not None:
            root = base_dir / root
        if not root.is_dir():
            raise FileNotFoundError(f"Input folder not found: {root}")
        self.root = root
        self.name = cfg.get("name", f"folder:{root.name}")

        self.fmt = cfg.get("format")
        pattern = cfg.get("pattern")
        recursive = bool(cfg.get("recursive", False))
        if pattern is None:
            pattern = f"*.{self.fmt}" if self.fmt else self._auto_pattern(root, recursive)
        finder = root.rglob if recursive else root.glob
        self.files = sorted((p for p in finder(pattern) if p.is_file()), key=natural_key)
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} under {root}")
        self.fmt = self.fmt or detect_format(self.files[0])
        if self.fmt not in SUPPORTED_POINT_FORMATS:
            raise ValueError(f"format must be one of {SUPPORTED_POINT_FORMATS}, got {self.fmt!r}")

        self.n_cols = int(cfg.get("n_cols", 4))
        self.skip_header = cfg.get("skip_header")
        cols = cfg.get("columns", {"x": 0, "y": 1, "z": 2, "intensity": 3})
        if isinstance(cols, dict):
            _check_keys(cols, {"x", "y", "z", "intensity"}, "'columns'")
            cols = (cols.get("x", 0), cols.get("y", 1), cols.get("z", 2), cols.get("intensity", 3))
        cols = tuple(cols)
        if len(cols) != 4:
            raise ValueError("'columns' must give x, y, z and intensity column indices (intensity may be null)")
        self.columns = cols
        self.geometry = dict(
            axes=cfg.get("axes", "x,y,z"),
            z_offset=float(cfg.get("z_offset", 0.0)),
            intensity_scale=float(cfg.get("intensity_scale", 1.0)),
            drop_nonfinite=bool(cfg.get("drop_nonfinite", True)),
            drop_zero_returns=bool(cfg.get("drop_zero_returns", False)),
            min_range_m=float(cfg.get("min_range_m", 0.0)),
            max_range_m=cfg.get("max_range_m"),
        )

        sensor = dict(cfg.get("sensor", {}))
        _check_keys(sensor, {"name", "hz", "beams", "width", "fov_up", "fov_down"}, "'sensor'")
        self.sensor = sensor
        self.sensor_hz = float(sensor.get("hz", DEFAULT_SENSOR_HZ))
        if self.sensor_hz <= 0:
            raise ValueError("sensor.hz must be positive")

        self._init_labels(cfg.get("labels"), base_dir)
        self._init_timestamps(cfg.get("timestamps"), base_dir)

    # -- setup helpers
    @staticmethod
    def _auto_pattern(root: Path, recursive: bool) -> str:
        finder = root.rglob if recursive else root.glob
        counts: dict[str, int] = {}
        for p in finder("*"):
            if p.is_file() and p.suffix.lower() in {f".{f}" for f in SUPPORTED_POINT_FORMATS}:
                counts[p.suffix.lower()] = counts.get(p.suffix.lower(), 0) + 1
        if not counts:
            raise FileNotFoundError(
                f"No point-cloud files ({', '.join(SUPPORTED_POINT_FORMATS)}) found under {root}"
            )
        return "*" + max(counts, key=counts.get)

    def _init_labels(self, lab: dict | None, base_dir: Path | None) -> None:
        self.label_dir = None
        if not lab:
            return
        _check_keys(lab, {"dir", "ext", "format", "map", "unknown_id"}, "'labels'")
        ldir = Path(lab.get("dir", "."))
        if not ldir.is_absolute():
            ldir = self.root / ldir
        if not ldir.is_dir():
            raise FileNotFoundError(f"Label folder not found: {ldir}")
        self.label_dir = ldir
        self.label_ext = lab.get("ext", ".label")
        self.label_fmt = lab.get("format", "semantickitti")
        m = lab.get("map", "semantickitti")
        self.label_map = _builtin_label_map(m) if isinstance(m, str) else {int(k): int(v) for k, v in m.items()}
        self.unknown_id = int(lab.get("unknown_id", 7))

    def _init_timestamps(self, ts: dict | None, base_dir: Path | None) -> None:
        self._ts: np.ndarray | None = None
        if not ts or ts.get("mode", "index") == "index":
            return
        _check_keys(ts, {"mode", "unit", "file"}, "'timestamps'")
        mode, unit = ts["mode"], ts.get("unit", "s")
        if unit not in _TIME_UNITS:
            raise ValueError(f"timestamps.unit must be one of {sorted(_TIME_UNITS)}")
        if mode == "filename":
            try:
                vals = np.array([float(p.stem) for p in self.files], dtype=np.float64)
            except ValueError as e:
                raise ValueError("timestamps.mode 'filename' needs numeric file names (e.g. 1651234567.123456.bin)") from e
        elif mode == "file":
            tf = Path(ts["file"])
            if not tf.is_absolute() and base_dir is not None:
                tf = base_dir / tf
            vals = np.loadtxt(tf, dtype=np.float64, ndmin=1)
            if vals.size != len(self.files):
                raise ValueError(f"timestamps file has {vals.size} entries but there are {len(self.files)} scans")
        else:
            raise ValueError("timestamps.mode must be 'index', 'filename' or 'file'")
        vals = vals * _TIME_UNITS[unit]
        if np.any(np.diff(vals) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        self._ts = vals - vals[0]

    # -- LidarSource API
    def __len__(self) -> int:
        return len(self.files)

    def timestamp(self, idx: int) -> float:
        check_index(idx, len(self))
        return float(self._ts[idx]) if self._ts is not None else idx / self.sensor_hz

    def frame(self, idx: int) -> LidarFrame:
        check_index(idx, len(self))
        path = self.files[idx]
        raw = read_points(path, self.fmt, self.n_cols, self.skip_header)
        pts, keep = to_pipeline_points(raw, columns=self.columns, **self.geometry)
        labels = None
        if self.label_dir is not None:
            lp = self.label_dir / (path.stem + self.label_ext)
            if lp.exists():
                ids = read_labels(lp, self.label_fmt)
                if ids.shape[0] != raw.shape[0]:
                    raise ValueError(
                        f"{lp.name} has {ids.shape[0]} labels but {path.name} has {raw.shape[0]} points"
                    )
                labels = map_labels(ids, self.label_map, self.unknown_id)[keep]
        return LidarFrame(pts, labels, self.timestamp(idx), path.stem, {"file": str(path)})


# -------------------------------------------------------------------- rellis3d
class Rellis3DSource(LidarSource):
    name = "rellis3d"
    sensor = {"name": "Ouster OS1-64", "beams": 64, "width": 2048, "fov_up": 17.02, "fov_down": -16.44}
    sensor_hz = DEFAULT_SENSOR_HZ

    def __init__(self, cfg: dict, base_dir: Path | None = None):
        _check_keys(cfg, {"type", "root", "sequence", "sensor_hz"}, "the 'rellis3d' source")
        root = Path(cfg.get("root", "C:/dev/data/rellis3d"))
        if not root.exists() and Path("data/rellis3d").exists():
            root = Path("data/rellis3d")
        if not root.is_absolute() and base_dir is not None and (base_dir / root).exists():
            root = base_dir / root
        self.root = root
        self.sequence = str(cfg.get("sequence", "00001"))
        self.sensor_hz = float(cfg.get("sensor_hz", DEFAULT_SENSOR_HZ))
        seq_dir = root / "Rellis-3D" / self.sequence / "os1_cloud_node_kitti_bin"
        self.files = sorted(seq_dir.glob("*.bin"), key=natural_key) if seq_dir.is_dir() else []
        if not self.files:
            raise FileNotFoundError(f"No RELLIS-3D scans found in {seq_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def frame(self, idx: int) -> LidarFrame:
        check_index(idx, len(self))
        path = self.files[idx]
        try:
            from src.perception.rellis3d_loader import load_rellis3d_frame

            rf = load_rellis3d_frame(self.root, self.sequence, path.stem)
            pts, labels = rf.points, rf.foveax_class_ids
        except (FileNotFoundError, OSError):  # scans without label files still play
            pts, labels = np.fromfile(path, dtype=np.float32).reshape(-1, 4), None
        return LidarFrame(pts, labels, self.timestamp(idx), path.stem, {"file": str(path)})


# --------------------------------------------------------------- semantickitti
class SemanticKittiSource(LidarSource):
    name = "semantickitti"
    sensor = {"name": "Velodyne HDL-64E", "beams": 64, "width": 2048, "fov_up": 3.0, "fov_down": -25.0}
    sensor_hz = DEFAULT_SENSOR_HZ

    def __init__(self, cfg: dict, base_dir: Path | None = None):
        _check_keys(cfg, {"type", "root", "sequence", "sensor_hz"}, "the 'semantickitti' source")
        root = Path(cfg.get("root", "data/semantic_kitti/dataset/sequences"))
        if not root.is_absolute() and base_dir is not None and (base_dir / root).exists():
            root = base_dir / root
        self.sequence = str(cfg.get("sequence", "00"))
        self.sensor_hz = float(cfg.get("sensor_hz", DEFAULT_SENSOR_HZ))
        self.seq_dir = root / self.sequence
        self.files = sorted((self.seq_dir / "velodyne").glob("*.bin"), key=natural_key)
        if not self.files:
            raise FileNotFoundError(f"No SemanticKITTI scans found in {self.seq_dir / 'velodyne'}")

    def __len__(self) -> int:
        return len(self.files)

    def frame(self, idx: int) -> LidarFrame:
        check_index(idx, len(self))
        path = self.files[idx]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        labels = None
        lp = self.seq_dir / "labels" / f"{path.stem}.label"
        if lp.exists():
            ids = read_labels(lp, "semantickitti")
            labels = map_labels(ids, _builtin_label_map("semantickitti"))
        return LidarFrame(pts, labels, self.timestamp(idx), path.stem, {"file": str(path)})


# ----------------------------------------------------------------------- arrays
class ArraySource(LidarSource):
    """Frames held in memory: points (F, N, 4) and optional FOVEAX labels (F, N).

    For tests and notebooks (and small fixtures committed to the repo). It is
    picklable, so it can also be handed to the worker processes.
    """

    name = "arrays"

    def __init__(self, points, labels=None, sensor_hz: float = DEFAULT_SENSOR_HZ, name: str = "arrays"):
        self.points = np.asarray(points, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.uint8)
        if self.points.ndim != 3 or self.points.shape[2] < 4:
            raise ValueError("points must have shape (frames, points, 4)")
        if self.labels is not None and self.labels.shape != self.points.shape[:2]:
            raise ValueError("labels must have shape (frames, points)")
        self.sensor_hz = float(sensor_hz)
        self.name = name
        self.sensor = {}

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def frame(self, idx: int) -> LidarFrame:
        check_index(idx, len(self))
        labels = None if self.labels is None else self.labels[idx]
        return LidarFrame(self.points[idx][:, :4], labels, self.timestamp(idx), str(idx))


# -------------------------------------------------------------------- registry
SourceFactory = Callable[[dict, "Path | None"], LidarSource]
SOURCE_TYPES: dict[str, SourceFactory] = {
    "folder": FolderSource,
    "rellis3d": Rellis3DSource,
    "semantickitti": SemanticKittiSource,
}


def register_source_type(name: str, factory: SourceFactory) -> None:
    """Make a custom source available as `type: <name>` in input configs."""
    SOURCE_TYPES[name] = factory


def create_source(spec: dict, base_dir: Path | None = None) -> LidarSource:
    spec = dict(spec)
    stype = spec.get("type", "folder")
    if stype not in SOURCE_TYPES:
        raise ValueError(f"Unknown source type {stype!r}. Available: {sorted(SOURCE_TYPES)}")
    return SOURCE_TYPES[stype](spec, base_dir)


def load_source_config(path: str | Path) -> LidarSource:
    """Build a source from a YAML or JSON file (relative paths resolve
    against the config file's own folder)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        spec = json.loads(text)
    else:
        import yaml

        spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError(f"{p.name}: expected a mapping at the top level")
    return create_source(spec, p.resolve().parent)


def source_from_dir(path: str | Path, **overrides: Any) -> LidarSource:
    """Quick path: point at a folder of scans; format is detected."""
    return create_source({"type": "folder", "path": str(path), **overrides})
