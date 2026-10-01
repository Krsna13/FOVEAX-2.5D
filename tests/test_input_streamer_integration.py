"""The dashboard's DataStreamerThread must run end to end on a custom source,
and speeds must come from the sensor's rate, not the playback rate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PyQt5", reason="dashboard needs PyQt5")

from src.dashboard.data_streamer import DataStreamerThread  # noqa: E402
from src.inputs import create_source  # noqa: E402
from src.perception.object_detector import MockObjectDetector  # noqa: E402


def _scene(n_frames, speed_mps, hz, tmp_path, extra_cols=0):
    """Flat ground plus one box-shaped obstacle moving along +x at speed_mps."""
    d = tmp_path / "scans"
    d.mkdir()
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(-15, 15, 0.5), np.arange(-15, 15, 0.5))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    for i in range(n_frames):
        cx = 5.0 + speed_mps * i / hz
        box = np.column_stack([
            cx + rng.uniform(-0.4, 0.4, 200), 3.0 + rng.uniform(-0.4, 0.4, 200), rng.uniform(0.4, 1.6, 200)])
        pts = np.vstack([ground, box])
        arr = np.column_stack([pts, np.ones(len(pts))] + [np.zeros(len(pts))] * extra_cols)
        arr.astype(np.float32).tofile(d / f"{i:04d}.bin")
    return d


def _streamer(src, rate_hz=30.0):
    s = DataStreamerThread(source="rellis3d", detector=MockObjectDetector(), num_frames=len(src),
                           rate_hz=rate_hz, lidar_source=src)
    s._setup_sources()
    return s


def test_custom_source_runs_the_whole_pipeline(tmp_path):
    d = _scene(6, 2.0, 10, tmp_path, extra_cols=1)  # 5 columns per point
    src = create_source({"type": "folder", "path": str(d), "n_cols": 5, "sensor": {"hz": 10}})
    s = _streamer(src)
    assert s.source == "custom" and s.available_frames() == 6
    for i in range(6):
        state = s.process_frame(i, s.timestamp_for(i))
    assert len(state.points) > 1000
    assert state.grid_maps["elevation"].shape[0] > 0
    assert state.has_ground_truth is False  # no labels supplied -> nothing faked


def test_speed_uses_sensor_rate_not_playback_rate(tmp_path):
    """A 2 m/s obstacle must read ~2 m/s whether it is replayed at 10 or 30 Hz."""
    speeds = []
    for playback_hz in (10.0, 30.0):
        sub = tmp_path / f"pb{int(playback_hz)}"
        sub.mkdir()
        d = _scene(12, 2.0, 10, sub)
        src = create_source({"type": "folder", "path": str(d), "sensor": {"hz": 10}})
        s = _streamer(src, rate_hz=playback_hz)
        for i in range(12):
            state = s.process_frame(i, s.timestamp_for(i), fps=playback_hz)
        moving = [t for t in state.tracks if abs(float(t.state[3])) > 0.3]
        assert moving, "the moving obstacle should be tracked with a velocity"
        speeds.append(abs(float(moving[0].state[3])))
    assert speeds[0] == pytest.approx(speeds[1], rel=0.05)
    assert 1.4 < speeds[0] < 2.6


def test_timestamp_for_is_strictly_increasing_past_the_end(tmp_path):
    d = _scene(3, 1.0, 10, tmp_path)
    s = _streamer(create_source({"type": "folder", "path": str(d), "sensor": {"hz": 20}}))
    ts = [s.timestamp_for(i) for i in range(8)]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert ts[1] == pytest.approx(0.05)


def test_builtin_sources_keep_their_behaviour():
    s = DataStreamerThread(source="sample", detector=MockObjectDetector(), num_frames=3)
    s._setup_sources()
    assert s.lidar_source is None and s.sensor_hz == 10.0
    assert s.timestamp_for(5) == pytest.approx(0.5)
    state = s.process_frame(0, s.timestamp_for(0))
    assert len(state.points) > 0


def test_cli_resolve_input_sets_source_rate_and_fov(tmp_path):
    import argparse
    import importlib

    dash = importlib.import_module("src.13_realtime_dashboard")
    d = _scene(2, 1.0, 10, tmp_path)
    (tmp_path / "in.yaml").write_text(
        f"type: folder\npath: {d.as_posix()}\nsensor:\n  hz: 20\n  fov_up: 15.0\n  fov_down: -25.0\n", encoding="utf-8")

    args = argparse.Namespace(source="rellis3d", input_config=str(tmp_path / "in.yaml"), input_dir=None,
                              sensor_hz=None, fov_up=None, fov_down=-10.0)
    dash.resolve_input(args)
    assert args.source == "custom" and args.sensor_hz == 20
    assert args.fov_up == 15.0 and args.fov_down == -10.0  # explicit flag wins over the config

    plain = argparse.Namespace(source="rellis3d", input_config=None, input_dir=None, sensor_hz=None)
    dash.resolve_input(plain)
    assert plain.lidar_source is None and plain.sensor_hz == 10.0 and plain.source == "rellis3d"

    both = argparse.Namespace(source="rellis3d", input_config="a", input_dir="b", sensor_hz=None)
    with pytest.raises(SystemExit):
        dash.resolve_input(both)
