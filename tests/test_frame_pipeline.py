"""The worker-pool path must give exactly the sequential result, frame for frame."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("PyQt5", reason="dashboard needs PyQt5")

from src.dashboard.data_streamer import DataStreamerThread  # noqa: E402
from src.dashboard.frame_pipeline import (  # noqa: E402
    FrameFinisher, FramePipeline, _NO_EGO, config_from_streamer, config_signature, default_worker_count,
)
from src.inputs import create_source  # noqa: E402
from src.perception.object_detector import MockObjectDetector  # noqa: E402
from test_input_streamer_integration import _scene  # noqa: E402

N_FRAMES = 10


def _plain(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _assert_same(a, b, path="state"):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, path
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (p, q) in enumerate(zip(a, b)):
            _assert_same(p, q, f"{path}[{i}]")
    elif isinstance(a, float) and a != a:
        assert b != b, path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _new_streamer(src):
    s = DataStreamerThread(source="rellis3d", detector=MockObjectDetector(), num_frames=N_FRAMES,
                           lidar_source=src, rate_hz=30.0)
    s._setup_sources()
    return s


@pytest.fixture(scope="module")
def scene_source(tmp_path_factory):
    d = _scene(N_FRAMES, 2.0, 10, tmp_path_factory.mktemp("pipe"))
    # a few labelled classes so accuracy/class logic is exercised too
    labels = tmp_path_factory.mktemp("pipe_labels")
    for i in range(N_FRAMES):
        n = len(np.fromfile(d / f"{i:04d}.bin", dtype=np.float32)) // 4
        ids = np.where(np.arange(n) % 5 == 0, 40, 50).astype(np.uint32)  # road / building
        ids.tofile(labels / f"{i:04d}.label")
    return create_source({"type": "folder", "path": str(d), "sensor": {"hz": 10},
                          "labels": {"dir": str(labels), "map": "semantickitti"}})


def test_pooled_frames_equal_sequential_frames(scene_source):
    seq = _new_streamer(scene_source)
    seq.display_voxel_m = 0.0
    seq_states = [_plain(seq.process_frame(i, seq.timestamp_for(i), fps=10.0)) for i in range(N_FRAMES)]

    pooled = _new_streamer(scene_source)
    pooled.display_voxel_m = 0.0
    pipe = FramePipeline(config_from_streamer(pooled), workers=2)
    try:
        pool_states = []
        for i in range(N_FRAMES):
            pipe.prefetch(i, pooled.timestamp_for)
            res = pipe.get(i, pooled.timestamp_for)
            pool_states.append(_plain(pooled.finish_frame(res, pooled.timestamp_for(i), fps=10.0)))
    finally:
        pipe.shutdown()

    for i, (a, b) in enumerate(zip(seq_states, pool_states)):
        for d in (a, b):
            d.pop("metrics")  # hardware/latency measurements, not perception
        assert a["accuracy_metrics"] is not None, "ground truth was supplied, accuracy must be computed"
        _assert_same(a, b, f"frame{i}")


def test_worker_ego_motion_matches_main_thread(scene_source):
    from src.perception.ego_motion_estimate import estimate_relative_displacement_m

    s = _new_streamer(scene_source)
    pipe = FramePipeline(config_from_streamer(s), workers=1, ego_every_n=3)
    try:
        res = pipe.get(3, s.timestamp_for)
    finally:
        pipe.shutdown()
    prev = s._get_frame_data(2)[0]
    cur = s._get_frame_data(3)[0]
    # Open3D's multi-threaded ICP sums in a varying order, so repeated runs differ at
    # the ~1e-16 level even in one process; that is not a worker-vs-main difference.
    assert res["ego_motion_m"] == pytest.approx(
        estimate_relative_displacement_m(prev[:, :3], cur[:, :3]), abs=1e-9)


def test_display_cloud_is_a_thinned_copy_and_never_used_for_perception(scene_source):
    s = _new_streamer(scene_source)
    s.display_voxel_m = 0.5
    res = s.compute_stateless(0, 0.0)
    assert res["display_points"] is not None
    assert 0 < len(res["display_points"]) < len(res["points"])
    off = _new_streamer(scene_source)
    off.display_voxel_m = 0.0
    res_off = off.compute_stateless(0, 0.0)
    assert res_off["display_points"] is None
    # perception inputs/outputs do not depend on the display setting
    np.testing.assert_array_equal(res["points"], res_off["points"])
    assert len(res["detections"]) == len(res_off["detections"])
    np.testing.assert_array_equal(res["grid_base"]["elevation"], res_off["grid_base"]["elevation"])


def test_point_bounds_precomputed_in_worker_give_identical_warnings(scene_source):
    s = _new_streamer(scene_source)
    res = s.compute_stateless(0, 0.0)
    st = s.finish_frame(res, 0.0)
    from src.dashboard.dashboard_state import assert_coordinate_frame_consistency

    assert assert_coordinate_frame_consistency(st) == assert_coordinate_frame_consistency(st, res["point_bounds"])


def test_config_signature_changes_with_settings_and_helpers(scene_source):
    s = _new_streamer(scene_source)
    base = config_signature(config_from_streamer(s))
    assert config_signature(config_from_streamer(s)) == base
    s.ground_noise_m = 0.5
    assert config_signature(config_from_streamer(s)) != base
    s.ground_noise_m = 0.30
    s.detector.voxel_size_m = 0.3
    assert config_signature(config_from_streamer(s)) != base
    assert 1 <= default_worker_count() <= 6


def test_background_finisher_matches_sequential(scene_source):
    """Tracking on the finisher thread must give exactly the sequential result."""
    seq = _new_streamer(scene_source)
    seq.display_voxel_m = 0.0
    seq_states = [_plain(seq.process_frame(i, seq.timestamp_for(i), fps=10.0)) for i in range(N_FRAMES)]

    s = _new_streamer(scene_source)
    s.display_voxel_m = 0.0
    pipe = FramePipeline(config_from_streamer(s), workers=2, ego_every_n=4)
    fin = FrameFinisher(pipe, s, 0, s.timestamp_for, lambda: 10.0)
    try:
        got = []
        for i in range(N_FRAMES):
            state, ego = fin.get(i)
            got.append(_plain(state))
            # frames 4 and 8 carry an ego estimate, the others must not
            assert (ego is not _NO_EGO) == (i in (4, 8)), i
    finally:
        fin.stop()
        pipe.shutdown()
    for i, (a, b) in enumerate(zip(seq_states, got)):
        for d in (a, b):
            d.pop("metrics")
        _assert_same(a, b, f"frame{i}")


def test_finisher_reports_errors_instead_of_hanging(scene_source):
    s = _new_streamer(scene_source)
    pipe = FramePipeline(config_from_streamer(s), workers=1)

    def boom(*a, **k):
        raise ValueError("finish failed")

    s.finish_frame = boom
    fin = FrameFinisher(pipe, s, 0, s.timestamp_for, lambda: 10.0)
    try:
        with pytest.raises(ValueError, match="finish failed"):
            fin.get(0)
    finally:
        fin.stop()
        pipe.shutdown()


def test_a_finished_frame_keeps_its_own_tracks(scene_source):
    """The tracker updates its TrackState objects in place on later frames; a
    FrameState (kept for STEP back, or read from another thread) must not
    change when tracking moves on."""
    s = _new_streamer(scene_source)
    s.process_frame(0, s.timestamp_for(0))
    st1 = s.process_frame(1, s.timestamp_for(1))
    before = [(t.track_id, t.state.copy(), t.covariance.copy(), dict(t.metadata)) for t in st1.tracks]
    assert before, "the scene has a moving obstacle that must be tracked"
    for i in range(2, 6):
        s.process_frame(i, s.timestamp_for(i))
    for t, (tid, state, cov, meta) in zip(st1.tracks, before):
        assert t.track_id == tid
        np.testing.assert_array_equal(t.state, state)
        np.testing.assert_array_equal(t.covariance, cov)
        assert t.metadata == meta
