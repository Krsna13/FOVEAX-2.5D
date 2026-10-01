"""Golden-output regression test for the perception pipeline.

Runs the real process_frame pipeline on fixed real RELLIS-3D frames and compares
the grids (elevation, traversability, ROI, uncertainty, overhead), hazard clusters,
tracks (state, covariance, metadata), classes, elevation/corridor profiles, road
width, uncertainty and live accuracy with a stored reference.

The reference files in tests/golden/ were generated from the code at commit 571c976,
BEFORE the "OPT-n" shortcuts (grid built from a thinned cloud, terrain features
skipped for far static objects, hazard/corridor profile recomputed only every 2nd or
3rd frame, reused grid buffers). Those shortcuts changed the output on every frame,
so this test fails if any of them, or anything like them, comes back
(tools/make_golden.py explains how to regenerate on purpose).

Three layers:
  * fixture frames (8 real frames, 20k real points each, committed): always runs,
    sequentially and through the worker pool + tracking thread;
  * full frames 0-9 of RELLIS-3D sequence 00001: runs wherever the dataset or the
    bundled release sample is present (this is also what the packaged app checks
    with `FOVEAX.exe --golden-check`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

pytest.importorskip("PyQt5", reason="dashboard needs PyQt5")

import golden_utils as gu  # noqa: E402
from src.dashboard.data_streamer import DataStreamerThread  # noqa: E402
from src.dashboard.frame_pipeline import FrameFinisher, FramePipeline, config_from_streamer  # noqa: E402
from src.inputs import ArraySource  # noqa: E402
from src.perception.object_detector import MockObjectDetector  # noqa: E402

GOLDEN = HERE / "golden"


@pytest.fixture(scope="module")
def fixture_source():
    with np.load(GOLDEN / "fixture_inputs.npz") as z:
        return ArraySource(z["points"], z["labels"], sensor_hz=10.0, name="golden-fixture")


@pytest.fixture(scope="module")
def fixture_golden():
    return gu.load_golden(GOLDEN / "fixture_golden.npz")


def _streamer(source):
    s = DataStreamerThread(source="rellis3d", detector=MockObjectDetector(), num_frames=len(source),
                           rate_hz=10.0, lidar_source=source, predictor_mode="ground_truth")
    s._setup_sources()
    s.display_voxel_m = 0.0
    return s


def _compare(frames, golden, label):
    assert len(frames) == len(golden)
    for i, (got, want) in enumerate(zip(frames, golden)):
        gu.assert_same(got, want, f"{label} frame {i}")


def test_golden_files_exist_and_are_not_trivial(fixture_golden):
    assert len(fixture_golden) == 8
    f = fixture_golden[3]
    assert f["tracks"] and f["hazard_clusters"], "the reference must contain tracks and hazards"
    assert {"elevation", "traversability", "roi", "uncertainty", "overhead"} <= set(f["grid_maps"])
    assert f["accuracy_metrics"] is not None


def test_sequential_pipeline_matches_golden(fixture_source, fixture_golden):
    s = _streamer(fixture_source)
    frames = [gu.plain(s.process_frame(i, s.timestamp_for(i), fps=10.0)) for i in range(len(fixture_source))]
    _compare(frames, fixture_golden, "sequential")


def test_worker_pool_and_tracking_thread_match_golden(fixture_source, fixture_golden):
    s = _streamer(fixture_source)
    pipe = FramePipeline(config_from_streamer(s), workers=2)
    fin = FrameFinisher(pipe, s, 0, s.timestamp_for, lambda: 10.0)
    try:
        frames = [gu.plain(fin.get(i)[0]) for i in range(len(fixture_source))]
    finally:
        fin.stop()
        pipe.shutdown()
    _compare(frames, fixture_golden, "pooled")


def _find_full_data_root():
    for cand in (os.environ.get("FOVEAX_RELLIS_ROOT"), REPO / "data" / "rellis3d", "C:/dev/data/rellis3d"):
        if cand and (Path(cand) / "Rellis-3D" / "00001" / "os1_cloud_node_kitti_bin" / "000009.bin").exists():
            return Path(cand)
    return None


def test_full_frames_match_golden():
    root = _find_full_data_root()
    if root is None:
        pytest.skip("RELLIS-3D sequence 00001 (or the bundled sample) not found")
    golden = gu.load_golden(GOLDEN / "full_00001_golden.npz")
    s = DataStreamerThread(source="rellis3d", detector=MockObjectDetector(), num_frames=len(golden) + 5,
                           sequence="00001", data_root=root, rate_hz=10.0, predictor_mode="ground_truth")
    s._setup_sources()
    s.display_voxel_m = 0.0
    frames = [gu.plain(s.process_frame(i, s.timestamp_for(i), fps=10.0)) for i in range(len(golden))]
    _compare(frames, golden, "full-frame")
