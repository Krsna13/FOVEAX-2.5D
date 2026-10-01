"""Launch profiles: one code path, only data location and defaults differ."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard import profiles  # noqa: E402


def make_data(root: Path, seq: str, n: int) -> Path:
    d = root / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i:06d}.bin").write_bytes(b"")
    return root


def args(**kw):
    base = dict(dataset_profile=None, source="rellis3d", sequence=None, frames=None, rate_hz=None,
                data_root=None, lidar_source=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_both_profiles_share_playback_defaults():
    full, rel = profiles.PROFILES["full"], profiles.PROFILES["release"]
    for key in ("sequence", "frames", "rate_hz"):
        assert full[key] == rel[key]
    assert full["sequence"] == "00001" and full["frames"] == 0 and full["rate_hz"] == 25.0


def test_full_profile_plays_every_frame(tmp_path):
    root = make_data(tmp_path, "00001", 37)
    a = args(dataset_profile="full", data_root=root)
    profiles.apply_profile(a)
    assert (a.frames, a.sequence, a.rate_hz, a.data_root) == (37, "00001", 25.0, root)
    assert "37" in a.dataset_banner and "sample" not in a.dataset_banner.lower()


def test_release_profile_banner_names_the_sample(tmp_path):
    root = make_data(tmp_path, "00001", 100)
    a = args(dataset_profile="release", data_root=root)
    profiles.apply_profile(a)
    assert a.dataset_banner == "Sample data: 100 frames (full dataset not included)"


def test_explicit_flags_override_the_profile(tmp_path):
    root = make_data(tmp_path, "00001", 50)
    a = args(dataset_profile="full", data_root=root, frames=10, rate_hz=1000.0)
    profiles.apply_profile(a)
    assert a.frames == 10 and a.rate_hz == 1000.0
    assert "10 frames" in a.dataset_banner


def test_more_frames_requested_than_exist_is_reported_honestly(tmp_path):
    root = make_data(tmp_path, "00001", 20)
    a = args(dataset_profile="release", data_root=root, frames=3000)
    profiles.apply_profile(a)
    assert a.dataset_banner == "Sample data: 20 frames (full dataset not included)"


def test_missing_data_is_an_error_not_a_silent_fallback(tmp_path):
    with pytest.raises(SystemExit, match="No RELLIS-3D frames"):
        profiles.apply_profile(args(dataset_profile="full", data_root=tmp_path / "nope"))


def test_env_variable_sets_the_full_profile_root(tmp_path, monkeypatch):
    root = make_data(tmp_path, "00001", 5)
    monkeypatch.setenv("FOVEAX_RELLIS_ROOT", str(root))
    a = args(dataset_profile="full")
    profiles.apply_profile(a)
    assert a.data_root == root and a.frames == 5


def test_default_profile_follows_frozen_state(monkeypatch):
    assert profiles.default_profile_name() == "full"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert profiles.default_profile_name() == "release"


def test_unknown_profile_rejected():
    with pytest.raises(SystemExit):
        profiles.apply_profile(args(dataset_profile="bogus"))
