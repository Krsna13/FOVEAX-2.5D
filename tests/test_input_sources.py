"""Tests for the pluggable LiDAR input layer (src/inputs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inputs import (  # noqa: E402
    FolderSource,
    LidarSource,
    create_source,
    load_source_config,
    register_source_type,
    source_from_dir,
)
from src.inputs.formats import map_labels, parse_axes, to_pipeline_points  # noqa: E402


def _cloud(n=50, seed=0, cols=4):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-10, 10, size=(n, cols)).astype(np.float32)
    a[:, 3] = rng.uniform(0, 1, n)
    return a


def _write_bin_frames(d: Path, n_frames=3, n=50, cols=4):
    d.mkdir(parents=True, exist_ok=True)
    clouds = []
    for i in range(n_frames):
        c = _cloud(n, seed=i, cols=cols)
        c.tofile(d / f"{i}.bin")
        clouds.append(c)
    return clouds


class TestFormats:
    def test_natural_sort_orders_2_before_10(self, tmp_path):
        for name in ("10", "2", "1"):
            _cloud().tofile(tmp_path / f"{name}.bin")
        src = source_from_dir(tmp_path)
        assert [p.stem for p in src.files] == ["1", "2", "10"]

    def test_bin_with_five_columns(self, tmp_path):
        clouds = _write_bin_frames(tmp_path, 1, cols=5)
        src = create_source({"type": "folder", "path": str(tmp_path), "n_cols": 5})
        np.testing.assert_allclose(src.frame(0).points, clouds[0][:, :4])

    def test_bin_wrong_column_count_is_a_clear_error(self, tmp_path):
        np.zeros(10, dtype=np.float32).tofile(tmp_path / "a.bin")
        src = create_source({"type": "folder", "path": str(tmp_path), "n_cols": 4})
        with pytest.raises(ValueError, match="n_cols"):
            src.frame(0)

    def test_npy_npz_csv_txt(self, tmp_path):
        c = _cloud()
        d = tmp_path / "npy"; d.mkdir(); np.save(d / "a.npy", c)
        np.testing.assert_allclose(source_from_dir(d).frame(0).points, c)
        d = tmp_path / "npz"; d.mkdir(); np.savez(d / "a.npz", points=c)
        np.testing.assert_allclose(source_from_dir(d).frame(0).points, c)
        d = tmp_path / "csv"; d.mkdir()
        np.savetxt(d / "a.csv", c, delimiter=",", header="x,y,z,i", comments="")
        np.testing.assert_allclose(source_from_dir(d).frame(0).points, c, atol=1e-4)
        d = tmp_path / "txt"; d.mkdir()
        np.savetxt(d / "a.txt", c)
        np.testing.assert_allclose(source_from_dir(d).frame(0).points, c, atol=1e-4)

    def test_ply_round_trip_with_intensity(self, tmp_path):
        o3d = pytest.importorskip("open3d")
        c = _cloud(20)
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = o3d.core.Tensor(c[:, :3])
        pcd.point.intensity = o3d.core.Tensor(c[:, 3:4])
        o3d.t.io.write_point_cloud(str(tmp_path / "a.ply"), pcd)
        got = source_from_dir(tmp_path).frame(0).points
        np.testing.assert_allclose(got[:, :3], c[:, :3], atol=1e-5)
        np.testing.assert_allclose(got[:, 3], c[:, 3], atol=1e-5)

    def test_unknown_extension_asks_for_format(self, tmp_path):
        (tmp_path / "a.dat").write_bytes(b"0" * 16)
        with pytest.raises(FileNotFoundError):
            source_from_dir(tmp_path)  # nothing auto-detectable
        src = create_source({"type": "folder", "path": str(tmp_path), "pattern": "*.dat", "format": "bin"})
        assert len(src) == 1


class TestGeometry:
    def test_axes_flip_and_swap(self):
        raw = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
        pts, _ = to_pipeline_points(raw, axes="-y,x,z")
        np.testing.assert_allclose(pts[0, :3], [-2.0, 1.0, 3.0])

    def test_axes_must_be_a_permutation(self):
        with pytest.raises(ValueError):
            parse_axes("x,x,z")
        with pytest.raises(ValueError):
            parse_axes("x,y")

    def test_z_offset_and_intensity_scale(self):
        raw = np.array([[0.0, 0.0, -1.7, 200.0]], dtype=np.float32)
        pts, _ = to_pipeline_points(raw, z_offset=1.7, intensity_scale=1 / 255)
        assert pts[0, 2] == pytest.approx(0.0, abs=1e-6)
        assert pts[0, 3] == pytest.approx(200 / 255, rel=1e-5)

    def test_custom_columns_and_missing_intensity(self):
        raw = np.array([[9, 1.0, 2.0, 3.0]], dtype=np.float32)
        pts, _ = to_pipeline_points(raw, columns=(1, 2, 3, None))
        np.testing.assert_allclose(pts[0], [1.0, 2.0, 3.0, 0.0])

    def test_column_out_of_range_is_reported(self):
        with pytest.raises(ValueError, match="column index"):
            to_pipeline_points(np.zeros((2, 3), dtype=np.float32), columns=(0, 1, 2, 3))

    def test_filters_nonfinite_zero_returns_and_range(self):
        raw = np.array([[1, 1, 1, 1], [np.nan, 0, 0, 1], [0, 0, 0, 1], [50, 0, 0, 1], [0.1, 0, 0, 1]],
                       dtype=np.float32)
        pts, keep = to_pipeline_points(raw)  # default: only non-finite dropped
        assert keep.tolist() == [True, False, True, True, True]
        pts, keep = to_pipeline_points(raw, drop_zero_returns=True, min_range_m=0.5, max_range_m=20)
        assert keep.tolist() == [True, False, False, False, False]
        assert pts.shape == (1, 4)


class TestLabels:
    def _src(self, tmp_path, ids, mapping="semantickitti", n=4):
        scans = tmp_path / "scans"; scans.mkdir()
        labels = tmp_path / "labels"; labels.mkdir()
        _cloud(n).tofile(scans / "0.bin")
        (np.array(ids, dtype=np.uint32) | (7 << 16)).astype(np.uint32).tofile(labels / "0.label")  # instance bits set
        return create_source({"type": "folder", "path": str(scans),
                              "labels": {"dir": "../labels", "map": mapping}})

    def test_semantickitti_labels_ignore_instance_bits_and_map(self, tmp_path):
        from src.perception.semantic_labels import SEMANTICKITTI_TO_FOVEAX

        f = self._src(tmp_path, [40, 10, 50, 999]).frame(0)
        assert f.foveax_labels.dtype == np.uint8
        assert f.foveax_labels[0] == SEMANTICKITTI_TO_FOVEAX[40]
        assert f.foveax_labels[1] == SEMANTICKITTI_TO_FOVEAX[10]
        assert f.foveax_labels[3] == 7  # unmapped -> UNKNOWN

    def test_custom_dict_map(self, tmp_path):
        f = self._src(tmp_path, [1, 2, 3, 1], mapping={1: 0, 2: 5}).frame(0)
        assert f.foveax_labels.tolist() == [0, 5, 7, 0]

    def test_labels_follow_point_filtering(self, tmp_path):
        scans = tmp_path / "s"; scans.mkdir()
        labs = tmp_path / "l"; labs.mkdir()
        raw = np.array([[1, 1, 1, 1], [np.nan, 0, 0, 1], [2, 2, 2, 1]], dtype=np.float32)
        raw.tofile(scans / "0.bin")
        np.array([10, 40, 50], dtype=np.uint32).tofile(labs / "0.label")
        f = create_source({"type": "folder", "path": str(scans), "labels": {"dir": "../l"}}).frame(0)
        assert f.points.shape[0] == 2 and f.foveax_labels.shape[0] == 2

    def test_label_count_mismatch_is_an_error(self, tmp_path):
        s = self._src(tmp_path, [10, 10, 10], n=4)
        with pytest.raises(ValueError, match="labels"):
            s.frame(0)

    def test_missing_label_file_means_no_labels(self, tmp_path):
        scans = tmp_path / "scans"; scans.mkdir()
        (tmp_path / "labels").mkdir()
        _cloud().tofile(scans / "0.bin")
        f = create_source({"type": "folder", "path": str(scans), "labels": {"dir": "../labels"}}).frame(0)
        assert f.foveax_labels is None

    def test_map_labels_handles_negative_and_empty(self):
        out = map_labels(np.array([-1, 0, 2]), {0: 3, 2: 4})
        assert out.tolist() == [7, 3, 4]
        assert map_labels(np.array([], dtype=np.int64), {}).size == 0


class TestTimestampsAndRate:
    def test_default_timestamps_use_sensor_hz(self, tmp_path):
        _write_bin_frames(tmp_path, 4)
        src = create_source({"type": "folder", "path": str(tmp_path), "sensor": {"hz": 20}})
        assert src.sensor_hz == 20
        assert [src.timestamp(i) for i in range(4)] == pytest.approx([0, 0.05, 0.1, 0.15])
        assert src.frame(2).timestamp_s == pytest.approx(0.1)

    def test_default_rate_is_ten_hz(self, tmp_path):
        _write_bin_frames(tmp_path, 2)
        assert source_from_dir(tmp_path).timestamp(1) == pytest.approx(0.1)

    def test_timestamps_from_filenames_in_nanoseconds(self, tmp_path):
        for t in (1_000_000_000, 1_100_000_000, 1_250_000_000):
            _cloud().tofile(tmp_path / f"{t}.bin")
        src = create_source({"type": "folder", "path": str(tmp_path),
                             "timestamps": {"mode": "filename", "unit": "ns"}})
        assert [src.timestamp(i) for i in range(3)] == pytest.approx([0.0, 0.1, 0.25])

    def test_timestamps_from_file_and_validation(self, tmp_path):
        _write_bin_frames(tmp_path / "s", 3)
        (tmp_path / "t.txt").write_text("10.0\n10.2\n10.5\n")
        src = create_source({"type": "folder", "path": "s",
                             "timestamps": {"mode": "file", "file": "t.txt"}}, base_dir=tmp_path)
        assert src.timestamp(2) == pytest.approx(0.5)
        (tmp_path / "bad.txt").write_text("1\n1\n2\n")
        with pytest.raises(ValueError, match="increasing"):
            create_source({"type": "folder", "path": "s",
                           "timestamps": {"mode": "file", "file": "bad.txt"}}, base_dir=tmp_path)

    def test_nonpositive_rate_rejected(self, tmp_path):
        _write_bin_frames(tmp_path, 1)
        with pytest.raises(ValueError):
            create_source({"type": "folder", "path": str(tmp_path), "sensor": {"hz": 0}})


class TestConfigAndRegistry:
    def test_yaml_config_with_relative_paths(self, tmp_path):
        _write_bin_frames(tmp_path / "scans", 2)
        (tmp_path / "in.yaml").write_text(
            "type: folder\npath: scans\nsensor:\n  name: Test\n  hz: 5\nz_offset: 1.0\n", encoding="utf-8")
        src = load_source_config(tmp_path / "in.yaml")
        assert len(src) == 2 and src.sensor_hz == 5 and src.sensor["name"] == "Test"
        assert src.frame(0).points[:, 2].min() > -10 + 1.0 - 1e-3

    def test_json_config(self, tmp_path):
        _write_bin_frames(tmp_path / "scans", 1)
        (tmp_path / "in.json").write_text(json.dumps({"type": "folder", "path": "scans"}), encoding="utf-8")
        assert len(load_source_config(tmp_path / "in.json")) == 1

    def test_typo_in_config_is_caught(self, tmp_path):
        _write_bin_frames(tmp_path, 1)
        with pytest.raises(ValueError, match="z_ofset"):
            create_source({"type": "folder", "path": str(tmp_path), "z_ofset": 1.0})

    def test_unknown_type_and_missing_folder(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source({"type": "nope"})
        with pytest.raises(FileNotFoundError):
            create_source({"type": "folder", "path": str(tmp_path / "missing")})
        with pytest.raises(ValueError, match="path"):
            create_source({"type": "folder"})

    def test_index_out_of_range(self, tmp_path):
        _write_bin_frames(tmp_path, 2)
        src = source_from_dir(tmp_path)
        with pytest.raises(IndexError):
            src.frame(2)
        with pytest.raises(IndexError):
            src.frame(-1)

    def test_register_custom_source_type(self):
        class Tiny(LidarSource):
            name = "tiny"
            sensor_hz = 1.0

            def __init__(self, cfg, base_dir=None):
                pass

            def __len__(self):
                return 1

            def frame(self, idx):
                raise NotImplementedError

        register_source_type("tiny", Tiny)
        assert isinstance(create_source({"type": "tiny"}), Tiny)


class TestBuiltinDatasets:
    def test_semantickitti_layout(self, tmp_path):
        from src.perception.semantic_labels import SEMANTICKITTI_TO_FOVEAX

        seq = tmp_path / "00"
        (seq / "velodyne").mkdir(parents=True); (seq / "labels").mkdir()
        _cloud(6).tofile(seq / "velodyne" / "000000.bin")
        np.array([40, 10, 50, 70, 0, 1], dtype=np.uint32).tofile(seq / "labels" / "000000.label")
        src = create_source({"type": "semantickitti", "root": str(tmp_path), "sequence": "00"})
        f = src.frame(0)
        assert f.points.shape == (6, 4) and f.foveax_labels[0] == SEMANTICKITTI_TO_FOVEAX[40]
        assert "HDL" in src.sensor["name"]

    def test_rellis_layout_and_sensor_metadata(self, tmp_path):
        seq = tmp_path / "Rellis-3D" / "00001"
        bins = seq / "os1_cloud_node_kitti_bin"; labs = seq / "os1_cloud_node_semantickitti_label_id"
        bins.mkdir(parents=True); labs.mkdir()
        _cloud(8).tofile(bins / "000000.bin")
        np.array([3, 3, 1, 4, 23, 0, 3, 3], dtype=np.uint32).tofile(labs / "000000.label")
        src = create_source({"type": "rellis3d", "root": str(tmp_path), "sequence": "00001"})
        f = src.frame(0)
        assert f.points.shape[1] == 4 and f.foveax_labels.shape[0] == f.points.shape[0]
        assert src.sensor["fov_up"] == 17.02 and src.sensor_hz == 10.0

    def test_rellis_scan_without_labels_still_plays(self, tmp_path):
        bins = tmp_path / "Rellis-3D" / "00001" / "os1_cloud_node_kitti_bin"
        bins.mkdir(parents=True)
        _cloud(5).tofile(bins / "000000.bin")
        f = create_source({"type": "rellis3d", "root": str(tmp_path)}).frame(0)
        assert f.foveax_labels is None and f.points.shape == (5, 4)

    def test_empty_dataset_folders_fail_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_source({"type": "rellis3d", "root": str(tmp_path)})
        with pytest.raises(FileNotFoundError):
            create_source({"type": "semantickitti", "root": str(tmp_path)})
