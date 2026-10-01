"""The optimized 2D-map renderers must produce exactly the pixels the original
matplotlib-colormap implementation produced, and the corridor chart must draw
exactly the data it is given (no 'skip redraw if unchanged' shortcut)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PyQt5", reason="dashboard needs PyQt5")

import matplotlib as mpl  # noqa: E402

from src.dashboard.qt_main_window import grid_to_rgba8, overhead_to_rgba8  # noqa: E402
from src.perception.overhead_detection import (  # noqa: E402
    EMPTY, GROUND, INDETERMINATE, OVERHEAD_OBSTACLE, SOLID_OBSTACLE,
)


def _reference_grid(grid, map_type):
    """The original implementation, verbatim: matplotlib colormap on the float grid."""
    import cv2

    valid_mask = ~np.isnan(grid)
    if not np.any(valid_mask):
        return None
    g_min, g_max = np.nanmin(grid), np.nanmax(grid)
    norm_grid = (grid - g_min) / (g_max - g_min) if g_max > g_min else np.zeros_like(grid)
    norm_grid[~valid_mask] = 0.0
    name = {"elevation": "turbo", "traversability": "RdYlGn", "roi": "magma",
            "uncertainty": "cividis"}.get(map_type, "viridis")
    rgba = mpl.colormaps[name](norm_grid)
    rgba[~valid_mask, 3] = 0.0
    rgba_8 = (rgba * 255).astype(np.uint8)
    if map_type == "traversability":
        lethal_mask = (grid < 0.40) & valid_mask
        if np.any(lethal_mask):
            for i in range(0, rgba_8.shape[0] + rgba_8.shape[1], 5):
                cv2.line(rgba_8, (0, i), (i, 0), (0, 0, 0, 255), 1)
            rgba_8_no_hatch = (rgba * 255).astype(np.uint8)
            rgba_8[~lethal_mask] = rgba_8_no_hatch[~lethal_mask]
    return rgba_8


def _reference_overhead(trav_grid, overhead_grid):
    import cv2

    h, w = overhead_grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    ground_mask = overhead_grid == GROUND
    if trav_grid is not None:
        valid_trav = ~np.isnan(trav_grid)
        norm = np.zeros_like(trav_grid, dtype=np.float32)
        norm[valid_trav] = np.clip(trav_grid[valid_trav], 0.0, 1.0)
        ground_rgba = (mpl.colormaps["RdYlGn"](norm) * 255).astype(np.uint8)
        fillable = ground_mask & valid_trav
        rgba[fillable] = ground_rgba[fillable]
    rgba[overhead_grid == SOLID_OBSTACLE] = [251, 146, 60, 255]
    overhead_mask = overhead_grid == OVERHEAD_OBSTACLE
    rgba[overhead_mask] = [34, 211, 238, 255]
    rgba[overhead_grid == INDETERMINATE] = [100, 100, 100, 255]
    rgba[overhead_grid == EMPTY] = [0, 0, 0, 0]
    if np.any(overhead_mask):
        hatched = rgba.copy()
        for i in range(0, h + w, 4):
            cv2.line(hatched, (0, i), (i, 0), (10, 10, 10, 255), 1)
        rgba = np.where(overhead_mask[:, :, None], hatched, rgba)
    return rgba


def _random_grid(rng, nan_frac=0.3, lo=-2.0, hi=3.0):
    g = rng.uniform(lo, hi, size=(80, 80)).astype(np.float32)
    g[rng.random((80, 80)) < nan_frac] = np.nan
    return g


@pytest.mark.parametrize("map_type", ["elevation", "traversability", "roi", "uncertainty", "other"])
def test_grid_pixels_match_original(map_type):
    rng = np.random.default_rng(7)
    for _ in range(25):
        grid = _random_grid(rng, lo=0.0 if map_type == "traversability" else -2.0, hi=1.0 if map_type == "traversability" else 3.0)
        np.testing.assert_array_equal(grid_to_rgba8(grid.copy(), map_type), _reference_grid(grid.copy(), map_type))


def test_grid_edge_cases_match_original():
    flat = np.full((80, 80), 0.5, dtype=np.float32)               # no range -> all zeros norm
    np.testing.assert_array_equal(grid_to_rgba8(flat.copy(), "elevation"), _reference_grid(flat.copy(), "elevation"))
    allnan = np.full((80, 80), np.nan, dtype=np.float32)
    assert grid_to_rgba8(allnan, "elevation") is None and _reference_grid(allnan, "elevation") is None
    ramp = np.tile(np.linspace(0, 1, 80, dtype=np.float32), (80, 1))   # includes exactly 0.0 and 1.0
    for t in ("elevation", "traversability"):
        np.testing.assert_array_equal(grid_to_rgba8(ramp.copy(), t), _reference_grid(ramp.copy(), t))


def test_overhead_pixels_match_original():
    rng = np.random.default_rng(3)
    codes = np.array([EMPTY, GROUND, SOLID_OBSTACLE, OVERHEAD_OBSTACLE, INDETERMINATE], dtype=np.int8)
    for _ in range(25):
        over = codes[rng.integers(0, len(codes), size=(80, 80))]
        trav = _random_grid(rng, lo=0.0, hi=1.0)
        np.testing.assert_array_equal(overhead_to_rgba8(trav, over), _reference_overhead(trav, over))
    np.testing.assert_array_equal(overhead_to_rgba8(None, over), _reference_overhead(None, over))
    assert overhead_to_rgba8(trav, None) is None


def test_corridor_chart_draws_exactly_the_given_data():
    """The old chart skipped redrawing when a cheap key matched, so real changes
    (e.g. a different height in a middle bin) were not shown."""
    from PyQt5.QtWidgets import QApplication

    from src.dashboard.qt_main_window import CenterlineProfileChart

    app = QApplication.instance() or QApplication([])  # noqa: F841
    chart = CenterlineProfileChart()
    chart.resize(800, 200)

    def bins(h):
        return [{"distance_m": float(d), "height_above_baseline_m": h if d == 5 else 0.1,
                 "depth_below_baseline_m": None, "point_count": 20} for d in range(10)]

    chart.set_data(bins(0.4), [])
    y_top_first = chart._ylim[1]
    chart.set_data(bins(1.8), [])  # same bin count, same first/last distance, same first point_count
    assert chart._ylim[1] > y_top_first
    assert any(abs(v - 1.8) < 1e-9 for _, v, _, _ in chart._bars)
    chart.grab()  # paints without error
    chart.set_data([], [])
    chart.grab()


def test_ego_motion_label_says_how_often_it_is_recomputed():
    """The ICP estimate runs only every Nth frame, so the label must not claim
    it is the displacement 'since the previous frame' of the frame on screen."""
    from PyQt5.QtWidgets import QApplication, QLabel

    from src.dashboard.qt_main_window import FoveaXDashboardWindow

    app = QApplication.instance() or QApplication([])  # noqa: F841
    win = FoveaXDashboardWindow.__new__(FoveaXDashboardWindow)
    win.lbl_ego_motion = QLabel()
    win.set_ego_motion_estimate(0.488, 89, 30)
    text = win.lbl_ego_motion.text()
    assert "frame 90" in text and "every 30 frames" in text and "0.488" in text
    win.set_ego_motion_estimate(None, None, 30)
    assert "every 30 frames" in win.lbl_ego_motion.text() and "N/A" in win.lbl_ego_motion.text()
    win.set_ego_motion_estimate(0.1)  # legacy call without cadence
    assert "since previous frame" in win.lbl_ego_motion.text()
