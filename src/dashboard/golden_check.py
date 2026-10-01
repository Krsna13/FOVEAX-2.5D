"""`--golden-check`: prove this build produces the reference perception output.

Runs the real pipeline on the first frames of RELLIS-3D sequence 00001 (from whatever
data the active profile points at: the full dataset or the bundled sample) and
compares every grid, hazard, track and class with tests/golden/full_00001_golden.npz,
which was generated from the code before the "OPT-n" shortcuts. Works from source
and from the packaged exe (build_app.py ships golden_utils.py + the .npz in golden/).
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.dashboard.profiles import REPO_ROOT, app_base_dir


def _golden_dirs():
    base = app_base_dir()
    for d in (base / "golden", REPO_ROOT / "tests" / "golden"):
        if (d / "full_00001_golden.npz").exists():
            utils = d if (d / "golden_utils.py").exists() else REPO_ROOT / "tests"
            return d, utils
    return None, None


def run_golden_check(args) -> int:
    gdir, udir = _golden_dirs()
    if gdir is None:
        print("GOLDEN CHECK: reference files not found (expected golden/ next to the app).")
        return 2
    sys.path.insert(0, str(udir))
    import golden_utils as gu

    from src.dashboard.data_streamer import DataStreamerThread
    from src.perception.object_detector import MockObjectDetector

    golden = gu.load_golden(gdir / "full_00001_golden.npz")
    s = DataStreamerThread(source="rellis3d", detector=MockObjectDetector(), num_frames=len(golden) + 5,
                           sequence="00001", data_root=args.data_root, rate_hz=10.0, predictor_mode="ground_truth")
    s._setup_sources()
    s.display_voxel_m = 0.0
    avail = s.available_frames() or 0
    if avail < len(golden):
        print(f"GOLDEN CHECK: only {avail} frames available, need {len(golden)}.")
        return 2
    for i in range(len(golden)):
        got = gu.plain(s.process_frame(i, s.timestamp_for(i), fps=10.0))
        try:
            gu.assert_same(got, golden[i], f"frame {i}")
        except AssertionError as exc:
            print(f"GOLDEN CHECK: FAIL  {exc}")
            return 1
    frozen = "packaged exe" if getattr(sys, "frozen", False) else "source"
    print(f"GOLDEN CHECK: PASS  ({len(golden)} real frames, data {args.data_root}, running as {frozen}; "
          "grids, hazards, tracks, classes identical to the pre-shortcut reference)")
    return 0
