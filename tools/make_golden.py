"""Generate the golden reference files used by tests/test_golden_output.py.

    python tools/make_golden.py [--root CODE_TREE] [--data-root DIR] [--out-dir tests/golden]

IMPORTANT: the reference must come from code known to be exact, not from
whatever is being tested. The committed files were generated from the tree at
commit 571c976 (before the "OPT-n" shortcuts), by extracting it with
`git archive 571c976 src tests` and passing that folder as --root. Regenerate
only on purpose, when perception output is meant to change, and say why in the
commit message.

Outputs (in --out-dir):
  fixture_inputs.npz        8 real RELLIS-3D frames, each thinned to 20,000 real
                            points (seeded, original order kept): small enough
                            to commit, so the test always runs
  fixture_golden.npz        perception output for those 8 frames
  full_00001_golden.npz     perception output for frames 0-9 of the full RELLIS-3D
                            sequence 00001 (used where the dataset or the
                            bundled sample is present)
"""
import argparse
import importlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE, help="code tree to run (src/ and tests/ inside)")
    ap.add_argument("--data-root", type=Path, default=Path("C:/dev/data/rellis3d"))
    ap.add_argument("--out-dir", type=Path, default=HERE / "tests" / "golden")
    ap.add_argument("--full-frames", type=int, default=10)
    ap.add_argument("--fixture-frames", type=int, default=8)
    ap.add_argument("--fixture-points", type=int, default=20000)
    ap.add_argument("--reuse-inputs", action="store_true", help="keep an existing fixture_inputs.npz")
    a = ap.parse_args()

    sys.path.insert(0, str(a.root))
    sys.path.insert(0, str(HERE / "tests"))  # golden_utils lives with the tests
    try:
        import torch  # noqa: F401  (must precede PyQt5)
    except ImportError:
        pass
    import numpy as np
    import golden_utils as gu

    from src.dashboard.data_streamer import DataStreamerThread
    from src.perception.object_detector import MockObjectDetector
    from src.perception.rellis3d_loader import load_rellis3d_frame

    a.out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = a.out_dir / "fixture_inputs.npz"

    # ---- fixture inputs: real frames, thinned deterministically
    if not (a.reuse_inputs and inputs_path.exists()):
        rng = np.random.default_rng(0)
        pts_all, lab_all = [], []
        for i in range(a.fixture_frames):
            f = load_rellis3d_frame(a.data_root, "00001", f"{i:06d}")
            keep = np.sort(rng.choice(len(f.points), size=min(a.fixture_points, len(f.points)), replace=False))
            pts_all.append(f.points[keep].astype(np.float32))
            lab_all.append(f.foveax_class_ids[keep].astype(np.uint8))
        np.savez_compressed(inputs_path, points=np.stack(pts_all), labels=np.stack(lab_all))
        print("wrote", inputs_path, inputs_path.stat().st_size // 1024, "KB")
    with np.load(inputs_path) as z:
        fx_pts, fx_lab = z["points"], z["labels"]

    def run(streamer, n):
        frames = []
        for i in range(n):
            frames.append(gu.plain(streamer.process_frame(i, i / 10.0, fps=10.0)))
        return frames

    # ---- fixture golden
    s = DataStreamerThread(source="rellis3d", detector=MockObjectDetector(), num_frames=len(fx_pts),
                           sequence="00001", rate_hz=10.0, predictor_mode="ground_truth")
    s._setup_sources()
    s._get_frame_data = lambda i: (fx_pts[i], fx_lab[i])  # works for any code tree
    gu.save_golden(a.out_dir / "fixture_golden.npz", run(s, len(fx_pts)))
    print("wrote fixture_golden.npz from", a.root)

    # ---- full-frame golden
    s2 = DataStreamerThread(source="rellis3d", detector=MockObjectDetector(), num_frames=a.full_frames + 5,
                            sequence="00001", data_root=a.data_root, rate_hz=10.0, predictor_mode="ground_truth")
    s2._setup_sources()
    gu.save_golden(a.out_dir / "full_00001_golden.npz", run(s2, a.full_frames))
    print("wrote full_00001_golden.npz from", a.root)


if __name__ == "__main__":
    main()
