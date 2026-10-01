"""Measure the REAL throughput of the live GUI dashboard.

    python tools/bench_gui_throughput.py [frames=100] [dashboard flags...]

Starts the real dashboard (PyQt5 window, embedded Open3D view, worker pool) with the
default launch profile (full local RELLIS-3D unless --dataset-profile says otherwise),
lets it run flat out (--rate-hz 1000 unless you pass another), stops after the given
number of frames and reports frames PAINTED per second, frames computed per second
and where each frame's time went. The packaged app does the same with
`FOVEAX.exe --bench-frames N`; both use src/dashboard/gui_bench.py.

Run it on AC power with the "Best performance" power mode: on battery the CPU is
power-limited and the numbers are much lower (see docs/validation_results.md).
"""
import importlib
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    args_in = sys.argv[1:]
    frames = int(args_in[0]) if args_in and args_in[0].isdigit() else 100
    passthrough = args_in[1:] if args_in and args_in[0].isdigit() else args_in
    if "--rate-hz" not in passthrough:
        passthrough = ["--rate-hz", "1000"] + passthrough

    os.chdir(os.path.join(os.environ.get("TEMP", "."), ""))  # keep outputs/ logs out of the repo
    try:
        import torch  # noqa: F401  (must be imported before PyQt5)
    except ImportError:
        pass

    mod = importlib.import_module("src.13_realtime_dashboard")
    sys.argv = ["bench", "--frames", str(frames + 60)] + passthrough
    args = mod.parse_args()
    mod.resolve_input(args)
    from src.dashboard.profiles import apply_profile
    apply_profile(args)
    app = mod.DashboardApplication(args)

    from src.dashboard.gui_bench import run_gui_bench
    run_gui_bench(app, frames)


if __name__ == "__main__":
    main()
