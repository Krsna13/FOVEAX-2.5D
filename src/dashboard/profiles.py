"""Launch profiles: ONE code path, two sets of defaults.

Both ways of starting the GUI run exactly the same src/ code:

  * "full"    - Launch_GUI.bat on a machine with the datasets: full local RELLIS-3D,
                every frame of the sequence.
  * "release" - the packaged FOVEAX.exe (GitHub Release): the small sample frames that
                are bundled next to the exe. Same playback defaults, and the window
                says plainly that this is sample data.

Only data location, sequence, frame count and playback rate differ, and they are all
defined here (nothing is hard-coded in the launchers). A profile only fills in what
the command line left unset, so every flag still overrides it.

    FOVEAX.exe                      -> release profile (frozen app)
    python src/13_realtime_dashboard.py            -> full profile
    ... --dataset-profile release   -> force either one anywhere
    ... --data-root D:/rellis3d     -> same profile, different data location
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PROFILES: dict[str, dict] = {
    "full": {
        "description": "Full local datasets, every frame",
        "data_root": "C:/dev/data/rellis3d",
        "data_root_env": "FOVEAX_RELLIS_ROOT",
        "sequence": "00001",
        "frames": 0,  # 0 = every frame the data has
        "rate_hz": 25.0,
        "banner": "Dataset: RELLIS-3D sequence {sequence}, {frames:,} frames (full dataset)",
    },
    "release": {
        "description": "Packaged app with the bundled sample",
        "data_root": "data/rellis3d",  # relative to the exe folder (or the repo when run from source)
        "data_root_env": None,
        "sequence": "00001",
        "frames": 0,
        "rate_hz": 25.0,
        "banner": "Sample data: {frames:,} frames (full dataset not included)",
    },
}

FALLBACK_FRAMES = 3000  # sources whose length cannot be counted up front


def app_base_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else REPO_ROOT


def default_profile_name() -> str:
    return "release" if getattr(sys, "frozen", False) else "full"


def rellis_frame_count(data_root: Path, sequence: str) -> int:
    d = Path(data_root) / "Rellis-3D" / sequence / "os1_cloud_node_kitti_bin"
    return len(list(d.glob("*.bin"))) if d.is_dir() else 0


def apply_profile(args) -> None:
    """Resolve args.dataset_profile / data_root / sequence / frames / rate_hz and set
    args.dataset_banner. Call after parse_args() and after resolve_input()."""
    name = getattr(args, "dataset_profile", None) or default_profile_name()
    if name not in PROFILES:
        raise SystemExit(f"Unknown --dataset-profile {name!r}; choose one of {sorted(PROFILES)}")
    prof = PROFILES[name]
    args.dataset_profile = name

    if getattr(args, "rate_hz", None) is None:
        args.rate_hz = float(prof["rate_hz"])
    custom = getattr(args, "lidar_source", None) is not None

    if custom:  # --input-config / --input-dir: the profile's dataset does not apply
        n = len(args.lidar_source)
        if not args.frames:
            args.frames = n
        args.data_root = None
        args.dataset_banner = f"Dataset: {args.lidar_source.name}, {n:,} frames"
        return

    if args.source != "rellis3d":
        if args.frames is None or args.frames == 0:
            args.frames = FALLBACK_FRAMES
        args.data_root = getattr(args, "data_root", None)
        args.dataset_banner = None
        return

    if args.sequence is None:
        args.sequence = prof["sequence"]
    root = getattr(args, "data_root", None)
    if root is None and prof["data_root_env"] and os.environ.get(prof["data_root_env"]):
        root = os.environ[prof["data_root_env"]]
    if root is None:
        root = prof["data_root"]
    root = Path(root)
    if not root.is_absolute():
        root = app_base_dir() / root
    args.data_root = root

    available = rellis_frame_count(root, args.sequence)
    if available == 0:
        hint = ("Pass --data-root, set FOVEAX_RELLIS_ROOT, or use --dataset-profile release for the bundled sample."
                if name == "full" else "The sample data folder is missing; reinstall from the release zip.")
        raise SystemExit(f"No RELLIS-3D frames found for sequence {args.sequence} under {root}.\n{hint}")
    if args.frames is None or args.frames == 0:
        args.frames = available
    shown = min(args.frames, available)
    args.dataset_banner = prof["banner"].format(sequence=args.sequence, frames=shown)
