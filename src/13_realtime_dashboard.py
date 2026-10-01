import sys
import argparse
import importlib
from pathlib import Path

# Fix sys.path for relative imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# torch must be imported before PyQt5 in this process: on Windows, PyQt5's
# bundled Qt DLLs conflict with torch's bundled CUDA DLLs (c10.dll) when
# PyQt5 loads first, causing an import-time crash (WinError 1114 / access
# violation). Only matters once a real (non-mock) torch-based detector is
# wired in below, but the ordering has to be set here regardless.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

from src.dashboard.qt_main_window import FoveaXDashboardWindow
from src.dashboard.open3d_viewer import FoveaX3DViewer
from src.dashboard.data_streamer import DataStreamerThread
from src.perception.object_detector import MockObjectDetector

def resolve_input(args):
    """Build the optional custom LiDAR source and fill in sensor settings on *args*.

    Sets args.lidar_source (None for the built-in sources) and args.sensor_hz.
    """
    args.lidar_source = None
    if getattr(args, "input_config", None) or getattr(args, "input_dir", None):
        from src.inputs import load_source_config, source_from_dir

        if args.input_config and args.input_dir:
            raise SystemExit("Use either --input-config or --input-dir, not both.")
        try:
            args.lidar_source = (load_source_config(args.input_config) if args.input_config
                                 else source_from_dir(args.input_dir))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Input error: {exc}")
        args.source = "custom"
        print(f"[INFO] Input: {args.lidar_source.describe()}")
        # Sensor geometry from the config feeds SalsaNext unless set explicitly.
        for key in ("fov_up", "fov_down"):
            if getattr(args, key, None) is None and args.lidar_source.sensor.get(key) is not None:
                setattr(args, key, float(args.lidar_source.sensor[key]))
    if getattr(args, "sensor_hz", None) is None:
        args.sensor_hz = args.lidar_source.sensor_hz if args.lidar_source is not None else 10.0


def parse_args():
    parser = argparse.ArgumentParser(description="FOVEAX Phase 9 - Real-Time Dashboard")
    parser.add_argument(
        "--source", type=str, choices=["sample", "semantickitti", "rellis3d", "custom"],
        default="rellis3d", help="Data source ('sample', 'semantickitti', 'rellis3d'; "
                                 "'custom' is set automatically by --input-config/--input-dir)"
    )
    parser.add_argument(
        "--input-config", type=str, default=None,
        help="YAML/JSON file describing any LiDAR dataset or sensor dump "
             "(see docs/input_sources.md and configs/inputs/)."
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Shortcut: a folder of point-cloud files (bin/npy/npz/ply/pcd/csv/txt), format auto-detected."
    )
    parser.add_argument(
        "--sensor-hz", type=float, default=None,
        help="Scan rate of the sensor that recorded the data, used for timestamps and m/s speeds "
             "(default: the input config's sensor.hz, else 10). Separate from --rate-hz, the playback speed."
    )
    parser.add_argument(
        "--dataset-profile", type=str, default=None, choices=["full", "release"],
        help="Launch profile (src/dashboard/profiles.py): 'full' = local full dataset, every frame "
             "(default from source / Launch_GUI.bat); 'release' = the bundled sample next to the exe "
             "(default for the packaged app). Same code either way; only data location and defaults differ."
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="RELLIS-3D folder (contains Rellis-3D/<sequence>/...); overrides the profile's location "
             "(for the full profile also settable with the FOVEAX_RELLIS_ROOT environment variable)."
    )
    parser.add_argument(
        "--golden-check", action="store_true",
        help="Run the real pipeline on the first 10 real frames and compare with the stored reference "
             "output (tests/golden/); prints PASS/FAIL and exits. Works in the packaged exe too."
    )
    parser.add_argument(
        "--bench-frames", type=int, default=None,
        help="Open the real GUI, play this many frames, print the painted/computed frame rates and exit."
    )
    parser.add_argument(
        "--detector", type=str, choices=["mock", "pointpillars"],
        default="mock", help="Detector type"
    )
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="Sequence name (e.g. '00' for KITTI, '00000' for RELLIS-3D)"
    )
    parser.add_argument(
        # Default: the curated ~2-minute real demo range. Sequence 00001
        # alone (real, unstitched) was confirmed to contain all three
        # terrain features -- real overhead obstacles from frame 0 onward,
        # a real depression (~3.5m into the corridor), and a real height
        # bump (~4.8m) -- see docs referenced in the task report. At the
        # --rate-hz default below (25 FPS, what the real GUI paints when paced,
        # measured on the development laptop) 3000 frames would be exactly 2:00;
        # sequence 00001 has 2319 real frames, so playback ends at frame 2319
        # (~1:33) -- no stitching or looping needed.
        "--frames", type=int, default=None,
        help="Number of frames to play (default: every frame the data has -- set by the launch profile, "
             "see src/dashboard/profiles.py; a number larger than the data simply ends at its last frame)."
    )
    parser.add_argument(
        "--rate-hz", type=float, default=None,
        help="Playback rate cap in Hz (default from the launch profile: 25 = the rate the real GUI actually paints when paced, "
             "measured on the development laptop; 1000 = free-running, which painted about 28 FPS there; "
             "run tools/bench_gui_throughput.py to find your machine's rate)"
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="Worker processes that compute upcoming frames in parallel while the current "
             "one is tracked and drawn (-1 = auto: half the physical cores plus one, at most 6; "
             "0 = off, fully sequential). Output is identical either way. Ignored when a "
             "SalsaNext/mock semantic predictor is active (sequential then)."
    )
    parser.add_argument(
        "--display-voxel", type=float, default=0.15,
        help="Voxel size (m) of the thinned point cloud sent to the 3D VIEW ONLY; every "
             "perception stage still uses all points (0 = send the full cloud)."
    )
    parser.add_argument(
        "--ego-motion-every-n", type=int, default=30,
        help="Run ego-motion estimation every N frames (higher reduces CPU load)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run streamer in headless mode without GUI/Open3D display (writes telemetry to outputs/phase9/session/)"
    )
    parser.add_argument(
        "--separate-windows", action="store_true",
        help="Force the original two-window layout (2D dashboard + a separate "
             "floating Open3D window) even if pywin32 is available. By default, "
             "when pywin32 is installed the Open3D view is embedded as a panel "
             "inside the main dashboard window."
    )
    parser.add_argument(
        "--screenshot-after", type=int, default=None,
        help="Save a QWidget.grab() of the main window to --screenshot-path "
             "after this many frames, then exit. Avoids relying on desktop "
             "screen capture (which can grab the wrong window when other "
             "windows overlap this one)."
    )
    parser.add_argument(
        "--screenshot-path", type=str, default="outputs/phase9/final_dashboard_confirmed.png",
    )
    parser.add_argument(
        "--capture-sequence", type=str, default=None,
        help="Advance through the entire real sequence saving a screenshot "
             "every --capture-every frames to this directory, then exit."
    )
    parser.add_argument(
        "--capture-every", type=int, default=10,
        help="Frame interval for --capture-sequence (default: every 10th real frame)."
    )
    parser.add_argument(
        "--predictor", type=str, choices=["ground_truth", "mock", "salsanext"],
        default="ground_truth",
        help="Semantic class source for the live accuracy comparison "
             "(default: 'ground_truth', matching export_web_dashboard_data.py's "
             "own default -- comparing ground truth against itself, trivially "
             "~100%%, clearly labeled as such). Use 'salsanext' to compare a "
             "real model prediction against real ground truth."
    )
    parser.add_argument("--salsanext-repo", type=str, default="external/SalsaNext")
    parser.add_argument("--checkpoint", type=str,
                         default="models/salsanext/pretrained/pretrained/SalsaNext")
    parser.add_argument("--config", type=str,
                         default="models/salsanext/pretrained/pretrained/arch_cfg.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fov-up", type=float, default=None)
    parser.add_argument("--fov-down", type=float, default=None)
    parser.add_argument("--rescale-intensity", action="store_true", default=False)
    parser.add_argument(
        "--profile", action="store_true",
        help="Run the per-stage timing harness (StageTimer) instead of the live GUI: "
             "--profile-warmup unrecorded frames, then --profile-frames recorded frames of "
             "the same fixed sequence, printing per-stage mean/p50/p95 and %% of frame time, "
             "plus a 1 Hz nvidia-smi log. Off by default; changes no pipeline behavior."
    )
    parser.add_argument("--profile-frames", type=int, default=500)
    parser.add_argument("--profile-warmup", type=int, default=30)
    return parser.parse_args()

from multiprocessing import Process, Queue
import time

class DashboardApplication:
    def __init__(self, args):
        self.app = QApplication(sys.argv)

        # --- Decide whether to attempt embedding the Open3D window ------
        # This is a windowing/visual change only: the Open3D render loop
        # still runs in its own process (self.o3d_process below), on its
        # own event loop, completely separate from the Qt event loop here
        # -- that split is unchanged and is not what this feature touches.
        from src.dashboard.win32_embed import WIN32_AVAILABLE
        self._attempt_embed = WIN32_AVAILABLE and not getattr(args, "separate_windows", False)
        self._embedded = False
        if WIN32_AVAILABLE and getattr(args, "separate_windows", False):
            print("[INFO] --separate-windows requested; using the original two-window layout.")
        elif not WIN32_AVAILABLE:
            print("[INFO] pywin32 (win32gui) is not installed -- falling back to the "
                  "original two-window layout. Install it with 'pip install pywin32' "
                  "to combine the 3D view into the main dashboard window.")

        # Initialize UI Components
        self.qt_window = FoveaXDashboardWindow(embed_3d=self._attempt_embed)

        # Multiprocessing for Open3D Viewer (unchanged: separate process,
        # separate render loop, communicated with only via self.o3d_queue).
        self.o3d_queue = Queue(maxsize=10)
        from src.dashboard.open3d_viewer import run_open3d_process
        self._ready_queue = Queue() if self._attempt_embed else None
        # Box labels are drawn by Qt over the embedded view, so they only
        # exist in the embedded layout.
        self._label_queue = Queue(maxsize=4) if self._attempt_embed else None
        self.o3d_process = Process(
            target=run_open3d_process,
            args=(self.o3d_queue, self._ready_queue),
            kwargs={"label_queue": self._label_queue},
        )
        self.o3d_process.start()

        # Real camera controls: the left panel's view/zoom/rotate/reset
        # buttons emit this signal, forwarded here as a command dict onto
        # the same o3d_queue (run_open3d_process distinguishes command
        # dicts from FrameState/QUIT items, see open3d_viewer.py). This
        # drives Open3D's actual ViewControl API, not a simulated camera.
        self.qt_window.camera_command_requested.connect(self._send_camera_command)

        # Detector
        if args.detector == "mock":
            detector = MockObjectDetector()
        else:
            detector = MockObjectDetector()
            print("[WARNING] PointPillars requested but not fully implemented in CLI args, using mock.")
            
        # Sequence default: "00001" for rellis3d -- the curated ~2-minute
        # real demo sequence (see --frames/--rate-hz defaults above).
        seq = args.sequence
        if seq is None:
            seq = "00001" if args.source == "rellis3d" else "00"

        # --- Playback controller ----------------------------------------
        # This dashboard replays a REAL, finite, recorded sequence of
        # LiDAR frames (RELLIS-3D/SemanticKITTI files, or the deterministic
        # synthetic sample generator) -- it is not a live sensor feed, so
        # frames are addressable by index and this can be a real pull-based
        # playback controller rather than a one-way autonomous stream.
        #
        # DataStreamerThread.process_frame(idx, ts, fps) is already a pure,
        # synchronous, fully-tested method that computes one real FrameState
        # for an arbitrary frame index (run_headless() already calls it this
        # way). Reusing it directly here -- driven by a single QTimer on the
        # Qt thread instead of DataStreamerThread's own background QThread
        # loop -- means every panel (3D view, 2D map, tracked objects,
        # hazard table, centerline profile) is updated from the exact same
        # FrameState object on every tick. The previous design ran two
        # independent, uncoordinated throttles (a fixed 10 FPS gate on the
        # 2D panels here, and a separate drop-oldest queue for the 3D view)
        # that could show two different frames on screen at once; this
        # design cannot desync because there is only one frame in flight.
        # Real semantic predictor for the live accuracy comparison
        # (Task: live metrics). Reuses
        # src/dashboard/export_web_dashboard_data.py::make_semantic_predictor
        # directly -- the exact same constructor already used and
        # validated by the web-export bridge -- rather than duplicating
        # its predictor-selection logic here.
        semantic_predictor = None
        if args.predictor != "ground_truth":
            export_module = importlib.import_module("src.dashboard.export_web_dashboard_data")
            semantic_predictor = export_module.make_semantic_predictor(args.predictor, args)

        # Data location comes from the launch profile (src/dashboard/profiles.py):
        # the full local dataset for Launch_GUI.bat, the bundled sample for the packaged app.
        data_root = getattr(args, "data_root", None)

        self.streamer = DataStreamerThread(
            source=args.source,
            detector=detector,
            num_frames=args.frames,
            sequence=seq,
            rate_hz=args.rate_hz,
            semantic_predictor=semantic_predictor,
            predictor_mode=args.predictor,
            data_root=data_root,
            lidar_source=getattr(args, "lidar_source", None),
            sensor_hz=getattr(args, "sensor_hz", None),
        )
        self.streamer._setup_sources()
        # A recorded sequence can hold fewer frames than --frames (e.g. the
        # small sample bundled with the packaged app). Playing past the last
        # file would wrap around and jump back to the start mid-run, so clamp.
        available = self.streamer.available_frames()
        if available:
            self.streamer.num_frames = min(args.frames, available)

        # Cache of already-computed real FrameStates, keyed by frame index.
        # Stepping backward replays an already-computed real frame from
        # this cache -- it never re-runs the (stateful, Kalman-filtered)
        # tracker "backward in time", which would corrupt its real track
        # history. Stepping forward past the cache computes a new real
        # frame exactly once, advancing the tracker forward exactly as
        # normal playback always has.
        self._frame_cache: dict[int, object] = {}
        self._playback_idx = -1
        self._playing = True
        self._rate_hz = args.rate_hz
        self._frame_render_times: list[float] = []
        self._achieved_fps: float = self._rate_hz

        self.qt_window.play_pause_requested.connect(self._toggle_play_pause)
        self.qt_window.step_forward_requested.connect(lambda: self._show_frame(self._playback_idx + 1))
        self.qt_window.step_backward_requested.connect(lambda: self._show_frame(self._playback_idx - 1))
        self.qt_window.rate_changed.connect(self._set_rate_hz)
        self.qt_window.restart_requested.connect(self._restart)
        self.qt_window.environment_requested.connect(self._switch_environment)
        self.qt_window.load_dataset_requested.connect(self._load_custom_dataset)
        if self.streamer.lidar_source is not None:
            # The two buttons switch between RELLIS-3D sequences; a custom input has
            # no such sequences, so showing them would mislabel the data.
            self.qt_window.env1_btn.hide()
            self.qt_window.env2_btn.hide()
        else:
            # Only offer a sequence that is actually on disk (the packaged sample has just 00001).
            from src.dashboard.profiles import rellis_frame_count

            root = data_root or Path("C:/dev/data/rellis3d")
            for btn, name in ((self.qt_window.env1_btn, "00001"), (self.qt_window.env2_btn, "00003")):
                if args.source != "rellis3d" or rellis_frame_count(root, name) == 0:
                    btn.hide()
        self.qt_window.set_dataset_banner(getattr(args, "dataset_banner", None),
                                          sample=getattr(args, "dataset_profile", None) == "release")
        self.qt_window.set_rate_hz(self._rate_hz)

        # Single-shot, re-armed after every frame for only the time LEFT in the
        # frame budget (see _on_timer_tick). A periodic QTimer waits a whole extra
        # interval after a frame that overran its budget, which left a third of each
        # frame idle at 30 Hz.
        # PreciseTimer: the default coarse timer on Windows has ~15.6 ms granularity, so a
        # requested 21 ms wait fires after ~31 ms and a capped rate ran 10-20% slower than
        # an uncapped one (measured, alternating runs).
        self.playback_timer = QTimer()
        self.playback_timer.setTimerType(Qt.PreciseTimer)
        self.playback_timer.setSingleShot(True)
        self.playback_timer.timeout.connect(self._on_timer_tick)
        self._apply_timer_interval()

        # Throttle the informational, real point-cloud-registration (ICP)
        # ego-displacement estimate (src/perception/ego_motion_estimate.py)
        # -- it costs tens of ms per real frame pair, real but non-trivial
        # next to a 10 Hz (100 ms) frame budget, and it is purely
        # informational (never used to reposition any view), so it is
        # computed only every few real frames rather than every one.
        self._ego_motion_every_n_frames = args.ego_motion_every_n
        self._last_ego_motion_m: float | None = None
        self._last_ego_frame: int | None = None  # frame the estimate was computed at

        # Frames kept for STEP back. Capped so a long run cannot fill RAM (each
        # state holds a full 131k-point cloud, ~2 MB); older frames drop first.
        self._frame_cache_max = 600
        self.streamer.display_voxel_m = float(getattr(args, "display_voxel", 0.15))
        self._pipeline = None
        self._finisher = None
        self._pipeline_workers = getattr(args, "workers", -1)
        self._start_pipeline()

        self._screenshot_after = getattr(args, "screenshot_after", None)
        self._screenshot_path = getattr(args, "screenshot_path", None)

    # ---- parallel frame pipeline (src/dashboard/frame_pipeline.py) ---------
    def _pipeline_allowed(self) -> bool:
        # A semantic predictor holds a model (GPU) that is not shared with workers.
        return self._pipeline_workers != 0 and self.streamer.semantic_predictor is None

    def _start_pipeline(self):
        if not self._pipeline_allowed():
            return
        from src.dashboard.frame_pipeline import FramePipeline, config_from_streamer, default_worker_count

        workers = default_worker_count() if self._pipeline_workers < 0 else self._pipeline_workers
        try:
            self._pipeline = FramePipeline(
                config_from_streamer(self.streamer), workers, self._ego_motion_every_n_frames)
            self._pipeline.prefetch(0, self.streamer.timestamp_for)  # workers boot while the window opens
            self._start_finisher(0)
            print(f"[INFO] Frame pipeline: {workers} worker process(es) computing frames ahead, "
                  "tracking on a background thread (output identical to sequential).")
        except Exception as exc:
            self._pipeline = None
            print(f"[WARNING] Could not start the frame pipeline ({exc}); running sequentially.")

    def _reset_pipeline(self):
        """After a restart / environment switch / dataset load: drop queued work and
        rebuild the workers only if their configuration changed."""
        if not self._pipeline_allowed():
            self._stop_pipeline()
            return
        from src.dashboard.frame_pipeline import config_from_streamer, config_signature

        if self._pipeline is not None:
            if config_signature(config_from_streamer(self.streamer)) == self._pipeline.signature:
                self._pipeline.num_frames = self.streamer.num_frames
                self._pipeline.cancel()
                self._pipeline.prefetch(0, self.streamer.timestamp_for)
                self._start_finisher(0)
                return
            self._stop_pipeline()
        self._start_pipeline()

    def _start_finisher(self, start_idx: int):
        from src.dashboard.frame_pipeline import FrameFinisher

        self._stop_finisher()
        self._finisher = FrameFinisher(
            self._pipeline, self.streamer, start_idx, self.streamer.timestamp_for,
            lambda: self._achieved_fps)

    def _stop_finisher(self):
        """Must run before the tracker is replaced or the pipeline is changed."""
        if getattr(self, "_finisher", None) is not None:
            self._finisher.stop()
            self._finisher = None

    def _stop_pipeline(self):
        self._stop_finisher()
        if self._pipeline is not None:
            try:
                self._pipeline.shutdown()
            except Exception:
                pass
            self._pipeline = None

    def _compute_new_frame(self, idx: int, ts: float):
        """FrameState for a frame not computed before: from the worker pool when
        active (then tracked here, in frame order), else fully sequential."""
        pipe = self._pipeline
        if pipe is not None:
            try:
                state, ego = self._finisher.get(idx)
                from src.dashboard.frame_pipeline import _NO_EGO

                if ego is not _NO_EGO:
                    self._last_ego_motion_m = ego
                    self._last_ego_frame = idx
                return state
            except Exception as exc:
                print(f"[WARNING] Frame pipeline failed ({exc!r}); continuing sequentially.")
                self._stop_pipeline()
        state = self.streamer.process_frame(idx, ts, fps=self._achieved_fps)
        # Informational-only real ego-displacement estimate, computed against
        # the immediately preceding REAL frame's real points (never fabricated),
        # throttled per the comment in __init__.
        if idx > 0 and idx % self._ego_motion_every_n_frames == 0 and (idx - 1) in self._frame_cache:
            try:
                from src.perception.ego_motion_estimate import estimate_relative_displacement_m
                prev_pts = self._frame_cache[idx - 1].points[:, :3]
                self._last_ego_motion_m = estimate_relative_displacement_m(prev_pts, state.points[:, :3])
            except Exception:
                self._last_ego_motion_m = None
            self._last_ego_frame = idx
        return state

    def _send_camera_command(self, cmd: str):
        """Forward a real camera command (from the dashboard's left panel)
        to the Open3D process via the shared o3d_queue."""
        try:
            self.o3d_queue.put_nowait({"cmd": cmd})
        except Exception:
            pass

    def _apply_timer_interval(self):
        interval_ms = max(1, int(round(1000.0 / self._rate_hz)))
        self.playback_timer.setInterval(interval_ms)

    def _set_rate_hz(self, hz: float):
        self._rate_hz = max(1.0, hz)
        self._apply_timer_interval()

    def _toggle_play_pause(self):
        self._playing = not self._playing
        self.qt_window.set_play_pause_state(self._playing)

    def _switch_environment(self, seq: str):
        """Switch between real RELLIS-3D recorded sequences."""
        if self.streamer.sequence == seq:
            return
            
        self.streamer.sequence = seq
        self.streamer._setup_sources()
        
        # Default view on switch: Traversability
        # so the difference (non-drivable cells) is immediately visible
        self.qt_window._on_map_type_clicked("traversability") if hasattr(self.qt_window, '_on_map_type_clicked') else self.qt_window._on_map_changed("traversability")
        self.qt_window.set_active_environment(seq)
        self._restart()

    def _load_custom_dataset(self, fmt, data_root, seq, kwargs):
        """Load a user-selected dataset and restart the dashboard."""
        from pathlib import Path
        self.streamer.source = fmt
        self.streamer.data_root = Path(data_root) if data_root else None
        self.streamer.sequence = seq
        
        # Apply overhead clearance parameters
        self.streamer.ground_noise_m = float(kwargs.get("ground_noise_m", 0.30))
        self.streamer.clearance_threshold_m = float(kwargs.get("clearance_threshold_m", 2.0))
        self.streamer.overhead_min_points = int(kwargs.get("min_points", 4))

        # Apply detector clustering & voxel parameters
        det = self.streamer.detector
        if hasattr(det, "cluster_min_points"):
            det.cluster_min_points = int(kwargs.get("cluster_min_points", kwargs.get("min_points", 5)))
        if hasattr(det, "voxel_size_m"):
            det.voxel_size_m = float(kwargs.get("voxel_size_m", 0.2))
        if hasattr(det, "ground_threshold_m"):
            det.ground_threshold_m = float(kwargs.get("ground_threshold_m", 0.25))

        import src.perception.overhead_detection as od
        if hasattr(od, "DEFAULT_GROUND_NOISE_M"):
            od.DEFAULT_GROUND_NOISE_M = self.streamer.ground_noise_m
        if hasattr(od, "DEFAULT_MIN_POINTS"):
            od.DEFAULT_MIN_POINTS = self.streamer.overhead_min_points
        if hasattr(od, "DEFAULT_CLEARANCE_THRESHOLD_M"):
            od.DEFAULT_CLEARANCE_THRESHOLD_M = self.streamer.clearance_threshold_m
            
        self.streamer._setup_sources()
        available = self.streamer.available_frames()
        if available:
            self.streamer.num_frames = available
            
        self.qt_window._on_map_changed("traversability")
        self.qt_window.set_active_environment("CUSTOM")
        self.qt_window.set_dataset_banner(f"Dataset: {fmt}, {self.streamer.num_frames:,} frames (loaded by user)")
        self._restart()


    def _restart(self):
        """Reset the curated real playback to frame 0 with completely
        fresh tracker state -- no track history, hazard clusters, or
        accuracy numbers carry over from the previous run.

        A fresh MultiObjectTracker() clears all track IDs back to 0 (its
        own __init__ resets _next_track_id and _tracks). The frame cache
        is also dropped entirely: cached FrameStates were computed against
        the OLD tracker instance, so keeping them would silently resurrect
        the old track history the moment playback revisited a cached
        frame index.
        """
        self.playback_timer.stop()
        self._stop_finisher()  # it owns the tracker while it runs
        from src.tracking.multi_object_tracker import MultiObjectTracker
        self.streamer.tracker = MultiObjectTracker()
        self.streamer.reset_caches()
        self._frame_cache.clear()
        self._reset_pipeline()
        self._frame_render_times.clear()
        self._achieved_fps = self._rate_hz
        self._playback_idx = -1
        self._last_ego_motion_m = None
        self._last_ego_frame = None
        self._show_frame(0)
        self._playing = True
        self.qt_window.set_play_pause_state(True)
        self.playback_timer.start()

    def _on_timer_tick(self):
        t0 = time.perf_counter()
        if self._playing and self._playback_idx + 1 < self.streamer.num_frames:
            # (past the last real frame we simply hold on it)
            self._show_frame(self._playback_idx + 1)
        # Wait only for what remains of this frame's budget; a frame that already
        # overran it starts the next one immediately. Nothing is skipped.
        budget_ms = 1000.0 / self._rate_hz
        self.playback_timer.start(max(0, int(round(budget_ms - (time.perf_counter() - t0) * 1000.0))))

    def _show_frame(self, idx: int):
        """Compute (or replay from cache) the real frame at *idx* and push
        the SAME FrameState to every panel: 3D view, 2D map, tracked
        objects, hazard table, and the centerline profile."""
        idx = max(0, min(idx, self.streamer.num_frames - 1))
        if idx == self._playback_idx and idx in self._frame_cache:
            return  # already showing this exact real frame

        went_backward = idx < self._playback_idx
        if went_backward and idx not in self._frame_cache:
            # Older than the kept history: recomputing it now would run the
            # tracker backward in time and corrupt its state, so stay put.
            print(f"[INFO] Frame {idx + 1} is older than the {self._frame_cache_max} kept frames; "
                  "cannot step back that far.")
            return
        self._playback_idx = idx

        import time
        now = time.perf_counter()
        self._frame_render_times.append(now)
        if len(self._frame_render_times) > 15:
            self._frame_render_times.pop(0)
        if len(self._frame_render_times) >= 2:
            dt = self._frame_render_times[-1] - self._frame_render_times[0]
            if dt > 0:
                self._achieved_fps = (len(self._frame_render_times) - 1) / dt
        else:
            self._achieved_fps = self._rate_hz

        if idx in self._frame_cache:
            state = self._frame_cache[idx]
        else:
            ts = self.streamer.timestamp_for(idx)
            state = self._compute_new_frame(idx, ts)
            self._frame_cache[idx] = state
            while len(self._frame_cache) > self._frame_cache_max:
                del self._frame_cache[min(self._frame_cache)]

        # The 3D process draws this same frame's tracks, over a voxel-thinned copy
        # of the cloud (display only; the 2D panels and every metric use all
        # points). Only what it reads is sent, not the whole FrameState.
        try:
            from src.dashboard.dashboard_state import FrameState

            slim = FrameState(
                frame_idx=state.frame_idx, timestamp_s=state.timestamp_s, tracks=state.tracks,
                points=state.display_points if state.display_points is not None else state.points,
            )
            if self.o3d_queue.full():
                self.o3d_queue.get_nowait()
            self.o3d_queue.put_nowait(slim)
        except Exception:
            pass

        self.qt_window.update_ui(state)
        self.qt_window.set_frame_counter(idx + 1, self.streamer.num_frames)
        self.qt_window.set_ego_motion_estimate(
            self._last_ego_motion_m, self._last_ego_frame, self._ego_motion_every_n_frames)

        if went_backward:
            print(f"[INFO] Stepped back to real frame {idx + 1}/{self.streamer.num_frames} "
                  "(replayed from cache, tracker not re-run).")

        self._maybe_take_screenshot(state)

    def _maybe_take_screenshot(self, state):
        if self._screenshot_after is None or self._playback_idx + 1 < self._screenshot_after:
            return
        self._screenshot_after = None  # only once
        from pathlib import Path
        out_path = Path(self._screenshot_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.qt_window.grab().save(str(out_path))
        print(f"[INFO] Screenshot saved to {out_path}")
        for i, tab_name in enumerate(["status", "terrain", "system"]):
            self.qt_window.tabs.setCurrentIndex(i)
            self.app.processEvents()
            tab_path = out_path.with_name(out_path.stem + f"_{tab_name}" + out_path.suffix)
            self.qt_window.grab().save(str(tab_path))
            print(f"[INFO] Screenshot saved to {tab_path}")

        self.qt_window.tabs.setCurrentIndex(0)
        self.qt_window._on_map_changed("overhead")
        self.qt_window.update_ui(state)  # force a re-render with the new layer selection
        self.app.processEvents()
        overhead_path = out_path.with_name(out_path.stem + "_overhead" + out_path.suffix)
        self.qt_window.grab().save(str(overhead_path))
        print(f"[INFO] Screenshot saved to {overhead_path}")
        self.qt_window.tabs.setCurrentIndex(0)
        self.qt_window._on_map_changed("elevation")
        self.app.processEvents()

        # Capture both PLAY and PAUSE button states for visual review
        # (Task: PAUSE/PLAY contrast fix).
        self.qt_window.set_play_pause_state(True)
        self.app.processEvents()
        play_path = out_path.with_name(out_path.stem + "_playing" + out_path.suffix)
        self.qt_window.grab().save(str(play_path))
        print(f"[INFO] Screenshot saved to {play_path}")

        self.qt_window.set_play_pause_state(False)
        self.app.processEvents()
        pause_path = out_path.with_name(out_path.stem + "_paused" + out_path.suffix)
        self.qt_window.grab().save(str(pause_path))
        print(f"[INFO] Screenshot saved to {pause_path}")

        self.app.quit()

    def _capture_frame_sequence(self, out_dir: str, every_n: int):
        """Advance through the real sequence, saving a screenshot every
        *every_n* real frames to *out_dir*, then quit. Used for the
        motion-playback verification sequence (--capture-sequence)."""
        from pathlib import Path
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        self.playback_timer.stop()
        self._playing = False

        for idx in range(self.streamer.num_frames):
            self._show_frame(idx)
            self.app.processEvents()
            if idx % every_n == 0 or idx == self.streamer.num_frames - 1:
                path = out_dir_path / f"frame_{idx:04d}.png"
                self.qt_window.grab().save(str(path))
                print(f"[INFO] Sequence screenshot saved: {path}")
        self.app.quit()

    def _try_reparent(self):
        """Wait for the Open3D child process to report its window handle,
        then reparent it into self.qt_window.embed_widget.

        Called only when self._attempt_embed is True (win32gui available
        and not opted out). Any failure here -- timeout waiting for the
        handle, or an error during the actual reparent call -- is caught
        and logged; the Open3D window simply remains a normal, separate,
        free-floating top-level window in that case (the pre-existing
        behavior), and nothing else about the app's operation changes.
        """
        # The Open3D child must spawn, import torch/open3d and create its GLFW
        # window before it can report a handle -- routinely well over 8s on a
        # loaded GPU. A short blocking wait here gave up early and left the
        # 3D view floating, so poll on a QTimer inside the running event loop
        # instead, for as long as the child itself keeps searching.
        self._reparent_deadline = time.time() + 60.0
        self._reparent_timer = QTimer()
        self._reparent_timer.setInterval(100)
        self._reparent_timer.timeout.connect(self._poll_reparent)
        self._reparent_timer.start()

    def _poll_reparent(self):
        try:
            child_hwnd = self._ready_queue.get_nowait()
        except Exception:
            if time.time() > self._reparent_deadline:
                self._reparent_timer.stop()
                print("[WARNING] Open3D window handle never arrived -- the 3D view "
                      "stays a separate floating window.", flush=True)
            return

        self._reparent_timer.stop()
        if not child_hwnd:
            print("[WARNING] Open3D child could not find its own window -- the 3D "
                  "view stays a separate floating window.", flush=True)
            return
        self._embed_child(child_hwnd)

    def _embed_child(self, child_hwnd: int):
        from src.dashboard.win32_embed import reparent_as_child

        try:
            embed_widget = self.qt_window.embed_widget
            # Force real native window handles to exist before SetParent.
            self.qt_window.winId()
            embed_widget.winId()
            parent_hwnd = int(embed_widget.winId())
            reparent_as_child(child_hwnd, parent_hwnd, embed_widget.width(), embed_widget.height())
            embed_widget.child_hwnd = child_hwnd
            self._embedded = True
            print("[INFO] Open3D 3D view embedded into the main dashboard window.", flush=True)

            self._label_timer = QTimer()
            self._label_timer.setInterval(33)
            self._label_timer.timeout.connect(self._drain_track_labels)
            self._label_timer.start()
        except Exception as e:
            print(f"[WARNING] Reparenting the Open3D window failed ({e!r}); "
                  "it will remain a separate floating window instead.", flush=True)

    def _drain_track_labels(self):
        labels = None
        try:
            while True:
                labels = self._label_queue.get_nowait()
        except Exception:
            pass
        if labels is not None:
            self.qt_window.embed_widget.set_track_labels(labels)
            # alert_container/threat_toast used to be fixed-position
            # overlays floating on top of the embedded 3D view, which meant
            # fighting its real native child windows (and these per-track
            # label widgets, also native) for z-order -- hence the repeated
            # bring_to_absolute_top() calls that used to live here. Both are
            # now laid out directly in the left panel instead (see
            # qt_main_window.py::_build_left_panel), so they no longer
            # overlap any native child window and need no special-casing.

    def run(self, capture_sequence: str | None = None, capture_every: int = 10):
        # Launch maximized: the fixed 1400x900 default (still used as the
        # restored-window size if the user un-maximizes) clips the right
        # panel's STATUS/TERRAIN/SYSTEM tabs and the FORWARD CORRIDOR
        # CROSS-SECTION panel below the fold, forcing a manual resize/zoom
        # just to see the full real layout on every launch.
        self.qt_window.showMaximized()
        if self._attempt_embed:
            self._try_reparent()

        self._show_frame(0)  # render the real first frame immediately

        if capture_sequence is not None:
            self._capture_frame_sequence(capture_sequence, capture_every)
        else:
            self.playback_timer.start()

        # Start event loop
        exit_code = self.app.exec_()

        # Cleanup
        self.playback_timer.stop()
        self._stop_pipeline()
        self.o3d_queue.put("QUIT")
        self.o3d_process.join(timeout=2.0)
        if self.o3d_process.is_alive():
            self.o3d_process.terminate()

        sys.exit(exit_code)

def run_headless(args):
    """Run data streamer without PyQt5 GUI or Open3D windows."""
    print("=" * 60)
    print("FOVEAX Phase 9 - Headless Session Replay & Telemetry")
    print("=" * 60)
    print(f"Source:     {args.source}")
    seq = args.sequence or ("00001" if args.source == "rellis3d" else "00")
    print(f"Sequence:   {seq}")
    print(f"Frames:     {args.frames}")
    print(f"Rate:       {args.rate_hz} Hz")
    print("-" * 60)

    detector = MockObjectDetector()
    streamer = DataStreamerThread(
        source=args.source,
        detector=detector,
        num_frames=args.frames,
        sequence=seq,
        rate_hz=args.rate_hz,
        lidar_source=getattr(args, "lidar_source", None),
        sensor_hz=getattr(args, "sensor_hz", None),
    )
    streamer._setup_sources()

    dt = 1.0 / args.rate_hz
    start_time = time.time()

    for idx in range(args.frames):
        ts = streamer.timestamp_for(idx)
        t0 = time.perf_counter()
        state = streamer.process_frame(idx, ts, fps=args.rate_hz)
        latency = (time.perf_counter() - t0) * 1000.0

        if (idx + 1) % 5 == 0 or idx == 0 or idx == args.frames - 1:
            print(
                f"[Frame {state.frame_idx + 1:3d}/{args.frames}] "
                f"Points: {len(state.points):6,d} | "
                f"Tracks: {len(state.tracks):2d} | "
                f"Latency: {latency:5.1f}ms | "
                f"CPU: {state.metrics.cpu_percent:4.1f}%"
            )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Headless replay completed in {total_time:.2f}s ({args.frames / max(total_time, 1e-3):.1f} FPS)")
    print("Session telemetry saved to outputs/phase9/session/dashboard_session.jsonl")
    print("=" * 60)


def run_profile(args):
    """Per-stage timing of the real pipeline (see src/dashboard/stage_timer.py).

    Drives the same objects the live GUI drives, per frame and in the same
    order as DashboardApplication._show_frame: DataStreamerThread.process_frame
    (with its internal stage regions), the every-Nth-frame ego-motion ICP,
    FoveaX3DViewer.update + poll_events, and FoveaXDashboardWindow.update_ui
    followed by a Qt event-loop drain (which is where the matplotlib canvases
    and widgets actually paint). The Open3D viewer and Qt window are real but
    off-screen (visible=False / WA_DontShowOnScreen) and run IN-PROCESS here so
    every frame maps 1:1 to a timing sample; the live dashboard instead runs
    the Open3D viewer in a separate process fed by a queue, so its rendering is
    decoupled from per-frame timing (see the printed caveats).
    """
    import subprocess

    from PyQt5.QtCore import Qt

    from src.dashboard.stage_timer import StageTimer

    warmup, measured = args.profile_warmup, args.profile_frames
    total_frames = warmup + measured
    seq = args.sequence or ("00001" if args.source == "rellis3d" else "00")

    print("=" * 78)
    print("FOVEAX per-stage profile")
    print("=" * 78)
    print(f"source={args.source} sequence={seq} predictor={args.predictor} "
          f"warmup_frames={warmup} measured_frames={measured} rate_hz_arg={args.rate_hz}")

    detector = MockObjectDetector()
    semantic_predictor = None
    if args.predictor != "ground_truth":
        export_module = importlib.import_module("src.dashboard.export_web_dashboard_data")
        semantic_predictor = export_module.make_semantic_predictor(args.predictor, args)

    streamer = DataStreamerThread(
        source=args.source, detector=detector, num_frames=total_frames, sequence=seq,
        rate_hz=args.rate_hz, semantic_predictor=semantic_predictor,
        predictor_mode=args.predictor,
        lidar_source=getattr(args, "lidar_source", None),
        sensor_hz=getattr(args, "sensor_hz", None),
    )
    streamer._setup_sources()

    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print(f"cuda device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    except Exception:
        pass
    timer = StageTimer(
        cuda_stages=frozenset({"h2d_transfer", "salsanext_forward"}) if cuda_ok else frozenset()
    )
    streamer.stage_timer = timer
    if semantic_predictor is not None and hasattr(semantic_predictor, "stage_timer"):
        semantic_predictor.stage_timer = timer
    else:
        for n in ("range_image_projection", "h2d_transfer", "salsanext_forward", "postprocessing"):
            timer.mark_not_applicable(n, f"predictor={args.predictor}: no SalsaNext inference is run")

    app = QApplication.instance() or QApplication(sys.argv)
    window = FoveaXDashboardWindow(embed_3d=False)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1536, 912)
    window.show()
    app.processEvents()
    viewer = FoveaX3DViewer(visible=False)

    ego_every_n = 5
    prev_points = None
    seen_metrics_fps = set()
    latency_ms_samples = []
    proc_frame_ms = []
    smi = None
    gpu_fh = None
    gpu_csv = Path("outputs/phase12/gpu_util_profile.csv")
    gpu_csv.parent.mkdir(parents=True, exist_ok=True)
    t_measured_start = None

    for idx in range(total_frames):
        if idx == 0:
            timer.set_recording(False)
        if idx == warmup:
            timer.set_recording(True)
            gpu_fh = open(gpu_csv, "w")
            smi = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw",
                 "--format=csv,noheader,nounits", "-l", "1"],
                stdout=gpu_fh, stderr=subprocess.DEVNULL,
            )
            t_measured_start = time.perf_counter()

        with timer.time("frame_total"):
            ts = streamer.timestamp_for(idx)
            t_pf = time.perf_counter()
            state = streamer.process_frame(idx, ts, fps=args.rate_hz)
            if idx >= warmup:
                proc_frame_ms.append((time.perf_counter() - t_pf) * 1000.0)
                seen_metrics_fps.add(round(state.metrics.fps, 6))
                latency_ms_samples.append(state.metrics.latency_ms)

            if idx > 0 and idx % ego_every_n == 0 and prev_points is not None:
                with timer.time("ego_motion_icp"):
                    from src.perception.ego_motion_estimate import estimate_relative_displacement_m
                    try:
                        estimate_relative_displacement_m(prev_points[:, :3], state.points[:, :3])
                    except Exception:
                        pass
            prev_points = state.points

            with timer.time("open3d_update"):
                viewer.update(state)
                viewer.poll_events()
            with timer.time("open3d_label_projection"):
                viewer.compute_track_labels()
            with timer.time("qt_ui_update"):
                window.update_ui(state)
            with timer.time("qt_event_loop_paint"):
                app.processEvents()

    wall_s = time.perf_counter() - t_measured_start
    if smi is not None:
        smi.terminate()
        smi.wait(timeout=5)
        gpu_fh.close()

    stage_order = [
        "load",
        "range_image_projection", "h2d_transfer", "salsanext_forward", "postprocessing",
        "live_accuracy_metrics",
        "clustering_heuristics",
        "kalman_tracking",
        "grid_maps_hazards",
        "metrics_state_logging",
        "ego_motion_icp",
        "open3d_update", "open3d_label_projection",
        "qt_ui_update", "qt_event_loop_paint",
    ]
    print()
    print(timer.report(stage_order, total_name="frame_total"))

    tot = timer.samples("frame_total")
    attributed = sum(sum(timer.samples(n)) for n in stage_order) / len(tot)
    tot_mean = sum(tot) / len(tot)
    print("-" * 78)
    print(f"unattributed (frame_total mean - sum of stage means, exact difference): "
          f"{tot_mean - attributed:.3f} ms = {100.0 * (tot_mean - attributed) / tot_mean:.1f}% of frame")
    print(f"measured frames: {len(tot)}   wall time for measured frames: {wall_s:.2f} s   "
          f"=> {len(tot) / wall_s:.2f} frames/s (includes profiling wrappers)")
    pf = sorted(proc_frame_ms)
    print(f"process_frame() alone (wall): mean {sum(pf)/len(pf):.3f} ms  "
          f"p50 {pf[len(pf)//2]:.3f}  p95 {pf[int(len(pf)*0.95)]:.3f}")
    print()
    print("ON-SCREEN COUNTER CHECK (values that state.metrics carried during the measured frames):")
    print(f"  distinct state.metrics.fps values seen: {sorted(seen_metrics_fps)}  "
          f"(args.rate_hz passed to process_frame = {args.rate_hz})")
    lm = sorted(latency_ms_samples)
    print(f"  state.metrics.latency_ms: mean {sum(lm)/len(lm):.3f}  p50 {lm[len(lm)//2]:.3f}  "
          f"p95 {lm[int(len(lm)*0.95)]:.3f}  (measured inside process_frame, t_start to just before metrics build)")

    print()
    print(f"GPU LOG (nvidia-smi, 1 Hz, measured phase only) -> {gpu_csv}")
    print("columns: timestamp, util.gpu %, util.mem %, mem.used MiB, clocks.sm MHz, power.draw W")
    rows = [l.strip() for l in open(gpu_csv) if l.strip()]
    for r in rows:
        print("  " + r)
    utils = []
    for r in rows:
        try:
            utils.append(float(r.split(",")[1]))
        except Exception:
            pass
    if utils:
        print(f"GPU util.gpu over {len(utils)} samples: mean {sum(utils)/len(utils):.1f}%  "
              f"min {min(utils):.0f}%  max {max(utils):.0f}%")
    print("=" * 78)


def main():
    args = parse_args()
    resolve_input(args)
    from src.dashboard.profiles import apply_profile
    apply_profile(args)
    print(f"[INFO] Launch profile: {args.dataset_profile}  |  data: "
          f"{args.data_root if args.data_root else args.source}  |  sequence: {args.sequence}  |  "
          f"frames: {args.frames}  |  target rate: {args.rate_hz:g} Hz")
    if args.golden_check:
        from src.dashboard.golden_check import run_golden_check
        sys.exit(run_golden_check(args))
    if args.bench_frames:
        from src.dashboard.gui_bench import run_gui_bench
        run_gui_bench(DashboardApplication(args), args.bench_frames)
    elif args.profile:
        run_profile(args)
    elif args.headless:
        run_headless(args)
    else:
        app = DashboardApplication(args)
        app.run(capture_sequence=args.capture_sequence, capture_every=args.capture_every)


if __name__ == "__main__":
    # Needed for the packaged (PyInstaller) build: the Open3D view runs in a
    # multiprocessing child, and a frozen exe re-launches itself to start it.
    from multiprocessing import freeze_support
    freeze_support()
    if getattr(sys, "frozen", False):
        # Sample data and outputs/ live next to the exe, not in the temp
        # folder the bundle unpacks to.
        import os
        os.chdir(Path(sys.executable).resolve().parent)
    main()
