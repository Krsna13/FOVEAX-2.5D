"""Overlap the tracker-independent part of frames across CPU cores.

For a frame N the expensive work (loading, detection + classification +
terrain features, live accuracy, grid maps / hazards / corridor profile, and the
display-only point thinning) depends only on that frame's own points, not on the
tracker. `DataStreamerThread.compute_stateless` does exactly that part. This
module runs it in a pool of worker processes for the frames AHEAD of the one the
main process is tracking and drawing, so while the main process finishes frame N
(`DataStreamerThread.finish_frame`: Kalman tracking, ROI layer, corridor
markers, FrameState assembly) and renders it, workers already compute N+1, N+2...

Nothing about the result changes:
  * the workers run the very same `compute_stateless` code as the sequential
    path (`process_frame` is `finish_frame(compute_stateless(...))`);
  * the tracker stays in the main process and is fed strictly in frame order;
  * no frame is skipped, thinned or approximated.
tests/test_frame_pipeline.py checks pooled output == sequential output.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
from concurrent.futures import Future, ProcessPoolExecutor

_ENGINE = None  # per-worker DataStreamerThread, built once by _init_worker

# Detector tunables the dashboard can change at run time (LOAD DATASET dialog);
# workers must mirror them or they would compute with different settings.
_DETECTOR_TUNABLES = ("cluster_min_points", "voxel_size_m", "ground_threshold_m")
_STREAMER_TUNABLES = ("ground_noise_m", "clearance_threshold_m", "overhead_min_points", "display_voxel_m")


def default_worker_count() -> int:
    """Auto setting: half the physical cores plus one, 1..6.

    The main (GUI) process and the Open3D process need cores too, and measured
    throughput stops improving past this point (see docs/validation_results.md).
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or (os.cpu_count() or 4) // 2
    except Exception:
        physical = (os.cpu_count() or 4) // 2
    return max(1, min(6, physical // 2 + 1))


def config_from_streamer(streamer) -> dict:
    """Snapshot everything a worker needs to rebuild an identical engine."""
    det = streamer.detector
    return {
        "source": streamer.source,
        "sequence": streamer.sequence,
        "data_root": str(streamer.data_root) if streamer.data_root else None,
        "num_frames": streamer.num_frames,
        "rate_hz": streamer.rate_hz,
        "sensor_hz": streamer.sensor_hz,
        "lidar_source": streamer.lidar_source,
        "detector": {k: getattr(det, k) for k in _DETECTOR_TUNABLES if hasattr(det, k)},
        "streamer": {k: getattr(streamer, k) for k in _STREAMER_TUNABLES if hasattr(streamer, k)},
    }


def config_signature(cfg: dict) -> str:
    """Hashable identity of a config (the lidar_source is compared by its repr/name)."""
    c = dict(cfg)
    src = c.pop("lidar_source")
    return repr(sorted(c.items(), key=lambda kv: kv[0])) + "|" + (
        f"{type(src).__name__}:{getattr(src, 'name', '')}:{len(src)}" if src is not None else "-")


def _init_worker(cfg: dict) -> None:
    global _ENGINE
    import warnings

    warnings.filterwarnings("ignore")
    try:
        import torch  # noqa: F401  -- must load before PyQt5 (Windows DLL clash, see CLAUDE.md)
    except ImportError:
        pass
    from src.dashboard.data_streamer import DataStreamerThread
    from src.perception.object_detector import MockObjectDetector

    det = MockObjectDetector()
    for k, v in cfg["detector"].items():
        setattr(det, k, v)
    eng = DataStreamerThread(
        source=cfg["source"], detector=det, num_frames=cfg["num_frames"], sequence=cfg["sequence"],
        data_root=cfg["data_root"], rate_hz=cfg["rate_hz"], predictor_mode="ground_truth",
        lidar_source=cfg["lidar_source"], sensor_hz=cfg["sensor_hz"],
    )
    for k, v in cfg["streamer"].items():
        setattr(eng, k, v)
    eng._setup_sources()
    if eng._eval_module is None:
        # Without it every frame would silently lose its live-accuracy numbers.
        raise RuntimeError("worker could not import src.10b_eval_distance_metrics")
    _ENGINE = eng


def _compute(idx: int, ts: float, ego_every_n: int | None) -> dict:
    res = _ENGINE.compute_stateless(idx, ts)
    if ego_every_n and idx > 0 and idx % ego_every_n == 0:
        # Informational ICP ego-displacement against the previous real frame,
        # moved off the main thread (it cost ~30 ms there on every Nth frame).
        try:
            from src.perception.ego_motion_estimate import estimate_relative_displacement_m

            prev = _ENGINE._get_frame_data(idx - 1)[0]
            res["ego_motion_m"] = estimate_relative_displacement_m(prev[:, :3], res["points"][:, :3])
        except Exception:
            res["ego_motion_m"] = None
    return res


class FramePipeline:
    """Prefetches compute_stateless() results for upcoming frames."""

    def __init__(self, cfg: dict, workers: int, ego_every_n: int | None = None):
        self.cfg = cfg
        self.signature = config_signature(cfg)
        self.workers = workers
        self.depth = workers + 1  # frames kept in flight ahead of the one being shown
        self.ego_every_n = ego_every_n
        self.num_frames = cfg["num_frames"]
        self._futures: dict[int, Future] = {}
        self._pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_init_worker, initargs=(cfg,),
        )

    def prefetch(self, start_idx: int, ts_fn) -> None:
        """Make sure frames start_idx .. start_idx+depth are queued or done."""
        for i in range(start_idx, min(start_idx + self.depth + 1, self.num_frames)):
            if i not in self._futures:
                self._futures[i] = self._pool.submit(_compute, i, ts_fn(i), self.ego_every_n)

    def get(self, idx: int, ts_fn) -> dict:
        """The computed result for frame idx (waits if the worker isn't done)."""
        fut = self._futures.pop(idx, None)
        if fut is None:
            fut = self._pool.submit(_compute, idx, ts_fn(idx), self.ego_every_n)
        return fut.result()

    def cancel(self) -> None:
        """Drop queued work (restart / new dataset); running tasks are discarded."""
        for f in self._futures.values():
            f.cancel()
        self._futures.clear()

    def shutdown(self) -> None:
        self.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)


class FrameFinisher:
    """Runs `finish_frame` (Kalman tracking, ROI, markers, FrameState assembly)
    on a background thread, strictly in frame order, so it overlaps the GUI
    thread's painting instead of adding to it.

    The thread is the ONLY caller of `streamer.finish_frame` (and so the only
    user of the tracker) while it runs; stop() it before touching either. It
    keeps at most `depth` finished frames waiting. The results are exactly what
    calling `finish_frame` on the GUI thread would have produced.
    """

    def __init__(self, pipeline: FramePipeline, streamer, start_idx: int, ts_fn, fps_fn, depth: int = 3):
        self._pipeline = pipeline
        self._streamer = streamer
        self._ts_fn = ts_fn
        self._fps_fn = fps_fn
        self._next = start_idx
        self._q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="frame-finisher", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        idx = self._next
        while not self._stop.is_set() and idx < self._pipeline.num_frames:
            try:
                self._pipeline.prefetch(idx, self._ts_fn)
                res = self._pipeline.get(idx, self._ts_fn)
                if self._stop.is_set():
                    return
                state = self._streamer.finish_frame(res, self._ts_fn(idx), fps=self._fps_fn())
                item = (idx, state, res.get("ego_motion_m", _NO_EGO), None)
            except BaseException as exc:  # reported to the consumer, which falls back
                item = (idx, None, _NO_EGO, exc)
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.05)
                    break
                except queue.Full:
                    continue
            if item[3] is not None:
                return
            idx += 1

    def get(self, idx: int):
        """(state, ego_motion_m or _NO_EGO) for frame idx; waits if not ready."""
        while True:
            try:
                got_idx, state, ego, exc = self._q.get(timeout=0.05)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._q.empty():
                    raise RuntimeError("frame finisher stopped before delivering frame %d" % idx)
        if exc is not None:
            raise exc
        if got_idx != idx:
            raise RuntimeError(f"frame finisher delivered frame {got_idx}, expected {idx}")
        return state, ego

    def stop(self) -> None:
        """Stop and wait; afterwards the streamer/tracker are safe to use again."""
        self._stop.set()
        self._thread.join(timeout=10.0)


_NO_EGO = object()  # "this frame carried no ego-motion estimate"
