"""Measure the REAL throughput of a running DashboardApplication.

Used by tools/bench_gui_throughput.py (source) and by `FOVEAX.exe --bench-frames N`
(packaged app), so both are measured by the same code.

The headline counts frames actually PAINTED on screen (paint events of the 2D map),
not frames merely computed: with no idle gap between frames Qt skips repaints, so
counting data updates would overstate what the user sees. The achieved rate also
depends on the pacing target the app was started with (--rate-hz); the target is
printed separately from the achieved numbers.

The first WARMUP frames are excluded (they include worker start-up).
"""

from __future__ import annotations

import math
import os
import sys
import time

WARMUP = 12


def run_gui_bench(app, frames: int) -> None:
    """Run the already-built app for *frames* new frames, print the report, exit."""
    from PyQt5.QtCore import QEvent, QObject

    wait = {"t": 0.0}
    acc = {"update_ui": 0.0}
    painted = {"n": 0}

    class PaintCounter(QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Paint:
                painted["n"] += 1
            return False

    paint_counter = PaintCounter()
    app.qt_window.map_label.installEventFilter(paint_counter)
    stamps = []  # (enter, leave, wait_total, update_total, painted_total) per newly computed frame

    pipe = getattr(app, "_pipeline", None)
    if pipe is not None:
        from src.dashboard.frame_pipeline import FrameFinisher

        orig_get = FrameFinisher.get

        def timed_get(self, idx):
            t0 = time.perf_counter()
            try:
                return orig_get(self, idx)
            finally:
                wait["t"] += time.perf_counter() - t0
        FrameFinisher.get = timed_get

    orig_update = app.qt_window.update_ui

    def timed_update(state):
        t0 = time.perf_counter()
        try:
            return orig_update(state)
        finally:
            acc["update_ui"] += time.perf_counter() - t0
    app.qt_window.update_ui = timed_update

    orig_show = app._show_frame

    def show(idx):
        fresh = idx not in app._frame_cache
        t0 = time.perf_counter()
        orig_show(idx)
        if fresh:
            stamps.append((t0, time.perf_counter(), wait["t"], acc["update_ui"], painted["n"]))
            if len(stamps) >= frames:
                app.app.quit()
    app._show_frame = show

    app.qt_window.showMaximized()
    if getattr(app, "_attempt_embed", False):
        app._try_reparent()
    app._show_frame(0)
    app.playback_timer.start()
    app.app.exec_()
    app.playback_timer.stop()
    app._stop_pipeline()
    app.o3d_queue.put("QUIT")
    app.o3d_process.join(timeout=2.0)
    if app.o3d_process.is_alive():
        app.o3d_process.terminate()

    n = len(stamps)
    k = min(WARMUP, n // 3)
    wall = stamps[-1][1] - stamps[k][1]
    ns = n - 1 - k
    inside = sum(st[1] - st[0] for st in stamps[k + 1:])
    waited = stamps[-1][2] - stamps[k][2]
    ui = stamps[-1][3] - stamps[k][3]
    shown_fps = ns / wall
    painted_fps = (stamps[-1][4] - stamps[k][4]) / wall
    fps = min(shown_fps, painted_fps)  # what is really on screen
    workers = pipe.workers if pipe is not None else 0
    print(f"\nGUI THROUGHPUT  ({n} frames, {workers} worker process(es); first {k} frames excluded)")
    print(f"  target playback rate (--rate-hz)     : {app._rate_hz:6.1f} Hz   (a cap, not a measurement)")
    print(f"  real steady-state throughput : {fps:6.1f} FPS  (frames PAINTED on screen per second)")
    print(f"    frames computed and handed to the UI : {shown_fps:6.1f} /s   ({1000 * wall / ns:.1f} ms per frame)")
    print(f"    2D-map repaints actually performed   : {painted_fps:6.1f} /s")
    print(f"  all {n} frames incl. start-up : {n / (stamps[-1][1] - stamps[0][0]):6.1f} FPS")
    print("  per frame, GUI thread (ms):")
    print(f"    waiting for the next frame  : {1000 * waited / ns:6.1f}   (0 = the GUI thread is the limit)")
    print(f"    update_ui                   : {1000 * ui / ns:6.1f}")
    print(f"    other (cache, bookkeeping)  : {1000 * (inside - waited - ui) / ns:6.1f}")
    print(f"    Qt event loop / painting    : {1000 * (wall - inside) / ns:6.1f}")
    print(f"\n  suggested --rate-hz for this machine: {math.floor(fps)}")
    sys.stdout.flush()
    os._exit(0)
