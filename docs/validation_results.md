# FOVEAX 2.5D — Validation Results

**All numbers on this page were freshly re-measured against current source
files and live runs on the date below — none are carried forward from
memory or prior conversation reports without re-verification.**

Verified: 2026-09-13, against commit `783e28f` (interpreter:
`C:\FOVEAX 2.5D\foveax_venv\Scripts\python.exe`, Python 3.11.16).
On 2026-09-30 the foveated zones moved from 0–15 / 15–35 / 35–100 m to
the problem statement's 0–10 / 10–30 / 30–100 m (cell sizes unchanged).
§1–§3 were re-measured with the new zones the same day: every overall
(0–100 m) accuracy reproduced exactly, only the per-bucket splits and
the memory figures changed. §5.1 is the per-stage profile taken on
2026-09-30; **§5.2 is the GUI throughput after the 2026-10-01 pipeline work**
(worker processes, in-place rendering). Each section states which run it
comes from.

---

## 1. Problem Statement (PS-26053) Compliance

| Requirement | Spec | Real measured value | Status |
|---|---|---|---|
| Near-zone resolution (0–10 m) | 5 cm | 5 cm (`SPEC_ZONES_100M`, `src/07_adaptive_2point5d_map.py`) | ✅ Met |
| Middle-zone resolution (10–30 m) | 20 cm | 20 cm | ✅ Met |
| Far-zone resolution (30–100 m) | 50 cm | 50 cm | ✅ Met |
| Memory reduction vs. uniform 3D voxel grid | Significant reduction | **99.99%** cell reduction, **99.81%** byte reduction (see §3) | ✅ Met |
| Real-time dashboard operation | Live playback | AC power: **about 31 frames/s computed and handed to the UI** (29.9–32.2), of which **25–28 frames/s were actually painted** where paints were counted; battery: about 22. Original GUI: 5.8–8.0 (see §5.2) | ⚠️ ~3.5–4.5x faster than before, output bit-identical; **30 FPS is reached in frames computed but not demonstrated in frames painted** |
| Semantic segmentation (in-domain) | Usable accuracy | 94.98% overall (SemanticKITTI, real SalsaNext checkpoint, see §2) | ✅ Met |
| Semantic segmentation (cross-domain, RELLIS-3D) | N/A (stretch) | 22.26% overall after sensor adaptation only, **81.53% after fine-tuning** (see §2.3) | ✅ Met after fine-tuning (held-out test sequence, never seen in training) |

## 2. Deep Learning (SalsaNext) Results

Real inference via `src/perception/salsanext_predictor.py` against the
official pretrained SalsaNext checkpoint
(`models/salsanext/pretrained/pretrained/SalsaNext`), scored with
`src/10b_eval_distance_metrics.py::compute_metrics()` (the single,
shared implementation — not reimplemented anywhere else in this
codebase; the live dashboard's accuracy card calls this exact function
too).

### 2.1 SemanticKITTI (in-domain — the checkpoint's native training domain)

Sequence `00`, frame `000000`:

| Bucket | Points evaluated | Accuracy |
|---|---:|---:|
| Near (0–10 m) | 62,325 | **97.70%** |
| Mid (10–30 m) | 52,916 | **92.82%** |
| Far (30–100 m) | 7,239 | **87.28%** |
| **Overall (0–100 m)** | **122,480** | **94.98%** |

### 2.2 RELLIS-3D (cross-domain — off-road, different LiDAR: Ouster OS1-64 vs. training-domain Velodyne HDL-64E)

Sequence `00000`, frame `000000`:

| Bucket | Points evaluated | Accuracy, no adaptation | Accuracy, with sensor adaptation* |
|---|---:|---:|---:|
| Near (0–10 m) | 103,534 | 3.78% | **15.07%** |
| Mid (10–30 m) | 26,810 | 17.36% | **49.57%** |
| Far (30–100 m) | 726 | 20.25% | **38.43%** |
| **Overall (0–100 m)** | **131,070** | **6.65%** | **22.26%** |

\* Sensor adaptation = real Ouster OS1-64 vertical FOV override
(`fov_up=17.02°, fov_down=-16.44°`, vs. the checkpoint's assumed Velodyne
HDL-64E FOV of `-25°..+3°`) plus intensity rescaling to `[0,1]` (RELLIS-3D
intensities are ~0.0001–0.01, the training domain's are ~0–1). Both real,
documented, measured adaptations — not a guess. Sensor adaptation alone
(22.26%) is a real, honest domain-transfer failure, not a usable
production number on its own — see §2.3 for the fine-tuned result that
resolves this.

### 2.3 RELLIS-3D fine-tuned (Stage 6 — SalsaNext fine-tuned directly on real RELLIS-3D data)

Real fine-tuning run (`src/14_finetune_salsanext.py`), starting from the
same official SemanticKITTI-pretrained checkpoint used in §2.1/§2.2, with
its 20-class head replaced by an 8-class FOVEAX head (see
`src/perception/salsanext_predictor.py`'s `taxonomy="foveax"` path) and
fine-tuned on real RELLIS-3D data using the same real Ouster OS1-64 sensor
adaptation as §2.2. Full detail (split, config, per-epoch trajectory,
forgetting check) in the Stage 6 report; summarized here.

**Split** (deterministic, `src/training/rellis3d_dataset.py::build_splits`):
sequence `00000` held out **entirely** as test (2,847 frames, never seen
during training); train/val drawn from sequences `00001`-`00004` with a
15% contiguous tail per sequence as val and a 10-frame gap between train
and val to prevent near-duplicate adjacent-frame leakage. Real counts:
train 4,532 frames (stride 2), val 162 frames (stride 10), test 2,847
frames (full sequence, stride 20 for evaluation = 100 frames actually
scored below).

**Config**: 10 epochs, batch size 2, SGD lr=0.01 (upstream SalsaNext's own
warmup+decay schedule), class-weighted `NLLLoss` (`w = 1/(content +
0.001)`, computed on the real training split's own class frequencies) +
`Lovasz_softmax`, both reused unmodified from the official SalsaNext repo.
Real per-epoch validation accuracy (held-out val split, 8 classes):

| Epoch | Train loss | Val accuracy |
|---:|---:|---:|
| 0 | 0.8431 | 84.99% |
| 1 | 0.5350 | 86.55% |
| 2 | 0.4908 | 86.30% |
| 3 | 0.4109 | 87.77% |
| 4 | 0.3913 | 88.87% |
| 5 | 0.3659 | 87.76% |
| 6 | 0.3806 | 87.52% |
| 7 | 0.4147 | 88.41% |
| 8 | 0.3744 | 88.47% |
| **9 (best)** | **0.3220** | **89.05%** |

Real training time: 101 minutes total training compute (sum of real
per-epoch `train_seconds`; wall-clock spanned longer across this session
due to two real interruptions — see the Stage 6 report — both resumed
cleanly from checkpoint, no lost epochs). Peak real GPU memory: 3,767 MB
(RTX 5050 Laptop, 8,151 MB total) at batch size 2.

**Held-out test-sequence result** (sequence `00000`, 100 frames, stride
20, real Ouster sensor adaptation applied, scored by the same
`compute_metrics()` used everywhere else on this page):

| Bucket | Points evaluated | Accuracy |
|---|---:|---:|
| Near (0–10 m) | 7,103,965 | 78.83% |
| Mid (10–30 m) | 2,522,815 | 89.42% |
| Far (30–100 m) | 61,051 | 69.85% |
| **Overall (0–100 m)** | **9,687,831** | **81.53%** |

Real progression on this held-out sequence: **6.65%** (pretrained,
no adaptation) → **22.26%** (pretrained + sensor adaptation) →
**81.53%** (fine-tuned on RELLIS-3D). The Far bucket's lower accuracy
(69.85%, only 61,051 points) reflects real sparse far-range coverage on
this off-road LiDAR, not a training defect.

**SemanticKITTI forgetting check**: because the fine-tuned head emits
FOVEAX classes directly (not SemanticKITTI's 20), its accuracy is
evaluated against SemanticKITTI ground truth *mapped into the same FOVEAX
space* the original pretrained checkpoint is scored in throughout this
page (frame `00/000000`, `--taxonomy foveax`):

| | Overall FOVEAX-space accuracy |
|---|---:|
| Original pretrained checkpoint (§2.1) | 94.98% |
| **Fine-tuned on RELLIS-3D** | **23.97%** |

This is real, substantial forgetting, not a near-zero collapse — expected
given 10 epochs of RELLIS-3D-only fine-tuning with no SemanticKITTI replay
or mixing. Flagged honestly as a known limitation (§6) rather than
omitted: this checkpoint should be used for off-road RELLIS-3D-like
scenes, not as a drop-in replacement for the original urban-domain
checkpoint.

**Checkpoints**: original SemanticKITTI checkpoint preserved unchanged at
`models/salsanext/pretrained/pretrained/SalsaNext` (confirmed via MD5,
matches its pre-fine-tuning hash). New fine-tuned checkpoints at
`models/salsanext/rellis3d_finetuned/` (`epoch_000.pt`...`epoch_009.pt`,
`last.pt`, `best.pt` = epoch 9).

## 3. Grid Engine Efficiency (Phase B)

Computed live via
`src/07_adaptive_2point5d_map.py::compute_memory_reduction_metrics()`
against the real `SPEC_ZONES_100M` config (Near 0–10m/5cm, Middle
10–30m/20cm, Far 30–100m/50cm) — a fixed architectural property of the
current zone config, not a per-frame measurement:

| Metric | Value |
|---|---:|
| Finest resolution | 0.05 m |
| Uniform 3D voxel grid (same finest resolution, 8 m height) | 2,560,000,000 voxels / 2,441.4 MB |
| Uniform 2.5D grid (same finest resolution) | 16,000,000 cells / 244.1 MB |
| Adaptive foveated 2.5D grid (actual FOVEAX design) | 302,850 cells / 4.6 MB |
| **Cell reduction vs. uniform 3D voxels** | **99.99%** |
| **Byte reduction vs. uniform 3D voxels** | **99.81%** |
| Cell reduction vs. uniform 2.5D grid | 98.11% |

## 4. Tracking / Detection Results

- Clustering: `MockObjectDetector` (deterministic geometric baseline, not
  a trained detector) using vectorized `cKDTree.query_pairs` +
  `scipy.sparse.csgraph.connected_components` (replaced a per-point BFS
  loop; verified bit-identical clustering output against the original
  BFS on real RELLIS-3D frames before/after).
- Tracking: constant-velocity Kalman filter + Hungarian assignment
  (`src/tracking/multi_object_tracker.py`).
- **Known, fixed false-positive class**: "Dynamic" classification is
  gated by real-world class semantics — classes that cannot physically
  move (VEGETATION, BUILDING_WALL, SOLID_OBSTACLE, DRIVABLE_GROUND,
  ROUGH_TERRAIN) can never be flagged Dynamic, regardless of measured
  speed. Root cause: this pipeline has no ego-motion compensation
  (no odometry), so a real, long-duration vehicle motion (translation
  and/or rotation) makes every real static object appear to drift
  through the ego-relative sensor frame. Verified over a real 200-frame
  RELLIS-3D run (sequence `00002`): **0** Dynamic-VEGETATION events
  across the full run (down from 12 tracks falsely flagged before the
  fix), while genuinely mobile classes still correctly reported Dynamic
  (VEHICLE: 176 events / 10 tracks; PEDESTRIAN: 185 events / 6 tracks).

## 5. Dashboard Performance

**Current GUI numbers are in §5.2.** §5.1 is the single-threaded per-stage
profile from before that work (128.4 ms/frame), and the headless replay
below is older still, taken on commit `783e28f` before the per-object
terrain features, 3D labels and overhead/corridor panels were added, so
neither describes today's GUI:

Real headless replay, RELLIS-3D sequence `00001`, 30 real frames
(`src/13_realtime_dashboard.py --headless`), 2026-09-13:

| Metric | Value |
|---|---:|
| Steady-state per-frame latency (frames 5–30) | 60.1–64.8 ms |
| Steady-state FPS (from latency) | ~15.5–16.6 FPS |
| Overall measured FPS (incl. one-time first-frame warmup) | **13.8 FPS** |
| Real points per frame | 131,072 |

This reflects the current state after two rounds of real profiling-driven
optimization (clustering vectorization + overhead-grid vectorization),
which together took real per-frame processing time from ~214 ms down to
~62–65 ms on this same real data (a ~3.3x improvement). See §6 for the
honest gap to a 25–30 FPS target.

### 5.1 Single-threaded per-stage profile, before the pipeline work (`--profile`, 2026-09-30)

`python src/13_realtime_dashboard.py --profile --profile-warmup 20
--profile-frames 200 --source rellis3d --sequence 00001` (default
`--predictor ground_truth`, so no SalsaNext inference). This harness
drives the same objects as the live GUI, frame by frame, with the Qt
window and Open3D viewer real but off-screen and **in-process** (see
`src/dashboard/stage_timer.py`). 200 measured frames:

| Stage | mean ms | p95 ms | % of frame |
|---|---:|---:|---:|
| load (RELLIS-3D `.bin` + labels) | 9.4 | 14.7 | 2.4% |
| live_accuracy_metrics | 4.3 | 6.8 | 1.1% |
| clustering_heuristics (detect + classify + terrain features) | 84.3 | 109.1 | 21.4% |
| kalman_tracking | 1.4 | 2.2 | 0.4% |
| grid_maps_hazards | 23.5 | 30.4 | 6.0% |
| metrics_state_logging | 5.4 | 7.6 | 1.4% |
| ego_motion_icp (every 5th frame in `--profile`) | 29.3 | 40.3 | 1.5% |
| open3d_update (in-process) | 184.5 | 221.9 | 46.9% |
| qt_ui_update + qt_event_loop_paint | 74.4 | — | 18.9% |
| **frame_total** | **393.8** | **479.2** | 100% |

- **`process_frame()` alone: 128.4 ms mean (p95 163.0 ms), ≈7.8 FPS
  ceiling for the pipeline itself.** The dashboard's on-screen latency
  (`state.metrics.latency_ms`) averaged 122.9 ms in the same run; it
  measures `process_frame()` only and excludes drawing.
- The in-process `open3d_update` cost is a profiling artifact: the live
  GUI runs Open3D in a separate process, so it is not added to the Qt
  thread's per-frame time.
- **The on-screen FPS figure is the target rate, not a measurement.**
  `state.metrics.fps` carried exactly `30.0` (the `--rate-hz` default)
  on every measured frame. Do not quote the dashboard's FPS label as
  achieved throughput.
- GPU utilisation during this run averaged 1.2% (no model inference).
  An earlier 2026-09-20 run with `--predictor salsanext` (pretrained
  checkpoint, before the dashboard performance changes) measured
  `salsanext_forward` at 42.4 ms and `range_image_projection` at 19.5 ms
  per frame (`outputs/phase12/profile_run_raw.txt`).

### 5.2 Real GUI throughput after the pipeline work (2026-10-01)

**Result, stated precisely** (a frame can be *computed and handed to the UI*
without being *painted*; Qt skips repaints when there is no idle gap, see
"Computed versus painted" below):

| | Original GUI | New GUI |
|---|---:|---:|
| Frames computed and handed to the UI, AC power | 5.8–8.0 /s | **29.9–32.2 /s** (about 31) when the laptop ran fast; 28.9–30.3 /s in a slower window |
| Frames actually painted (map repaints counted), AC power | not counted (cannot exceed the row above) | **25.2–28.3 /s** (free-running 28.3; paced at 25.5–25.9) |
| Frames computed, battery power | 5.2–6.7 /s | 17.9–24.7 /s (median about 22) |

**So the GUI is about 3.5–4.5x faster than before with identical output. The
30 FPS goal is reached in frames computed, but 30 painted frames per second
was not demonstrated on this laptop: the best painted rate measured is 28.3.**
(An earlier version of this section called the 30 FPS goal reached; that
rested on counting computed frames, which overstates what is on screen by
about 10%.)

How it was measured: `tools/bench_gui_throughput.py` (and the equivalent
scratch harness used for the comparisons) runs the **real** dashboard
(PyQt5 window, embedded Open3D process, worker pool) on RELLIS-3D sequence
`00001`, 131k points/frame, stops after 100 frames, and reports wall-clock
frames per second excluding the first 12 frames (worker start-up, about
10 s). Old and new code were run **alternately, back to back**, so both saw
the same machine state.

AC power (Windows "Balanced" plan, plugged in, battery 26–48%; processor
performance counter 72–141% as turbo came and went), steady-state frames
computed per second (this table counts computed frames, not painted ones):

| Run | Original GUI | New GUI | Workers |
|---|---:|---:|---:|
| 1 | 7.0 | 28.5 | 6 (auto) |
| 2 | 7.7 | 28.9 | 6 (auto) |
| with tracking on a background thread: 3 | 7.2 | 30.0 | 6 |
| 4 | 5.8 | 31.5 | 6 |
| 5 | 8.0 | 29.9 | 6 |
| worker-count test | — | 31.3 / 31.2 | 6 |
| worker-count test | — | 30.8 / 30.9 | 8 |
| worker-count test | — | 31.2 | 10 |
| `tools/bench_gui_throughput.py` | — | 32.2 / 31.7 | 6 |

Runs 1–2 are before tracking moved to a background thread (28.5–28.9 FPS);
everything after it is 29.9–32.2, about 31 on average. More than 6 workers does
not help once the GUI thread is the limit.

Battery power (same code, charge falling from 66% to 27% while unplugged):

| Run (battery %) | Original GUI | New GUI | Workers |
|---|---:|---:|---:|
| 57% | 5.22 FPS | 19.61 FPS | 4 |
| 57% | 5.47 FPS | 17.89 FPS | 4 |
| 42% | 6.69 FPS | 21.86 FPS | 4 |
| 42% | — | 24.38 FPS | 6 |
| 42% | 6.32 FPS | 21.27 FPS | 4 |
| 34% | 5.2 FPS | 21.8 FPS | 6 |
| 34% | 5.6 FPS | 24.7 FPS | 6 |

**Computed versus painted.** Counting `QEvent.Paint` events on the 2D map and
the corridor chart showed that with no wait between frames (`--rate-hz 1000`)
the GUI computed 28.3–30.3 frames per second but repainted the map only
25.5–28.3 times per second, because Qt coalesces repaints when its event loop
never goes idle. Paced by a cap (`--rate-hz 30`), every computed frame was
painted (25.2–25.9 per second) but the free-running mode was still faster
(28.3 painted versus 25.2 in the same window), probably because idle gaps let
cores clock down. `tools/bench_gui_throughput.py` therefore reports the painted
rate as its headline number. Tried and rejected: Qt's precise timer (no change)
and batching repaint requests with `setUpdatesEnabled` (no gain per label
batch, clearly worse when applied to the whole window).

**The power state decides the answer.** On battery the CPU is power-limited and
its speed fell with the charge: the same single-process benchmark took 84.5 ms
per frame early on and 167 ms at 27% (Windows reported 64% processor
performance), and the original GUI measured 2.0 FPS at the worst moment. On AC
(after the laptop cooled) a worker frame took 84.5–91.5 ms again. Always
compare old and new alternately in one session, and measure on AC power with
`python tools/bench_gui_throughput.py`.

What limits it now (AC power, steady state, 6 workers, GUI thread per frame,
`tools/bench_gui_throughput.py`, laptop at about 80% processor performance):

| Per-frame GUI-thread time | Free-running | Paced at 30 Hz |
|---|---:|---:|
| Waiting for the next finished frame | 2.7 ms | 0.0 ms |
| `update_ui` | 5.5 ms | 6.7 ms |
| Qt event loop and painting | 24.7 ms | 32.9 ms |
| **Frame time** | **33.0 ms (30.3 computed, 28.3 painted per second)** | **39.7 ms (25.2 per second)** |

Earlier, with the laptop faster (processor performance up to 141%), the same
free-running configuration computed 31.7–32.2 frames per second (about 31 ms
per frame; waiting 3.7 ms, `update_ui` 4.5–5.0 ms, Qt 22.6–22.9 ms); paints
were not counted in those runs.

Further measurements: the worker pool alone (no GUI) delivered 7.2 / 10.7 /
21.9 / 28.5 / 33.6 / 37.1 FPS with 1 / 2 / 3 / 4 / 6 / 8 workers (on battery),
so workers are not the limit in isolation. On battery, with the track labels,
the Open3D frame push and the UI update all switched off, the GUI was still
limited to 28.3 FPS: the GUI thread and the Open3D process compete with the
workers for one limited CPU budget (about 4 cores' worth). The Open3D push cost
about 12 ms of GUI-thread time per frame, the native label windows about 12 ms,
and embedding the 3D window about 5 ms.

What changed:

- **Rendering, in place:** the corridor chart is drawn with QPainter instead of
  rebuilding a matplotlib figure (47 ms per frame in a profile); tracked-object
  list, hazard table and every label write only what changed; track labels reuse
  their native windows and only resize when the rounded width changes; the 2D
  map layers use a precomputed colour table (pixel-identical, tested);
  the session log stays open.
- **Playback timer:** a periodic `QTimer` waited a whole extra interval after
  every frame that overran its budget, leaving a third of each frame idle. It
  is now re-armed for only the time remaining. No frame is skipped.
- **Tracking on a background thread:** `FrameFinisher` runs `finish_frame` on a
  thread, in frame order, so tracking overlaps Qt's C++ painting instead of
  adding to the GUI thread (28.5–28.9 -> about 31 frames computed per second on AC). This exposed an
  older latent bug: `MultiObjectTracker.update()` returns the tracker's own live
  `TrackState` objects, which later frames update in place, so a `FrameState`
  (including every frame cached for STEP back) changed retroactively. Each
  frame now keeps a snapshot of its tracks (`tests/test_frame_pipeline.py`).
- **Parallel frames:** `DataStreamerThread.process_frame` is split into
  `compute_stateless` (load, detection, terrain features, accuracy, grids,
  hazards, corridor profile; independent of the tracker) and `finish_frame`
  (Kalman tracking in strict frame order, ROI layer, corridor markers). Worker
  processes (`src/dashboard/frame_pipeline.py`) run `compute_stateless` for the
  next few frames while the main process finishes and draws the current one.
  `--workers` sets the count (auto: half the physical cores plus one, at most 6;
  0 = single-threaded).
- **Detector:** the cluster graph is converted to CSR with scipy's own kernel,
  skipping duplicate-summing and index sorting it never needed. Clusters are
  identical; a worker frame went from 98 to 84.5 ms.
- **Display only:** the Open3D view gets a voxel-thinned copy of the cloud
  (0.15 m, about 29k points) and only the fields it draws; the footer shows
  "3D view display: N pts (thinned; all metrics use all points)". Every
  perception stage and metric still uses all points. `--display-voxel 0` sends
  the full cloud.
- **Labelled cadence:** the ICP ego-displacement estimate is recomputed every
  30th frame and the panel now says which frame it is from and how often it is
  recomputed. No other stage is throttled.
- **Removed:** the earlier "OPT-n" shortcuts (grid built from a thinned cloud,
  terrain features skipped for far static objects, hazard and corridor profile
  recomputed only every 2nd/3rd frame, reused grid buffers, lossy redraw
  caches). They changed the output: against the pre-shortcut reference, **all
  40 of 40 test frames differed** (1,134 of 6,400 elevation cells in frame 0,
  36 vs 41 hazard clusters, different road width, uncertainty and track
  metadata).

Exactness: for 40 real frames of sequence `00001`, every perception output
(all five grid layers, hazard clusters, tracks and their metadata, classes,
elevation and corridor profiles, road width, uncertainty, live accuracy) of the
new single-threaded path **and** of the 3-worker path equals the output of the
code before the shortcuts (commit `571c976`) bit for bit, NaN-aware.
`tests/test_frame_pipeline.py` keeps pooled == sequential in the test suite.

Headless: the single-threaded `--headless` path was not re-measured cleanly
(the throttled laptop varied 135–529 ms for identical code within minutes, so
no honest figure exists for this session). From measurements taken in stable
windows, one worker frame is 84.5 ms and finishing it about 7 ms, so about
10–11 FPS single-threaded; treat that as an estimate, not a measurement.

What was not done:

- **GPU step skipped on purpose.** The grid binning that could move to CUDA is
  about 2 ms of a worker frame and the overhead-clearance grid about 9 ms. That
  saves at most 8% of worker time at the cost of a CUDA context per worker, and
  it was not judged worth it (30 FPS is reached in frames computed, not in
  frames painted — see §5.2).
- **Headroom, if more is wanted (not applied):** the GUI thread is now the
  limit, with about 25–33 ms of Qt painting per frame. Options: draw the track
  labels inside one overlay instead of about 20 native windows (up to about
  12 ms; needs a layered-window design); render the Open3D view off-screen and
  show it as an image instead of embedding a foreign window (removes the
  cross-process window sync, adds image transfer); run object detection every
  2nd frame with Kalman prediction in between (saves about a quarter of worker
  time, **would change detections** so it needs an explicit "updated every 2nd
  frame" UI label and an accuracy comparison first).

**Do not quote the "145–171 FPS" MockObjectDetector figure as
representative of real dashboard performance** — that number was
measured on the ~2,150-point synthetic `sample` source, not a real
~131,000-point RELLIS-3D/SemanticKITTI scan; it does not reflect this
project's real-data throughput.

## 6. Known Limitations

1. **No ego-motion compensation.** This pipeline has no odometry/SLAM.
   Over a long real playback, genuine vehicle motion makes static real
   objects appear to move in the ego-relative frame. Mitigated for
   physically-static classes (see §4) but not solved for genuinely
   mobile classes (VEHICLE/PEDESTRIAN velocity readings may still be
   partially contaminated by uncompensated ego motion).
2. **The fine-tuned RELLIS-3D checkpoint (81.53% held-out overall, §2.3)
   substantially forgets the original SemanticKITTI domain (94.98% →
   23.97% overall, FOVEAX-space).** Real, measured, not a near-zero
   collapse, but a real tradeoff: this checkpoint is not a drop-in
   replacement for the original — use `models/salsanext/pretrained/`
   for urban/SemanticKITTI-like scenes and
   `models/salsanext/rellis3d_finetuned/` for off-road/RELLIS-3D-like
   scenes. Resolving this would need replaying real SemanticKITTI
   samples during fine-tuning (not done here).
3. **Dashboard throughput is 3.5–4.5x the original but still short of 30
   painted FPS on the development laptop.** On AC power the GUI computes about
   31 frames per second but paints 25–28 of them (about 22 computed on battery),
   up from 5.2–8.0 for the original GUI (§5.2). Other machines need
   `tools/bench_gui_throughput.py` to find their rate. The GUI thread's Qt
   painting (about 25–33 ms per frame) is the largest single cost.
4. **`MockObjectDetector` is a deterministic geometric baseline, not a
   trained AI detector** — it clusters points above a ground-height
   threshold; it does not perform learned object recognition. Real
   semantic classification (VEGETATION/VEHICLE/PEDESTRIAN/etc.) only
   applies when real ground-truth or SalsaNext labels are cross-
   referenced against a cluster.

## 7. Test Suite Status

Fresh run, re-confirmed 2026-09-13 (post Stage 6 SalsaNext RELLIS-3D
fine-tuning + terrain-feature/distance-display dashboard work -- 138 new
regression tests added since this doc's original 352-test baseline):

```
490 passed, 33 warnings in 8.60s
```

No skipped tests, no failures, no regressions against any prior baseline
recorded during this project's development.

**Fresh run, 2026-09-30** (the committed code, with the 0–10 / 10–30 /
30–100 m zones, the packaging changes and the new
`tests/test_alert_card_placement.py`; SemanticKITTI data and the SalsaNext
checkpoints present):

```
505 passed, 32 warnings in 5.46s
```

**Fresh run, 2026-10-01** (after the pipeline work; includes the new
input-layer tests, pooled-vs-sequential tests, pixel-equivalence tests for
the 2D map renderers, and detector-graph tests):

```
569 passed in about 20 s
```

(One run on the throttled laptop took 74 s; the count was the same.) The
2026-09-30 note about two failing tests no longer applies: the "OPT-n"
shortcuts that caused them were removed in this work.
