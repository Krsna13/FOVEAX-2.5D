import dataclasses
import math
import time
from contextlib import nullcontext
import numpy as np
import psutil
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

import os
from pathlib import Path

from typing import Any
from PyQt5.QtCore import QThread, pyqtSignal

from src.dashboard.dashboard_state import FrameState, HardwareMetrics, assert_coordinate_frame_consistency
from src.tracking.multi_object_tracker import MultiObjectTracker
from src.perception.object_detector import ObjectDetector, MockObjectDetector
import importlib
_trk = importlib.import_module('src.11_object_detection_tracking')
generate_synthetic_sample_cloud = _trk.generate_synthetic_sample_cloud
repeated_cloud_with_translation = _trk.repeated_cloud_with_translation

try:
    from src.perception.rellis3d_loader import load_rellis3d_frame
    HAS_RELLIS_LOADER = True
except ImportError:
    HAS_RELLIS_LOADER = False

def _snapshot_track(t):
    """An independent copy of a TrackState.

    MultiObjectTracker.update() returns a new list but the SAME TrackState
    objects, which it keeps updating in place on later frames. A FrameState
    holding them would change retroactively (a frame cached for STEP back would
    show later positions, and a reader on another thread would see an object
    mid-update). Each frame therefore keeps its own copy.
    """
    return dataclasses.replace(
        t, state=t.state.copy(), covariance=t.covariance.copy(),
        size_lwh=t.size_lwh.copy(), metadata=dict(t.metadata))


class DataStreamerThread(QThread):
    state_ready = pyqtSignal(FrameState)
    
    def __init__(
        self,
        source: str = "sample",
        detector: ObjectDetector = None,
        num_frames: int = 100,
        sequence: str = "00",
        data_root: str | Path | None = None,
        rate_hz: float = 10.0,
        semantic_predictor: Any = None,
        predictor_mode: str = "ground_truth",
        lidar_source: Any = None,
        sensor_hz: float | None = None,
    ):
        super().__init__()
        # Optional src.inputs.LidarSource (any dataset/sensor via a config file).
        # When set it replaces the built-in sample/semantickitti/rellis3d readers.
        self.lidar_source = lidar_source
        self.source = "custom" if lidar_source is not None else source.lower()
        self.detector = detector if detector else MockObjectDetector()
        self.semantic_predictor = semantic_predictor
        # "ground_truth" | "mock" | "salsanext" -- matches
        # src/dashboard/export_web_dashboard_data.py's --predictor choices.
        # Only used to decide what the live accuracy comparison's
        # "prediction" side is; semantic_predictor itself is what actually
        # produces predictions when this is not "ground_truth".
        self.predictor_mode = predictor_mode
        self.tracker = MultiObjectTracker()
        self.num_frames = num_frames
        self.sequence = sequence
        self.data_root = Path(data_root) if data_root else None
        self.rate_hz = rate_hz
        # Scan rate of the SENSOR that recorded the data. Timestamps (and so the
        # tracker's m/s speeds) come from this, not from rate_hz, which is only
        # how fast the replay is played back.
        self.sensor_hz = float(sensor_hz if sensor_hz is not None
                               else (lidar_source.sensor_hz if lidar_source is not None else 10.0))
        self.running = True
        self.ground_noise_m: float = 0.30
        self.clearance_threshold_m: float = 2.0
        self.overhead_min_points: int = 4
        # Voxel size (m) of the display-only thinned cloud sent to the 3D view; 0 disables.
        self.display_voxel_m: float = 0.15
        
        # Grid parameters for 2D maps
        self.grid_res = 0.5
        self.grid_extent = (-20.0, 20.0, -20.0, 20.0) # x_min, x_max, y_min, y_max
        self.x_bins = int((self.grid_extent[1] - self.grid_extent[0]) / self.grid_res)
        self.y_bins = int((self.grid_extent[3] - self.grid_extent[2]) / self.grid_res)

        # Cache the accuracy-metrics module once (importing it per frame is wasted work).
        try:
            self._eval_module = importlib.import_module("src.10b_eval_distance_metrics")
        except Exception:
            self._eval_module = None

        # Hardware telemetry is sampled at most every 100 ms (psutil/pynvml are
        # system calls); each frame still gets its own HardwareMetrics object.
        self._last_hw_time: float = 0.0
        self._last_hw_snapshot: HardwareMetrics | None = None
        self._last_extras: dict = {}
        self._log_fh = None
        self._log_writes = 0

        global HAS_PYNVML
        
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                HAS_PYNVML = False

    def reset_caches(self):
        """Forget per-run state so nothing bleeds into a new sequence.

        Every perception stage is now recomputed from scratch for each frame
        (no throttled/cached stage), so only the last-extras scratch remains.
        """
        self._last_extras = {}

    def _attach_terrain_features(self, detections: list, points: np.ndarray) -> None:
        """Compute real geometric features per detection and stash them in
        `det.metadata`, so the tracker (and, from there, the display layer)
        can surface them without recomputing or guessing.

        Every feature here is derived from this detection's own real member
        points (selected from the real frame point cloud, including the
        ground points MockObjectDetector itself discards before clustering
        -- needed here to estimate real local ground height) via
        src/perception/terrain_features.py. A feature is simply absent from
        metadata when there isn't enough real data to support it (e.g. a
        one-point detection can't support a plane fit) -- never a
        placeholder value.

        Uses the plain per-detection brute-force lookups
        (points_in_oriented_box / local_ground_z_percentile), each an O(n)
        scan of the full frame. Two alternatives were tried and REJECTED
        after clean, repeated real profiling on 131k-point RELLIS-3D frames
        with ~15-20 detections/frame -- both were slower, not faster:
          - A per-detection cKDTree (points_in_oriented_box_via_tree /
            local_ground_z_percentile_via_tree): ~90ms/frame. Building a
            fresh tree over 131k points every frame costs more than it
            saves across only ~15-20 queries -- not enough queries to
            amortize tree construction.
          - Broadcasting all detections against the frame at once
            (points_in_oriented_boxes_batch / local_ground_z_percentiles_batch):
            ~70ms/frame. Allocating (131072, N)-shaped intermediate arrays
            every frame outweighs the saved per-detection Python overhead
            at this N.
        The plain brute-force loop measured ~17ms/frame, cheapest of the
        three -- kept as the real implementation. All three give
        bit-identical results (verified on real data); the batch/tree
        variants remain in terrain_features.py, tested, for a workload
        where n_detections is large enough to change this trade-off.
        """
        from src.perception.terrain_features import (
            ground_clearance_m,
            is_overhang,
            is_probable_rock_heuristic,
            local_ground_z_percentile,
            points_in_oriented_box,
            slope_angle_deg,
        )

        ground_like = {"DRIVABLE_GROUND", "ROUGH_TERRAIN"}

        for d in detections:
            member_mask = points_in_oriented_box(points, d.center_xyz, d.size_lwh)
            member_xyz = points[member_mask, :3]
            metadata: dict = {}

            if d.class_name in ground_like:
                slope = slope_angle_deg(member_xyz)
                if slope is not None:
                    metadata["slope_deg"] = slope

            ground_z = local_ground_z_percentile(points, d.center_xyz[:2], radius_m=1.5)
            if ground_z is not None:
                bbox_bottom_z = float(d.center_xyz[2] - d.size_lwh[2] / 2.0)
                clearance = ground_clearance_m(bbox_bottom_z, ground_z)
                metadata["ground_clearance_m"] = clearance
                metadata["is_overhang"] = is_overhang(clearance, d.class_name)

            volume_m3 = float(d.size_lwh[0] * d.size_lwh[1] * d.size_lwh[2])
            metadata["is_probable_rock"] = is_probable_rock_heuristic(
                volume_m3, int(member_mask.sum()), d.class_name
            )

            d.metadata.update(metadata)

    # ------------------------------------------------------------------
    # 2.5D grid maps, split in two so the expensive, tracker-independent part
    # can run in a worker process (src/dashboard/frame_pipeline.py) while the
    # cheap part that needs this frame's tracks runs in the main process:
    #   _grid_base(points)        elevation / traversability / uncertainty /
    #                             overhead grids, road width, elevation
    #                             profile, hazard clusters, corridor profile
    #   _finish_grids(base, tracks)  ROI layer + corridor object markers
    # _generate_maps(points, tracks) runs both (the original entry point).
    # ------------------------------------------------------------------
    def _generate_maps(self, points: np.ndarray, tracks: list) -> dict:
        """Generate the 2D grid maps for dashboard visualization."""
        return self._finish_grids(self._grid_base(points), tracks)

    def _display_downsample(self, points: np.ndarray):
        """Voxel-thinned copy of the cloud for the 3D VIEW ONLY (see
        FrameState.display_points). Perception never sees it."""
        try:
            voxel = float(self.display_voxel_m)
        except (AttributeError, RuntimeError):
            voxel = 0.15
        if voxel <= 0.0 or len(points) == 0:
            return None
        try:
            import open3d as o3d

            # Empty (0, 0, 0) "no return" padding is not a real point, so it is
            # left out of the picture (the full cloud is untouched elsewhere).
            real = points[(points[:, :3] != 0).any(axis=1), :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(real.astype(np.float64))
            xyz = np.asarray(pcd.voxel_down_sample(voxel).points, dtype=np.float32)
        except ImportError:  # keep the first point of every occupied voxel
            keys = np.floor(points[:, :3] / voxel).astype(np.int64)
            _, first = np.unique(keys, axis=0, return_index=True)
            xyz = points[np.sort(first), :3].astype(np.float32)
        return np.column_stack([xyz, np.zeros(len(xyz), dtype=np.float32)])

    def _overhead_params(self) -> dict:
        """Overhead-detection thresholds with defaults. Reading attributes of a
        Qt object whose constructor never ran raises RuntimeError (not
        AttributeError), which tests that build a bare streamer rely on us
        absorbing."""
        out = {}
        for kwarg, attr, default in (("min_points", "overhead_min_points", 4),
                                     ("clearance_threshold_m", "clearance_threshold_m", 2.0),
                                     ("ground_noise_m", "ground_noise_m", 0.30)):
            try:
                out[kwarg] = getattr(self, attr)
            except (AttributeError, RuntimeError):
                out[kwarg] = default
        return out

    def _grid_base(self, points: np.ndarray) -> dict:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x_min, x_max, y_min, y_max = self.grid_extent

        # Filter points within bounds
        mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) == 0:
            return {
                "empty": True,
                "empty_grid": np.full((self.y_bins, self.x_bins), np.nan, dtype=np.float32),
                "empty_overhead": np.full((self.y_bins, self.x_bins), -1, dtype=np.int8),
            }

        col = np.floor((x - x_min) / self.grid_res).astype(np.int32)
        row = np.floor((y - y_min) / self.grid_res).astype(np.int32)

        # Need to flip Y axis for image rendering (0,0 is top-left in image, bottom-left in world)
        # However, matplotlib imshow handles origin='lower'. For PyQt QImage, (0,0) is top-left.
        # Let's just create standard arrays and let UI handle it.
        flat_idx = row * self.x_bins + col

        total_cells = self.y_bins * self.x_bins
        # Fresh arrays every frame: a reused buffer would be overwritten by the
        # next frame while earlier FrameStates (cached for STEP back, or in
        # flight to a display process) still point into it.
        z_max_arr = np.full(total_cells, -np.inf, dtype=np.float32)
        np.maximum.at(z_max_arr, flat_idx, z)
        z_max_arr[z_max_arr == -np.inf] = np.nan

        elevation = z_max_arr.reshape(self.y_bins, self.x_bins)

        # Traversability mock: High if elevation variance is high
        z_min_arr = np.full(total_cells, np.inf, dtype=np.float32)
        np.minimum.at(z_min_arr, flat_idx, z)
        z_min_arr[z_min_arr == np.inf] = np.nan
        z_min_arr = z_min_arr.reshape(self.y_bins, self.x_bins)

        height_range = elevation - z_min_arr
        # Traversability: 1.0 = safe (flat), 0.0 = blocked (rough/obstacle),
        # matching canonical convention from 06_terrain_traversability.py
        traversability = 1.0 - np.clip(height_range / 1.0, 0.0, 1.0)

        # Point count per cell (real occupancy density) -- used below for
        # the real Phase 5 (src/08_spatial_importance_roi.py) uncertainty
        # formula, applied verbatim to this live local grid.
        point_count = np.bincount(flat_idx, minlength=total_cells).astype(np.float32)
        occupied = point_count > 0
        density_confidence = np.clip(np.log1p(point_count) / np.log(8.0), 0.0, 1.0)
        uncertainty_flat = 1.0 - density_confidence
        uncertainty_flat[~occupied] = 1.0
        uncertainty = uncertainty_flat.reshape(self.y_bins, self.x_bins)
        point_count_2d = point_count.reshape(self.y_bins, self.x_bins)
        occupied_2d = occupied.reshape(self.y_bins, self.x_bins)

        extras = self._compute_real_extras(
            traversability, elevation, z_min_arr, point_count_2d, occupied_2d, uncertainty
        )

        # Overhead-clearance classification (EV-2): per-cell, from this
        # frame's real raw point z-values only -- distinguishes a genuine
        # vertical gap under a raised obstacle (overhead_obstacle) from a
        # column solid all the way to the ground (solid_obstacle). See
        # src/perception/overhead_detection.py for the algorithm.
        from src.perception.overhead_detection import compute_overhead_clearance_grid
        overhead = compute_overhead_clearance_grid(
            x, y, z, x_min, y_min, self.x_bins, self.y_bins, self.grid_res,
            **self._overhead_params(),
        )

        # Flip vertically so row 0 is max Y (top of image)
        return {
            "empty": False,
            "elevation": np.flipud(elevation),
            "traversability": np.flipud(traversability),
            "uncertainty": np.flipud(uncertainty),
            "overhead": np.flipud(overhead),
            "extras": extras,
        }

    def _finish_grids(self, base: dict, tracks: list) -> dict:
        """Add the track-dependent parts (ROI layer, corridor object markers)."""
        x_min, x_max, y_min, y_max = self.grid_extent
        if base["empty"]:
            empty = base["empty_grid"]
            self._last_extras = {
                "road_width_m": None,
                "elevation_profile": [],
                "elevation_hazards_m": [],
                "hazard_clusters": [],
                "mean_uncertainty": None,
                "occupied_fraction": None,
                "centerline_profile": [],
                "centerline_markers": [],
            }
            return {
                "elevation": empty, "traversability": empty, "roi": empty,
                "uncertainty": empty, "overhead": base["empty_overhead"],
            }

        # ROI mock: higher around dynamic objects
        roi = np.zeros((self.y_bins, self.x_bins), dtype=np.float32)
        for t in tracks:
            if t.dynamic:
                tx, ty = t.state[0], t.state[1]
                t_col = int((tx - x_min) / self.grid_res)
                t_row = int((ty - y_min) / self.grid_res)
                if 0 <= t_col < self.x_bins and 0 <= t_row < self.y_bins:
                    # Draw a blob
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            r, c = t_row + i, t_col + j
                            if 0 <= r < self.y_bins and 0 <= c < self.x_bins:
                                roi[r, c] = max(roi[r, c], 1.0 - 0.2*max(abs(i), abs(j)))

        extras = dict(base["extras"])
        extras["centerline_markers"] = self._corridor_markers(extras["hazard_clusters"], tracks)
        self._last_extras = extras

        return {
            "elevation": base["elevation"],
            "traversability": base["traversability"],
            "roi": np.flipud(roi),
            "uncertainty": base["uncertainty"],
            "overhead": base["overhead"],
        }

    def _corridor_markers(self, hazard_clusters: list, tracks: list) -> list:
        """Hazard + tracked-object markers along the forward corridor."""
        from src.perception.centerline_profile import find_corridor_markers

        y_max = self.grid_extent[3]
        corridor_objects = list(hazard_clusters)
        for t in tracks:
            tx, ty = float(t.state[0]), float(t.state[1])
            corridor_objects.append({
                "x": tx, "y": ty,
                "distance_m": round(float(np.hypot(tx, ty)), 2),
                "label": t.class_name,
                "type": "track",
                "track_id": t.track_id,
            })
        return find_corridor_markers(
            corridor_objects, self.grid_extent,
            ego_position=(0.0, 0.0), ego_heading_rad=math.pi / 2.0,
            width_m=2.0, max_range_m=y_max,
        )

    def _compute_real_extras(
        self,
        traversability: np.ndarray,
        elevation: np.ndarray,
        z_min_2d: np.ndarray,
        point_count_2d: np.ndarray,
        occupied_2d: np.ndarray,
        uncertainty_2d: np.ndarray,
    ) -> dict:
        """Compute road width, elevation profile, hazard clusters, the Phase 5
        uncertainty summary and the corridor profile from the real (pre-flip)
        local grid. `centerline_markers` needs the tracks, so it is a
        placeholder here and filled in by _finish_grids.

        All arrays are indexed [row=y, col=x] over self.grid_extent, not yet
        flipped for display -- kept that way here so real-world x/y math
        below stays simple.

        FORWARD_AXIS = +Y. This convention (road width / elevation
        profile sweep over rows at a fixed centerline column, i.e. Y is
        "ahead" and X is "lateral") must match
        src/dashboard/export_web_dashboard_data.py::compute_ego_terrain_status,
        which previously used the opposite (+X forward) and was corrected
        to agree with this file -- do not let the two drift apart again.
        """
        x_min, x_max, y_min, y_max = self.grid_extent

        # --- Road width: contiguous Safe (>=0.70) run of columns at the
        # ego's row (y nearest 0), containing the ego's column (x nearest 0).
        ego_row = int(np.clip(round((0.0 - y_min) / self.grid_res), 0, self.y_bins - 1))
        ego_col = int(np.clip(round((0.0 - x_min) / self.grid_res), 0, self.x_bins - 1))
        road_width_m = None
        row_trav = traversability[ego_row, :]
        safe_mask = np.nan_to_num(row_trav, nan=0.0) >= 0.70
        if safe_mask[ego_col]:
            left = ego_col
            while left - 1 >= 0 and safe_mask[left - 1]:
                left -= 1
            right = ego_col
            while right + 1 < self.x_bins and safe_mask[right + 1]:
                right += 1
            road_width_m = float((right - left + 1) * self.grid_res)

        # --- Elevation profile + hazard markers along the ego-forward
        # centerline (column nearest x=0), real z_max per row.
        centerline_col = ego_col
        elevation_profile = []
        elevation_hazards_m = []
        for r in range(self.y_bins):
            z = elevation[r, centerline_col]
            if not np.isnan(z):
                y_world = y_min + (r + 0.5) * self.grid_res
                elevation_profile.append((float(y_world), float(z)))
                t = traversability[r, centerline_col]
                if not np.isnan(t) and t < 0.40:
                    elevation_hazards_m.append(float(y_world))

        # --- Hazard clusters: connected components of Blocked (<0.40)
        # cells, same method as export_web_dashboard_data.find_hazard_clusters.
        hazard_clusters = []
        try:
            from scipy import ndimage

            from src.perception.terrain_features import classify_hazard_kind

            blocked = np.nan_to_num(traversability, nan=1.0) < 0.40
            labeled, n_clusters = ndimage.label(blocked)
            cell_area_m2 = self.grid_res ** 2
            for cluster_id in range(1, n_clusters + 1):
                cmask = labeled == cluster_id
                rows_idx, cols_idx = np.nonzero(cmask)
                if len(rows_idx) == 0:
                    continue
                min_trav = float(np.nanmin(traversability[cmask]))
                mean_density = float(point_count_2d[cmask].mean())
                confidence = float(np.clip(mean_density / 20.0, 0.0, 1.0))
                centroid_row = float(rows_idx.mean())
                centroid_col = float(cols_idx.mean())
                world_x = x_min + (centroid_col + 0.5) * self.grid_res
                world_y = y_min + (centroid_row + 0.5) * self.grid_res
                distance_m = float(np.hypot(world_x, world_y))
                severity = "Critical" if min_trav < 0.20 else "Caution"

                # Real Pothole/Bump classification: this cluster's own real
                # min/max elevation vs. a real local baseline taken from the
                # real occupied ring of cells immediately around it
                # (dilate minus the cluster itself) -- never the cluster's
                # own cells, or a dip would be compared against itself.
                dilated = ndimage.binary_dilation(cmask, iterations=2)
                ring = dilated & ~cmask & occupied_2d
                baseline_z = float(np.nanmedian(elevation[ring])) if np.any(ring) else None
                z_max_local = float(np.nanmax(elevation[cmask]))
                z_min_local = float(np.nanmin(z_min_2d[cmask]))
                kind, extent_m = classify_hazard_kind(z_max_local, z_min_local, baseline_z)

                hazard_clusters.append({
                    "cluster_id": cluster_id,
                    "x": round(world_x, 3),
                    "y": round(world_y, 3),
                    "distance_m": round(distance_m, 2),
                    "area_m2": round(len(rows_idx) * cell_area_m2, 2),
                    "min_traversability": round(min_trav, 3),
                    "severity": severity,
                    "confidence": round(confidence, 3),
                    "kind": kind,
                    "extent_m": round(extent_m, 3) if extent_m is not None else None,
                })
            hazard_clusters.sort(key=lambda c: c["distance_m"])
        except ImportError:
            pass

        # --- Uncertainty summary (Phase 5 formula, see _grid_base).
        occupied_fraction = float(occupied_2d.mean())
        mean_uncertainty = float(np.mean(uncertainty_2d))

        # --- Real forward-corridor cross-section (src/perception/centerline_profile.py).
        # No existing ego heading/position tracking was found anywhere in
        # this pipeline (checked dashboard_state.py, this module, and the
        # tracker) -- every FrameState is a single ego-centered frame with
        # no odometry. ego_position=(0,0) is the pipeline's established
        # ego-origin convention; heading=+Y matches this module's own
        # pre-existing forward-axis convention above (elevation_profile,
        # road_width already treat y as the forward axis). Capped to this
        # local grid's own 20m forward extent, not the function's 100m
        # default, since this grid does not extend further.
        from src.perception.centerline_profile import compute_centerline_profile
        centerline = compute_centerline_profile(
            elevation, z_min_2d, point_count_2d, self.grid_extent, self.grid_res,
            ego_position=(0.0, 0.0), ego_heading_rad=math.pi / 2.0,
            width_m=2.0, max_range_m=y_max, bin_size_m=1.0,
        )
        centerline_profile = [
            {
                "distance_m": b.distance_m,
                "height_above_baseline_m": b.height_above_baseline_m,
                "depth_below_baseline_m": b.depth_below_baseline_m,
                "point_count": b.point_count,
            }
            for b in centerline.bins
        ]

        for h in hazard_clusters:
            h.setdefault("label", h["severity"])
            h.setdefault("type", "hazard")

        return {
            "road_width_m": road_width_m,
            "elevation_profile": elevation_profile,
            "elevation_hazards_m": elevation_hazards_m,
            "hazard_clusters": hazard_clusters,
            "occupied_fraction": occupied_fraction,
            "mean_uncertainty": mean_uncertainty,
            "centerline_profile": centerline_profile,
            "centerline_markers": [],
        }

    def _compute_live_accuracy(
        self, points: np.ndarray, gt_labels: np.ndarray, pred_labels: np.ndarray
    ) -> dict:
        """Real Near/Mid/Far/Overall distance-bucketed accuracy for the
        current frame, using the SAME src/10b_eval_distance_metrics.py
        ::compute_metrics() function as the already-validated offline
        evaluation -- imported and called directly, not reimplemented.

        Uses the identical bucket boundaries as that script's main():
        Near 0-10m, Mid 10-30m, Far 30-100m, Overall 0-100m (2D radial
        distance). compute_metrics() itself prints a full report to
        stdout per call (by design, for the offline CLI tool) -- that
        output is redirected away here since this runs every live frame,
        without touching the reused function itself.
        """
        import io
        import contextlib

        # OPT-2: Use the module cached at __init__ time instead of calling
        # importlib.import_module on every single frame.
        if self._eval_module is None:
            return {}
        compute_metrics = self._eval_module.compute_metrics

        x = points[:, 0]
        y = points[:, 1]
        distance = np.sqrt(x ** 2 + y ** 2)

        mask_near = (distance >= 0.0) & (distance < 10.0)
        mask_mid = (distance >= 10.0) & (distance < 30.0)
        mask_far = (distance >= 30.0) & (distance <= 100.0)
        mask_overall = (distance >= 0.0) & (distance <= 100.0)

        with contextlib.redirect_stdout(io.StringIO()):
            near = compute_metrics(gt_labels, pred_labels, mask_near, "Near (0-10m)")
            mid = compute_metrics(gt_labels, pred_labels, mask_mid, "Mid (10-30m)")
            far = compute_metrics(gt_labels, pred_labels, mask_far, "Far (30-100m)")
            overall = compute_metrics(gt_labels, pred_labels, mask_overall, "Overall (0-100m)")

        return {"near": near, "mid": mid, "far": far, "overall": overall}

    def _get_hardware_metrics(self, fps: float, target_fps: float, latency: float) -> HardwareMetrics:
        # psutil/pynvml are system calls (~0.5-1 ms each), so the system-level
        # snapshot is reused for up to 100 ms. Every call still returns its OWN
        # object with this frame's fps/latency: FrameStates are kept for STEP
        # back, so a shared, mutated object would rewrite old frames' numbers.
        now = time.perf_counter()
        if (self._last_hw_snapshot is not None
                and (now - self._last_hw_time) < 0.1):
            return dataclasses.replace(
                self._last_hw_snapshot,
                fps=fps, achieved_fps=fps, target_fps=target_fps, latency_ms=latency * 1000.0,
            )

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        
        hm = HardwareMetrics(
            fps=fps,
            target_fps=target_fps,
            achieved_fps=fps,
            latency_ms=latency * 1000.0,
            cpu_percent=cpu_percent,
            ram_percent=mem.percent,
            ram_used_gb=mem.used / (1024**3),
            ram_total_gb=mem.total / (1024**3),
            gpu_available=HAS_PYNVML
        )
        
        if HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                hm.gpu_percent = util.gpu
                hm.vram_used_gb = mem_info.used / (1024**3)
                hm.vram_total_gb = mem_info.total / (1024**3)
                hm.vram_percent = (mem_info.used / mem_info.total) * 100.0
            except:
                hm.gpu_available = False

        self._last_hw_time = now
        self._last_hw_snapshot = hm
        return dataclasses.replace(hm)

    def _setup_sources(self):
        """Prepare file list or synthetic generators based on self.source."""
        self._sample_static_pts = None
        self._sample_moving_pts = None
        self._sample_translation_step = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        self._file_list = []

        if self.lidar_source is not None:
            return

        if self.source == "sample":
            self._sample_static_pts, self._sample_moving_pts = generate_synthetic_sample_cloud(
                n_static=2000, n_moving=150
            )
        elif self.source == "semantickitti":
            kitti_root = Path(self.data_root) if self.data_root else Path("data/semantic_kitti/dataset/sequences")
            if (kitti_root / "dataset" / "sequences").exists():
                kitti_root = kitti_root / "dataset" / "sequences"
            elif (kitti_root / "sequences").exists():
                kitti_root = kitti_root / "sequences"
            seq_dir = kitti_root / self.sequence / "velodyne"
            if seq_dir.exists():
                self._file_list = sorted(list(seq_dir.glob("*.bin")))
            if not self._file_list:
                print(f"[WARNING] No SemanticKITTI .bin files found in {seq_dir}. Falling back to sample generator.")
                self.source = "sample"
                self._sample_static_pts, self._sample_moving_pts = generate_synthetic_sample_cloud(
                    n_static=2000, n_moving=150
                )
        elif self.source == "rellis3d":
            rellis_root = Path(self.data_root) if self.data_root else Path("C:/dev/data/rellis3d")
            if not rellis_root.exists():
                rellis_root = Path("data/rellis3d")
            seq_dir = rellis_root / "Rellis-3D" / self.sequence / "os1_cloud_node_kitti_bin"
            if seq_dir.exists():
                self._file_list = sorted(list(seq_dir.glob("*.bin")))
            if not self._file_list:
                print(f"[WARNING] No RELLIS-3D .bin files found in {seq_dir}. Falling back to sample generator.")
                self.source = "sample"
                self._sample_static_pts, self._sample_moving_pts = generate_synthetic_sample_cloud(
                    n_static=2000, n_moving=150
                )

    def _get_frame_data(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Fetch (points, semantic_labels) for frame_idx from active source."""
        if self.lidar_source is not None:
            f = self.lidar_source.frame(frame_idx % len(self.lidar_source))
            return f.points, f.foveax_labels

        if self.source == "sample":
            offset = self._sample_translation_step * frame_idx
            frame_moving = repeated_cloud_with_translation(self._sample_moving_pts, offset)
            return np.concatenate([self._sample_static_pts, frame_moving], axis=0), None

        if not self._file_list:
            return np.zeros((0, 4), dtype=np.float32), None

        file_path = self._file_list[frame_idx % len(self._file_list)]

        if self.source == "rellis3d" and HAS_RELLIS_LOADER:
            rellis_root = Path(self.data_root) if self.data_root else Path("C:/dev/data/rellis3d")
            if not rellis_root.exists():
                rellis_root = Path("data/rellis3d")
            try:
                frame_id = file_path.stem
                rf = load_rellis3d_frame(rellis_root, self.sequence, frame_id)
                return rf.points, rf.foveax_class_ids
            except Exception:
                # Fallback to direct raw bin loading if labels missing
                raw = np.fromfile(file_path, dtype=np.float32)
                return raw.reshape(-1, 4), None

        # Standard SemanticKITTI or raw binary
        raw = np.fromfile(file_path, dtype=np.float32)
        if raw.size % 4 != 0:
            return np.zeros((0, 4), dtype=np.float32), None
        points = raw.reshape(-1, 4)

        semantic_labels = None
        label_file = file_path.parent.parent / "labels" / f"{file_path.stem}.label"
        if label_file.exists():
            try:
                from src.perception.semantic_predictor import (
                    extract_semantickitti_ids,
                    remap_semantickitti_to_foveax,
                )
                raw_labels = np.fromfile(label_file, dtype=np.uint32)
                sem_ids, _ = extract_semantickitti_ids(raw_labels)
                semantic_labels = remap_semantickitti_to_foveax(sem_ids)
            except Exception:
                semantic_labels = None

        return points, semantic_labels

    def available_frames(self) -> int | None:
        """Number of real frames on disk for the active source (None = unbounded)."""
        if self.lidar_source is not None:
            return len(self.lidar_source)
        return len(self._file_list) or None

    def timestamp_for(self, frame_idx: int) -> float:
        """Timestamp (s) of a frame: the data's own if the source has them,
        otherwise frame_idx / sensor_hz. Past the end of a finite source it
        keeps counting at sensor_hz so it stays strictly increasing."""
        src = self.lidar_source
        if src is not None:
            n = len(src)
            if frame_idx < n:
                return src.timestamp(frame_idx)
            return src.timestamp(n - 1) + (frame_idx - n + 1) / self.sensor_hz
        return frame_idx / self.sensor_hz

    def _get_frame_points(self, frame_idx: int) -> np.ndarray:
        """Fetch points for frame_idx from active source."""
        points, _ = self._get_frame_data(frame_idx)
        return points

    def compute_stateless(self, frame_idx: int, timestamp_s: float) -> dict:
        """Everything about a frame that does NOT depend on the tracker:
        load, semantic labels, live accuracy, detection + classification +
        terrain features, and the tracker-independent grid maps.

        It is a pure function of (frame_idx, timestamp_s) and this object's
        configuration, so it can run in a worker process for frames ahead of
        the one being tracked/rendered (src/dashboard/frame_pipeline.py). The
        result is finished, in frame order, by finish_frame().
        """
        t_start = time.perf_counter()
        # Optional StageTimer (set only by `--profile`); None => every region
        # below is a no-op and behavior is identical to the un-instrumented code.
        _timer = getattr(self, "stage_timer", None)

        def _t(name):
            return _timer.time(name) if _timer is not None else nullcontext()

        with _t("load"):
            points, gt_labels = self._get_frame_data(frame_idx)
        # gt_labels is REAL ground truth ONLY (loaded from a .label file or
        # the RELLIS-3D loader) -- never a model prediction. Kept separate
        # from any prediction below so the two can be honestly compared.

        semantic_labels = gt_labels

        # If external semantic predictor is supplied and no real ground
        # truth exists for this frame, fall back to its prediction for
        # object-classification cross-referencing (existing behavior,
        # unchanged).
        if semantic_labels is None and getattr(self, "semantic_predictor", None) is not None and len(points) > 0:
            try:
                prediction = self.semantic_predictor.predict(points)
                semantic_labels = prediction.class_ids
            except Exception:
                pass

        # --- Real distance-bucketed accuracy (Task: live metrics) -------
        # Reuses src/10b_eval_distance_metrics.py::compute_metrics()
        # directly -- the exact same, already-validated function used by
        # the offline evaluation script, not a reimplementation. Only
        # computed when real ground truth exists for this frame; a
        # "ground_truth" predictor_mode compares gt against itself
        # (trivially ~100%, still real, not fabricated -- labeled as such
        # in the UI) while "mock"/"salsanext" compare a real independent
        # prediction against real ground truth.
        has_ground_truth = gt_labels is not None
        accuracy_metrics = None
        if has_ground_truth and len(points) > 0:
            if self.predictor_mode == "ground_truth" or self.semantic_predictor is None:
                pred_labels_for_accuracy = gt_labels
            else:
                try:
                    accuracy_prediction = self.semantic_predictor.predict(points)
                    pred_labels_for_accuracy = accuracy_prediction.class_ids
                except Exception:
                    pred_labels_for_accuracy = None

            if pred_labels_for_accuracy is not None:
                with _t("live_accuracy_metrics"):
                    accuracy_metrics = self._compute_live_accuracy(points, gt_labels, pred_labels_for_accuracy)

        # Coordinate frame sanity check
        if getattr(self.detector, "_SOURCE", "") != "mock_geometric_clusterer" and len(points) > 0:
            x_max_cloud = np.max(points[:, 0])
            if x_max_cloud < 0 or x_max_cloud > 100:
                print(f"[WARNING] Point cloud X range out of expected KITTI coordinate bounds (x_max={x_max_cloud:.1f}).")

        # Detect with semantic cross-referencing
        with _t("clustering_heuristics"):
            try:
                detections = self.detector.detect(points, timestamp_s=timestamp_s, semantic_labels=semantic_labels)
            except TypeError:
                detections = self.detector.detect(points, timestamp_s=timestamp_s)

            from src.perception.object_detector import classify_cluster_from_semantics
            for d in detections:
                if d.class_name == "unknown_obstacle":
                    cid, cname = classify_cluster_from_semantics(
                        d.center_xyz, d.size_lwh, points, semantic_labels, fallback_class="unclassified"
                    )
                    d.class_id = cid
                    d.class_name = cname

            self._attach_terrain_features(detections, points)

        # Grids (every frame, from every point -- nothing is skipped or thinned)
        with _t("grid_maps_hazards"):
            grid_base = self._grid_base(points)

        with _t("display_downsample"):
            display_points = self._display_downsample(points)

        # Cloud bounds for the coordinate-frame sanity check (a full pass over
        # every point, so it is computed here, off the GUI thread, not in finish_frame).
        point_bounds = (np.min(points[:, :2], axis=0), np.max(points[:, :2], axis=0)) if len(points) > 0 else None

        return {
            "frame_idx": frame_idx,
            "points": points,
            "has_ground_truth": has_ground_truth,
            "accuracy_metrics": accuracy_metrics,
            "detections": detections,
            "grid_base": grid_base,
            "display_points": display_points,
            "point_bounds": point_bounds,
            "compute_s": time.perf_counter() - t_start,
        }

    def finish_frame(self, result: dict, timestamp_s: float, fps: float = 10.0) -> FrameState:
        """Track (stateful, strictly in frame order), add the track-dependent
        grid parts, and assemble the FrameState from a compute_stateless()
        result computed here or in a worker process."""
        t_start = time.perf_counter()
        _timer = getattr(self, "stage_timer", None)

        def _t(name):
            return _timer.time(name) if _timer is not None else nullcontext()

        with _t("kalman_tracking"):
            tracks = [_snapshot_track(t) for t in self.tracker.update(result["detections"], timestamp_s=timestamp_s)]

        track_velocities = {
            t.track_id: (float(t.state[3]), float(t.state[4]))
            for t in tracks
        }

        with _t("grid_maps_hazards"):
            grids = self._finish_grids(result["grid_base"], tracks)

        # Processing time of this frame (excludes time spent waiting in a queue).
        latency = result["compute_s"] + (time.perf_counter() - t_start)
        target_fps = self.rate_hz

        _tail = _t("metrics_state_logging")
        _tail.__enter__()
        metrics = self._get_hardware_metrics(fps, target_fps, latency)

        extras = getattr(self, "_last_extras", {})
        state = FrameState(
            frame_idx=result["frame_idx"],
            timestamp_s=timestamp_s,
            points=result["points"],
            tracks=tracks,
            track_velocities=track_velocities,
            grid_maps=grids,
            grid_resolution_m=self.grid_res,
            grid_extent_m=self.grid_extent,
            metrics=metrics,
            road_width_m=extras.get("road_width_m"),
            elevation_profile=extras.get("elevation_profile", []),
            elevation_hazards_m=extras.get("elevation_hazards_m", []),
            hazard_clusters=extras.get("hazard_clusters", []),
            mean_uncertainty=extras.get("mean_uncertainty"),
            occupied_fraction=extras.get("occupied_fraction"),
            centerline_profile=extras.get("centerline_profile", []),
            centerline_markers=extras.get("centerline_markers", []),
            has_ground_truth=result["has_ground_truth"],
            predictor_mode=self.predictor_mode,
            accuracy_metrics=result["accuracy_metrics"],
            display_points=result.get("display_points"),
        )
        state.coordinate_warnings = assert_coordinate_frame_consistency(state, result.get("point_bounds"))
        for w in state.coordinate_warnings:
            print(f"[WARNING] {w}")

        self._log_session(state)
        _tail.__exit__(None, None, None)
        return state

    def process_frame(self, frame_idx: int, timestamp_s: float, fps: float = 10.0) -> FrameState:
        """Process a single frame through detection, tracking, grid mapping, and metrics."""
        return self.finish_frame(self.compute_stateless(frame_idx, timestamp_s), timestamp_s, fps)

    def run(self):
        self._setup_sources()
        dt = 1.0 / self.rate_hz
        current_time = 0.0
        last_t = time.perf_counter()

        for frame_idx in range(self.num_frames):
            if not self.running:
                break

            t_start = time.perf_counter()
            fps = 1.0 / (t_start - last_t) if (t_start - last_t) > 0 else self.rate_hz
            last_t = t_start

            state = self.process_frame(frame_idx, current_time, fps)
            self.state_ready.emit(state)

            current_time += dt
            elapsed = time.perf_counter() - t_start
            sleep_t = max(0, dt - elapsed)
            time.sleep(sleep_t)
            
    def _log_session(self, state: FrameState):
        import json
        import os
        from pathlib import Path
        
        # The log file stays open between frames (reopening it every frame cost
        # more than the rest of the logging); the size-based rotation (10 MB) is
        # checked every 200 frames, which is plenty for ~100-byte records.
        try:
            fh, n = self._log_fh, self._log_writes
        except (AttributeError, RuntimeError):
            fh, n = None, 0
        if fh is None or n % 200 == 0:
            log_dir = Path("outputs/phase9/session")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "dashboard_session.jsonl"
            if fh is not None and log_path.exists() and log_path.stat().st_size > 10 * 1024 * 1024:
                fh.close()
                fh = None
                log_path.rename(log_dir / f"dashboard_session_{int(time.time())}.jsonl")
            if fh is None:
                fh = open(log_path, "a", buffering=1)  # line-buffered: each record reaches the file
            self._log_fh = fh
        self._log_writes = n + 1

        record = {
            "schema_version": state.schema_version,
            "frame_idx": state.frame_idx,
            "timestamp_s": state.timestamp_s,
            "num_points": len(state.points),
            "num_tracks": len(state.tracks),
            "fps": state.metrics.fps,
            "target_fps": state.metrics.target_fps,
            "achieved_fps": state.metrics.achieved_fps,
            "latency_ms": state.metrics.latency_ms,
            "cpu_percent": state.metrics.cpu_percent
        }
        
        fh.write(json.dumps(record) + chr(10))

    def stop(self):
        self.running = False
        self.wait()
        fh = getattr(self, "_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
            self._log_fh = None
