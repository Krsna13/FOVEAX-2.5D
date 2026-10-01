# Input sources: run FOVEAX on any LiDAR dataset or sensor dump

The dashboard no longer needs the data to be RELLIS-3D or SemanticKITTI. A
small input layer (`src/inputs/`) converts any recording into what the
pipeline consumes per frame:

- points: an (N, 4) float32 array `[x, y, z, intensity]`, metres, vehicle
  frame, z up, ground near z = 0
- optional ground-truth labels: one FOVEAX class ID (0-7) per point
- a timestamp, so tracker speeds (m/s) are correct

Everything after that (grids, traversability, foveated zones, clustering,
tracking, dashboard) is sensor-independent.

## Quick start

```bash
# a folder of scans, format detected from the file extension
python src/13_realtime_dashboard.py --input-dir D:/my_lidar/scans --sensor-hz 10

# full control through a config file
python src/13_realtime_dashboard.py --input-config configs/inputs/example_folder.yaml

# the two built-in datasets, through the same mechanism
python src/13_realtime_dashboard.py --input-config configs/inputs/rellis3d.yaml
python src/13_realtime_dashboard.py --input-config configs/inputs/semantickitti.yaml
```

`--headless` and `--profile` accept the same flags. Without any input flag the
dashboard behaves exactly as before (`--source rellis3d|semantickitti|sample`).

## Source types

| `type` | Reads | Labels |
|---|---|---|
| `folder` (default) | any directory of scan files: `.bin` `.npy` `.npz` `.ply` `.pcd` `.csv` `.txt` `.xyz` | optional, configurable |
| `rellis3d` | `Rellis-3D/<seq>/os1_cloud_node_*` | from the dataset (sky removed) |
| `semantickitti` | `sequences/<seq>/velodyne` + `labels` | from the dataset |

## The `folder` config

Every key except `path` is optional. `configs/inputs/example_folder.yaml` is a
commented template. Relative paths resolve against the config file's folder.

| Key | Meaning |
|---|---|
| `path` | Directory with the scan files (required) |
| `pattern`, `recursive` | Which files; default is the most common supported extension |
| `format` | `bin`, `npy`, `npz`, `ply`, `pcd`, `csv`, `txt`, `xyz` |
| `n_cols` | float32 values per point in a `.bin` (KITTI 4, nuScenes 5) |
| `columns` | Which columns hold x, y, z and intensity (`intensity: null` if absent) |
| `axes` | Sensor axes to vehicle frame, e.g. `"-y,x,z"` for a sensor that is not x-forward |
| `z_offset` | Metres added to z so the ground sits near 0 (the pipeline's ground threshold assumes this) |
| `intensity_scale` | e.g. `0.00392` turns 0-255 into 0-1 |
| `drop_nonfinite` | Drop NaN/inf points (default true) |
| `drop_zero_returns` | Drop `(0,0,0)` padding returns (default false; RELLIS-3D has many) |
| `min_range_m`, `max_range_m` | Range gate |
| `sensor` | `name`, `hz` (scan rate), optional `beams`, `width`, `fov_up`, `fov_down` |
| `timestamps` | `mode: index` (frame / hz), `filename` (numeric file names) or `file` (one time per line), with `unit: s|ms|us|ns` |
| `labels` | `dir`, `ext`, `format` (`semantickitti`, `uint8`, `uint16`, `int32`, `npy`), `map` |

Label `map` is `semantickitti`, `rellis3d`, `foveax` (already FOVEAX IDs) or
your own `{raw_id: foveax_id}` table. Unmapped IDs become UNKNOWN (7). The
FOVEAX classes are 0 drivable ground, 1 rough terrain, 2 vegetation, 3
building/wall, 4 solid obstacle, 5 vehicle, 6 pedestrian, 7 unknown.

Typos in a config are rejected with the list of valid keys, and mismatches
(label count vs. point count, timestamps not increasing, a column outside the
file) raise a specific error instead of producing a wrong map.

## Sensor rate and speeds

Timestamps come from the data (`timestamps`) or from `sensor.hz`, **not** from
the playback speed (`--rate-hz`). Before this layer existed the dashboard used
`frame / rate_hz`, so replaying 10 Hz data at 30 Hz made every object look
faster (a 2.0 m/s test obstacle read 3.1 m/s over 12 frames; it now reads
1.9 m/s). `--sensor-hz` overrides the config value.

## Changing the sensor: what else to check

- The grid, traversability, tracker and clustering work on any x/y/z cloud,
  but thresholds are tuned for a 64-beam sensor: ground threshold 0.25 m,
  minimum cluster size 5 points, voxel 20 cm, cluster distance 1.5 m.
- Semantic labelling with SalsaNext is sensor-specific. `sensor.fov_up` and
  `fov_down` are passed to its range projection, but its image is fixed at 64
  x 2048 and the weights were trained on a Velodyne HDL-64E or fine-tuned on an
  Ouster OS1-64. A different beam count or a solid-state sensor needs a new
  projection and retraining (see `docs/rellis3d_integration.md` for what a
  mismatch costs: 6.65% accuracy).
- Without labels, objects show as "unclassified" and the accuracy panel has
  nothing to compare against. Nothing is invented.
- The RELLIS-3D environment buttons are hidden for custom inputs.

## Adding a new dataset type in code

```python
from src.inputs import LidarSource, LidarFrame, register_source_type

class MyDataset(LidarSource):
    name = "mine"; sensor_hz = 10.0
    def __init__(self, cfg, base_dir=None): ...
    def __len__(self): ...
    def frame(self, idx): return LidarFrame(points, labels_or_None, self.timestamp(idx), str(idx))

register_source_type("mine", MyDataset)   # then: type: mine  in a config
```

## Not included

- Live sensor streams: the dashboard replays recorded files. Live input goes
  through the ROS 2 node (`docs/phase10_ros2_integration.md`, never run on real
  ROS 2).
- Sensor fusion (several LiDARs, camera, radar, IMU). A multi-LiDAR merge would
  be a new source type that transforms each cloud into the vehicle frame and
  concatenates them.
