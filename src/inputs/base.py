"""Common interface every LiDAR input source implements.

A source turns *any* dataset or sensor recording into the two things the
FOVEAX pipeline consumes per frame: an (N, 4) float32 array of
[x, y, z, intensity] in the vehicle frame (metres, z up, ground near z = 0),
and optionally one FOVEAX class ID per point (ground truth) for the live
accuracy comparison. Everything downstream (grids, traversability, tracker,
dashboard) is sensor-agnostic once it has that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

DEFAULT_SENSOR_HZ = 10.0


@dataclass(frozen=True)
class LidarFrame:
    """One frame, already converted to the pipeline's conventions."""

    points: np.ndarray  # (N, 4) float32: x, y, z, intensity
    foveax_labels: np.ndarray | None  # (N,) uint8 FOVEAX class IDs, or None
    timestamp_s: float
    frame_id: str
    meta: dict = field(default_factory=dict)


class LidarSource(ABC):
    """A finite, indexable recording of LiDAR frames."""

    name: str = "source"
    #: Nominal scan rate of the sensor that produced the data. Used to build
    #: timestamps when the data has none, so tracker speeds (m/s) are correct.
    sensor_hz: float = DEFAULT_SENSOR_HZ
    #: Free-form sensor description: name, beams, width, fov_up, fov_down, ...
    sensor: dict = {}

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def frame(self, idx: int) -> LidarFrame:
        """Load frame *idx* (0-based). Raises IndexError when out of range."""

    def timestamp(self, idx: int) -> float:
        """Timestamp of frame *idx* in seconds, without loading the frame."""
        return idx / self.sensor_hz

    def describe(self) -> str:
        return f"{self.name}: {len(self)} frames @ {self.sensor_hz:g} Hz"

    def close(self) -> None:  # pragma: no cover - default is a no-op
        """Release any resources (file handles, network connections)."""


def check_index(idx: int, n: int) -> int:
    if not 0 <= idx < n:
        raise IndexError(f"frame index {idx} out of range for a source with {n} frames")
    return idx
