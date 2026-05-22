"""Frame timestamp mapping for synchronized video playback.

FrameTimestamps maps frame indices to wall-clock timestamps recorded at capture time.
This enables synchronized playback across multiple cameras.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Self

import pandas as pd


@dataclass(frozen=True, slots=True)
class FrameTimestamps:
    """Maps frame indices to timestamps recorded at capture time.

    Frame indices may not start at 0 for synchronized recordings where
    cameras started at different times.

    Attributes:
        frame_times: Immutable mapping of frame_index -> timestamp (seconds).
    """

    frame_times: Mapping[int, float]

    @property
    def start_frame_index(self) -> int:
        """First valid frame index (may not be 0 for synced recordings)."""
        return min(self.frame_times.keys())

    @property
    def last_frame_index(self) -> int:
        """Last valid frame index."""
        return max(self.frame_times.keys())

    def get_time(self, frame_index: int) -> float:
        """Get wall-clock timestamp for a frame index.

        Raises:
            KeyError: If frame_index is not in the mapping.
        """
        return self.frame_times[frame_index]

    @classmethod
    def from_csv(cls, csv_path: Path, cam_id: int) -> Self:
        """Load timing from timestamps.csv.

        If a frame_index column exists, those original video frame indices are
        preserved. Legacy CSVs without frame_index still infer sequential
        indices by rank-ordering frame_time within the cam_id's rows.

        Args:
            csv_path: Path to timestamps.csv.
            cam_id: Camera identifier to extract timing for.

        Raises:
            FileNotFoundError: If csv_path doesn't exist.
            KeyError: If cam_id not found in CSV.
        """
        df = pd.read_csv(csv_path)
        cam_df = df[df["cam_id"] == cam_id].copy()

        if cam_df.empty:
            raise KeyError(f"cam_id {cam_id} not found in {csv_path}")

        if "frame_index" in cam_df.columns:
            cam_df["frame_index"] = cam_df["frame_index"].astype(int)
            cam_df = cam_df.sort_values("frame_index")
            frame_times = _interpolate_missing_frame_times(dict(zip(cam_df["frame_index"], cam_df["frame_time"])))
        else:
            # Legacy CSVs only have frame_time; infer sequential frame indices.
            cam_df["frame_index"] = cam_df["frame_time"].rank(method="min").astype(int) - 1
            frame_times = dict(zip(cam_df["frame_index"], cam_df["frame_time"]))

        return cls(MappingProxyType(frame_times))

    @classmethod
    def inferred(cls, fps: float, frame_count: int) -> Self:
        """Create timing inferred from FPS when no CSV exists.

        Generates timestamps assuming constant frame rate starting at t=0.

        Args:
            fps: Frames per second.
            frame_count: Total number of frames.
        """
        frame_times = {i: i / fps for i in range(frame_count)}
        return cls(MappingProxyType(frame_times))


def _interpolate_missing_frame_times(frame_times: dict[int, float]) -> dict[int, float]:
    """Fill gaps so sequential playback can ask for any frame in the synced span."""
    if len(frame_times) <= 1:
        return dict(frame_times)

    items = sorted((int(frame_index), float(frame_time)) for frame_index, frame_time in frame_times.items())
    filled: dict[int, float] = {}
    for (start_index, start_time), (end_index, end_time) in zip(items, items[1:]):
        filled[start_index] = start_time
        gap = end_index - start_index
        if gap <= 1:
            continue

        step = (end_time - start_time) / gap
        for frame_index in range(start_index + 1, end_index):
            filled[frame_index] = start_time + step * (frame_index - start_index)

    last_index, last_time = items[-1]
    filled[last_index] = last_time
    return filled
