from pathlib import Path

import pandas as pd
import pytest

from caliscope.recording.frame_timestamps import FrameTimestamps


def test_from_csv_interpolates_sparse_original_frame_indices(tmp_path: Path):
    """Audio-synced CSVs may skip frames; playback still needs dense timings."""
    timestamps_path = tmp_path / "timestamps.csv"
    pd.DataFrame(
        [
            {"sync_index": 0, "cam_id": 1, "frame_index": 10, "frame_time": 0.0},
            {"sync_index": 1, "cam_id": 1, "frame_index": 12, "frame_time": 1 / 30},
        ]
    ).to_csv(timestamps_path, index=False)

    frame_timestamps = FrameTimestamps.from_csv(timestamps_path, cam_id=1)

    assert frame_timestamps.start_frame_index == 10
    assert frame_timestamps.last_frame_index == 12
    assert frame_timestamps.get_time(10) == pytest.approx(0.0)
    assert frame_timestamps.get_time(11) == pytest.approx(1 / 60)
    assert frame_timestamps.get_time(12) == pytest.approx(1 / 30)
