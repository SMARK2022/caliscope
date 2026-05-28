"""Raw frame access for recorded video files.

FrameSource provides synchronous frame reading and seeking with no threading,
queues, or tracking. It wraps OpenCV's VideoCapture for fast decoding while
explicitly disabling display-orientation metadata so decoded frames stay in the
encoded raster coordinate system used by calibration intrinsics.
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Self

import cv2
import numpy as np

from caliscope.recording.video_utils import _open_video_capture_no_auto_rotation

logger = logging.getLogger(__name__)


class FrameSource:
    """Raw frame access for recorded video files.

    Provides synchronous frame reading and seeking. Thread-safe for concurrent
    method calls (internal state protected by lock), but NOT thread-safe for
    access patterns: if multiple threads share a FrameSource, they must
    coordinate externally to avoid interleaved seek/read sequences.

    Typical usage: one owner thread, or explicit external synchronization.

    Note: get_frame() and get_nearest_keyframe() invalidate the sequential read
    position. Use a fresh FrameSource for predictable mixed random/sequential
    access.
    """

    # Decode forward instead of re-seeking for moderate positive jumps. For the
    # calibration path this preserves the intended seek-first + sequential-grab
    # behavior when frame_step is 5/10/30 while avoiding long accidental scans.
    _MIN_SEQUENTIAL_GRAB_THRESHOLD = 30
    _SEQUENTIAL_GRAB_SECONDS = 4.0

    def __init__(self, video_directory: Path, cam_id: int) -> None:
        """Open a video file for frame access.

        Args:
            video_directory: Directory containing cam_N.mp4 video files.
            cam_id: Camera identifier (used to construct filename).

        Raises:
            ValueError: If the video stream lacks required metadata.
            FileNotFoundError: If the video file doesn't exist.
        """
        video_path = video_directory / f"cam_{cam_id}.mp4"
        self._open(video_path=video_path, cam_id=cam_id)

    @classmethod
    def from_path(cls, video_path: Path, cam_id: int | None = None) -> Self:
        """Construct a FrameSource from an explicit video file path.

        Bypasses the video_directory / cam_N.mp4 naming convention used by
        __init__.
        """
        resolved_cam_id = cam_id if cam_id is not None else 0
        instance = cls.__new__(cls)
        instance._open(video_path=video_path, cam_id=resolved_cam_id)
        return instance

    def _open(self, video_path: Path, cam_id: int) -> None:
        """Shared initialization body for __init__ and from_path."""
        self.cam_id = cam_id
        self.video_path = video_path

        self._container = _open_video_capture_no_auto_rotation(self.video_path)
        self._lock = Lock()

        fps = float(self._container.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            self._container.release()
            raise ValueError(f"Video stream has no valid FPS: {self.video_path}")
        self.fps = fps

        width = int(round(self._container.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self._container.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width <= 0 or height <= 0:
            self._container.release()
            raise ValueError(f"Video stream has invalid dimensions: {self.video_path}")
        self.size = (width, height)

        metadata_frame_count = int(round(self._container.get(cv2.CAP_PROP_FRAME_COUNT)))
        if metadata_frame_count <= 0:
            self._container.release()
            raise ValueError(f"Video stream has no frame count: {self.video_path}")

        self._keyframe_indices: list[int] = []
        self._keyframe_pts: list[int] = []
        self._actual_last_frame_index = self._find_last_accessible_frame(metadata_frame_count)
        self.frame_count = self._actual_last_frame_index + 1

        self._sequential_position: int = -1
        self._sequential_grab_threshold = max(
            self._MIN_SEQUENTIAL_GRAB_THRESHOLD,
            int(round(self.fps * self._SEQUENTIAL_GRAB_SECONDS)),
        )
        self._reset_to_start()

        logger.debug(
            "FrameSource for cam_id %s: OpenCV metadata says %s frames, "
            "actual last accessible frame is %s, size=%s, fps=%s",
            cam_id,
            metadata_frame_count,
            self._actual_last_frame_index,
            self.size,
            self.fps,
        )

        self._closed = False

    @property
    def start_frame_index(self) -> int:
        """First valid frame index (always 0 for raw video)."""
        return 0

    @property
    def last_frame_index(self) -> int:
        """Last accessible frame index."""
        return self._actual_last_frame_index

    def _find_last_accessible_frame(self, metadata_frame_count: int) -> int:
        """Probe near the metadata frame count to avoid full-video scans."""
        assert self._container is not None
        last_candidate = metadata_frame_count - 1
        # Most MP4s report the correct last frame. Probe a small window for the
        # occasional container that overstates accessibility near EOF.
        max_backoff = min(metadata_frame_count, 120)
        for offset in range(max_backoff):
            frame_index = last_candidate - offset
            if frame_index < 0:
                break
            if self._read_exact_frame_locked(frame_index) is not None:
                return frame_index
        return max(0, last_candidate)

    def _reset_to_start(self) -> None:
        if self._container is not None:
            self._container.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._sequential_position = -1

    def _read_exact_frame_locked(self, frame_index: int) -> np.ndarray | None:
        """Seek to a frame and read it. Caller must hold the lock when public."""
        if self._container is None:
            return None
        if not self._container.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            return None
        ok, frame = self._container.read()
        if not ok or frame is None:
            return None
        self._sequential_position = frame_index
        return frame

    def get_frame(self, frame_index: int) -> np.ndarray | None:
        """Seek to exact frame and return it as BGR numpy array.

        Uses OpenCV VideoCapture with orientation auto-rotation disabled.

        Note:
            Invalidates sequential read position.
        """
        with self._lock:
            if self._container is None:
                return None
            if frame_index < 0 or frame_index > self.last_frame_index:
                return None
            return self._read_exact_frame_locked(frame_index)

    def read_frame_at(self, frame_index: int) -> np.ndarray | None:
        """Read frame at index, optimizing for sequential access patterns.

        Cold starts and large jumps seek directly to the target. Moderate
        positive gaps decode forward with ``grab()`` and one ``read()``, which is
        substantially faster than repeated seeking on GoPro H.265 files.

        Not protected by self._lock -- designed for single-owner access in the
        parallel processing pipeline (one FrameSource per camera per thread).
        """
        if frame_index < 0 or frame_index > self.last_frame_index:
            return None
        if self._container is None:
            return None

        gap = frame_index - self._sequential_position
        if 0 < gap <= self._sequential_grab_threshold:
            current_pos = int(round(self._container.get(cv2.CAP_PROP_POS_FRAMES)))
            expected_pos = self._sequential_position + 1
            if current_pos == expected_pos:
                for _ in range(max(0, gap - 1)):
                    if not self._container.grab():
                        return None
                ok, frame = self._container.read()
                if not ok or frame is None:
                    return None
                self._sequential_position = frame_index
                return frame

        return self._read_exact_frame_locked(frame_index)

    def get_nearest_keyframe(self, frame_index: int) -> tuple[np.ndarray | None, int]:
        """Return a frame suitable for fast scrubbing.

        OpenCV does not expose portable keyframe-index access through
        VideoCapture, so this backend returns the requested frame itself. The
        public contract remains the same: a BGR frame and its actual index, or
        ``(None, -1)`` on invalid input/failure.
        """
        with self._lock:
            if self._container is None:
                return None, -1
            if frame_index < 0 or frame_index > self.last_frame_index:
                return None, -1
            frame = self._read_exact_frame_locked(frame_index)
            if frame is None:
                return None, -1
            return frame, frame_index

    def read_frame(self) -> np.ndarray | None:
        """Read next frame sequentially, returning BGR numpy array or None at EOF."""
        with self._lock:
            if self._container is None:
                return None
            current_pos = int(round(self._container.get(cv2.CAP_PROP_POS_FRAMES)))
            ok, frame = self._container.read()
            if not ok or frame is None:
                return None
            self._sequential_position = current_pos
            return frame

    def close(self) -> None:
        """Release video container resources."""
        with self._lock:
            self._closed = True
            if self._container is not None:
                self._container.release()
                self._container = None
            self._sequential_position = -1

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - releases resources."""
        self.close()

    def __del__(self) -> None:
        """Destructor - warns if resources were not properly released."""
        if not getattr(self, "_closed", True):
            logger.warning(
                f"FrameSource for {self.video_path} was not closed properly. "
                "Use context manager or call close() explicitly."
            )
            self.close()
