"""Video file utilities.

Functions for reading video metadata without full frame decoding. Uses OpenCV
with display-orientation auto-rotation disabled so reported dimensions match the
encoded raster coordinate system used by calibration.
"""

import logging
from pathlib import Path
from typing import TypedDict

import cv2

logger = logging.getLogger(__name__)


class VideoProperties(TypedDict):
    """Video metadata returned by read_video_properties."""

    fps: float
    frame_count: int
    width: int
    height: int
    size: tuple[int, int]


def _open_video_capture_no_auto_rotation(source_path: Path) -> cv2.VideoCapture:
    """Open a VideoCapture in encoded-raster coordinates.

    GoPro MP4 files may carry display-rotation metadata. OpenCV applies that
    metadata by default on this platform, which changes the pixel coordinate
    system for 90-degree files and center-rotates 180-degree files. Calibration
    intrinsics are stored in encoded 1920x1080 coordinates, so all pipeline reads
    must disable auto-rotation explicitly.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(
            f"Could not open video file: {source_path}. "
            "The file may be corrupted or in an unsupported format."
        )

    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    return capture


def read_video_properties(source_path: Path) -> VideoProperties:
    """Read video metadata (fps, frame_count, dimensions) via OpenCV.

    Opens the video briefly to inspect stream metadata, then closes it. No full
    frame scan is performed.

    Raises:
        FileNotFoundError: If source_path does not exist.
        ValueError: If the file cannot be opened as video or metadata cannot be
            determined.
    """
    logger.info(f"Reading video properties from: {source_path}")
    capture = _open_video_capture_no_auto_rotation(source_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError(f"Could not determine frame rate for: {source_path}")

        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count <= 0:
            raise ValueError(f"Could not determine frame count for: {source_path}")

        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not determine dimensions for: {source_path}")

        return VideoProperties(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            size=(width, height),
        )
    finally:
        capture.release()
