"""Audio-based timeline synchronization for multi-camera GoPro recordings.

This module writes timestamp metadata only. It does not re-encode or trim videos.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rtoml
from scipy import signal
from scipy.io import wavfile

from caliscope.persistence import _safe_write_toml
from caliscope.recording.video_utils import read_video_properties


@dataclass(frozen=True, slots=True)
class CameraSyncInfo:
    cam_id: int
    path: str
    offset_seconds: float
    fps: float
    frame_count: int
    original_duration_seconds: float
    start_frame: int
    end_frame: int
    synced_frame_count: int
    synced_duration_seconds: float
    timecode: str | None = None


@dataclass(frozen=True, slots=True)
class RecordingSyncInfo:
    reference_cam_id: int
    common_start_seconds: float
    common_end_seconds: float
    common_duration_seconds: float
    cameras: dict[int, CameraSyncInfo]
    timestamps_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "reference_cam_id": self.reference_cam_id,
            "common_start_seconds": self.common_start_seconds,
            "common_end_seconds": self.common_end_seconds,
            "common_duration_seconds": self.common_duration_seconds,
            "timestamps_path": self.timestamps_path,
            "cameras": {
                str(cam_id): {
                    key: value
                    for key, value in {
                    "cam_id": info.cam_id,
                    "path": info.path,
                    "offset_seconds": info.offset_seconds,
                    "fps": info.fps,
                    "frame_count": info.frame_count,
                    "original_duration_seconds": info.original_duration_seconds,
                    "start_frame": info.start_frame,
                    "end_frame": info.end_frame,
                    "synced_frame_count": info.synced_frame_count,
                    "synced_duration_seconds": info.synced_duration_seconds,
                    "timecode": info.timecode,
                    }.items()
                    if value is not None
                }
                for cam_id, info in sorted(self.cameras.items())
            },
        }


def load_sync_summary(recording_dir: Path) -> RecordingSyncInfo | None:
    path = recording_dir / "sync_offsets.toml"
    if not path.exists():
        return None

    data = rtoml.load(path)
    cameras = {
        int(cam_id): CameraSyncInfo(
            cam_id=int(value["cam_id"]),
            path=value["path"],
            offset_seconds=float(value["offset_seconds"]),
            fps=float(value["fps"]),
            frame_count=int(value["frame_count"]),
            original_duration_seconds=float(value["original_duration_seconds"]),
            start_frame=int(value["start_frame"]),
            end_frame=int(value["end_frame"]),
            synced_frame_count=int(value["synced_frame_count"]),
            synced_duration_seconds=float(value["synced_duration_seconds"]),
            timecode=value.get("timecode"),
        )
        for cam_id, value in data.get("cameras", {}).items()
    }
    return RecordingSyncInfo(
        reference_cam_id=int(data["reference_cam_id"]),
        common_start_seconds=float(data["common_start_seconds"]),
        common_end_seconds=float(data["common_end_seconds"]),
        common_duration_seconds=float(data["common_duration_seconds"]),
        cameras=cameras,
        timestamps_path=str(recording_dir / "timestamps.csv"),
        summary_path=str(path),
    )


def synchronize_recording_timeline(
    recording_dir: Path,
    cam_ids: list[int],
    *,
    target_sr: int = 16000,
    max_lag_sec: float = 30.0,
    max_seconds_for_xcorr: float = 120.0,
    is_cancelled: Callable[[], bool] | None = None,
) -> RecordingSyncInfo:
    """Synchronize videos by audio and persist a cropped shared timeline.

    The written ``timestamps.csv`` contains original frame indices plus aligned
    frame times for the common intersection window. Consumers read original
    frames directly from the existing videos.
    """
    if not cam_ids:
        raise ValueError("No cameras available for synchronization")

    clips = []
    for cam_id in sorted(cam_ids):
        _raise_if_cancelled(is_cancelled)
        clips.append(_load_clip(recording_dir / f"cam_{cam_id}.mp4", cam_id, target_sr))

    _raise_if_cancelled(is_cancelled)
    offsets = _match_audio_offsets(clips, max_lag_sec=max_lag_sec, max_seconds_for_xcorr=max_seconds_for_xcorr)

    common_start = max(offsets[clip["cam_id"]] for clip in clips)
    common_end = min(offsets[clip["cam_id"]] + clip["duration"] for clip in clips)
    if common_start >= common_end:
        raise ValueError("No overlapping time window after audio synchronization")

    rows: list[dict[str, float | int]] = []
    cameras: dict[int, CameraSyncInfo] = {}
    target_fps = min(float(clip["fps"]) for clip in clips)
    target_frames = max(1, int(math.floor((common_end - common_start) * target_fps + 1e-9)))
    for clip in clips:
        cam_id = int(clip["cam_id"])
        offset = offsets[cam_id]
        fps = float(clip["fps"])
        frame_count = int(clip["frame_count"])
        frame_indices: list[int] = []

        last_frame = -1
        for sync_index in range(target_frames):
            sync_time = sync_index / target_fps
            local_time = common_start + sync_time - offset
            frame_index = int(round(local_time * fps))
            if frame_index <= last_frame:
                frame_index = last_frame + 1
            if frame_index < 0 or frame_index >= frame_count:
                continue
            last_frame = frame_index
            frame_indices.append(frame_index)
            rows.append(
                {
                    "sync_index": sync_index,
                    "cam_id": cam_id,
                    "frame_index": frame_index,
                    "frame_time": sync_time,
                }
            )

        if not frame_indices:
            raise ValueError(f"cam_{cam_id} has no frames in the synchronized overlap window")

        start_frame = frame_indices[0]
        end_frame = frame_indices[-1]
        synced_frame_count = len(frame_indices)
        cameras[cam_id] = CameraSyncInfo(
            cam_id=cam_id,
            path=str(clip["path"]),
            offset_seconds=offset,
            fps=fps,
            frame_count=frame_count,
            original_duration_seconds=float(clip["duration"]),
            start_frame=start_frame,
            end_frame=end_frame,
            synced_frame_count=synced_frame_count,
            synced_duration_seconds=target_frames / target_fps,
            timecode=clip.get("timecode"),
        )

    timestamps_path = recording_dir / "timestamps.csv"
    _raise_if_cancelled(is_cancelled)
    _write_timestamps_csv(timestamps_path, rows)

    summary_path = recording_dir / "sync_offsets.toml"
    summary = RecordingSyncInfo(
        reference_cam_id=int(clips[0]["cam_id"]),
        common_start_seconds=common_start,
        common_end_seconds=common_end,
        common_duration_seconds=common_end - common_start,
        cameras=cameras,
        timestamps_path=str(timestamps_path),
        summary_path=str(summary_path),
    )
    _safe_write_toml(summary.to_dict(), summary_path)
    return summary


def _raise_if_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError


def _load_clip(path: Path, cam_id: int, target_sr: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    props = read_video_properties(path)
    duration = _ffprobe_duration(path)
    timecode = _ffprobe_timecode(path)
    audio = _extract_audio(path, target_sr)
    return {
        "cam_id": cam_id,
        "path": str(path),
        "audio": audio,
        "sr": target_sr,
        "duration": duration,
        "fps": float(props["fps"]),
        "frame_count": int(props["frame_count"]),
        "timecode": timecode,
    }


def _match_audio_offsets(
    clips: list[dict[str, Any]],
    *,
    max_lag_sec: float,
    max_seconds_for_xcorr: float,
) -> dict[int, float]:
    ref = clips[0]
    sr = int(ref["sr"])
    ref_proc = _normalized_prefix(ref["audio"], sr, max_seconds_for_xcorr)
    offsets = {int(ref["cam_id"]): 0.0}

    for clip in clips[1:]:
        sig_proc = _normalized_prefix(clip["audio"], sr, max_seconds_for_xcorr)
        expected_lag_sec = _timecode_diff_seconds(ref.get("timecode"), clip.get("timecode"), float(ref["fps"]))
        expected_lag_samples = int(round(expected_lag_sec * sr))
        max_lag = int(min(max_lag_sec * sr, len(ref_proc), len(sig_proc)))

        corr = signal.fftconvolve(ref_proc, sig_proc[::-1], mode="full")
        zero_lag = len(sig_proc) - 1
        center = zero_lag + expected_lag_samples
        lo = max(0, center - max_lag)
        hi = min(len(corr), center + max_lag + 1)
        if lo >= hi:
            raise ValueError(f"Invalid sync search window for cam_{clip['cam_id']}")

        peak = int(np.argmax(corr[lo:hi]) + lo)
        offsets[int(clip["cam_id"])] = (peak - zero_lag) / float(sr)

    return offsets


def _normalized_prefix(audio: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    length = min(len(audio), int(sr * max_seconds))
    if length <= 0:
        raise ValueError("Audio stream is empty")
    proc = audio[:length]
    return (proc - np.mean(proc)) / (np.std(proc) + 1e-8)


def _extract_audio(path: Path, target_sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-f",
            "wav",
            tmp_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        _sr, audio = wavfile.read(tmp_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
        if max_abs > 0:
            audio = audio / max_abs
        return audio
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return float(out.decode().strip())


def _ffprobe_timecode(path: Path) -> str | None:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    data = json.loads(out.decode())
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream.get("tags", {}).get("timecode")
    return None


def _timecode_diff_seconds(ref_timecode: str | None, target_timecode: str | None, fps: float) -> float:
    ref_seconds = _parse_timecode_seconds(ref_timecode, fps)
    target_seconds = _parse_timecode_seconds(target_timecode, fps)
    if ref_seconds is None or target_seconds is None:
        return 0.0

    diff = target_seconds - ref_seconds
    day_seconds = 24 * 3600
    half_day = day_seconds / 2
    if diff > half_day:
        diff -= day_seconds
    elif diff < -half_day:
        diff += day_seconds
    return diff


def _parse_timecode_seconds(timecode: str | None, fps: float) -> float | None:
    if timecode is None:
        return None
    match = re.match(r"^(\d+):(\d+):(\d+)[:;\.](\d+)$", timecode)
    if match is None:
        return None
    hours, minutes, seconds, frames = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + frames / fps


def _write_timestamps_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        f.write("sync_index,cam_id,frame_index,frame_time\n")
        for row in sorted(rows, key=lambda value: (value["sync_index"], value["cam_id"])):
            f.write(f"{row['sync_index']},{row['cam_id']},{row['frame_index']},{row['frame_time']:.9f}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
