"""Repository for per-camera source video identity metadata."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import rtoml

from caliscope.persistence import _safe_write_toml
from caliscope.recording.gopro_metadata import read_gopro_metadata


@dataclass(slots=True)
class CameraSource:
    """Source identity for a project camera."""

    cam_id: int
    label: str | None = None
    serial_number: str | None = None
    model: str | None = None
    firmware: str | None = None
    original_filename: str | None = None
    source_folder: str | None = None
    workspace_video: str | None = None
    intrinsic_video: str | None = None
    extrinsic_video: str | None = None
    casn_source: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraSource":
        return cls(
            cam_id=int(data["cam_id"]),
            label=data.get("label"),
            serial_number=data.get("serial_number"),
            model=data.get("model"),
            firmware=data.get("firmware"),
            original_filename=data.get("original_filename"),
            source_folder=data.get("source_folder"),
            workspace_video=data.get("workspace_video"),
            intrinsic_video=data.get("intrinsic_video"),
            extrinsic_video=data.get("extrinsic_video"),
            casn_source=data.get("casn_source"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class CameraSourcesRepository:
    """Persistence gateway for camera_sources.toml."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.path = workspace_dir / "camera_sources.toml"
        self.mapping_csv_path = workspace_dir / "camera_name_mapping.csv"

    def load(self) -> dict[int, CameraSource]:
        if not self.path.exists():
            return {}

        data = rtoml.load(self.path)
        cameras = data.get("cameras", {})
        return {int(cam_id): CameraSource.from_dict(value) for cam_id, value in cameras.items()}

    def save(self, sources: dict[int, CameraSource]) -> None:
        data = {
            "schema_version": 1,
            "cameras": {str(cam_id): source.to_dict() for cam_id, source in sorted(sources.items())},
        }
        _safe_write_toml(data, self.path)

    def load_mapping_csv(self) -> dict[int, CameraSource]:
        """Load legacy camera_name_mapping.csv when present."""
        if not self.mapping_csv_path.exists():
            return {}

        sources: dict[int, CameraSource] = {}
        with self.mapping_csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                camera_name = row.get("camera", "")
                if not camera_name.startswith("cam_"):
                    continue
                try:
                    cam_id = int(camera_name.split("_", 1)[1])
                except ValueError:
                    continue

                original_filename = row.get("original_filename") or None
                label = Path(original_filename).stem if original_filename else camera_name
                sources[cam_id] = CameraSource(
                    cam_id=cam_id,
                    label=label,
                    original_filename=original_filename,
                    source_folder=row.get("source_folder") or None,
                    workspace_video=row.get("video_path") or None,
                    intrinsic_video=row.get("intrinsic_link") or None,
                    extrinsic_video=row.get("extrinsic_link") or None,
                )
        return sources

    def refresh_from_workspace_videos(self, *, save: bool = True) -> dict[int, CameraSource]:
        """Merge existing source records, legacy mapping CSV, and MP4 CASN data."""
        sources = self.load()
        for cam_id, source in self.load_mapping_csv().items():
            existing = sources.get(cam_id)
            if existing is None:
                sources[cam_id] = source
                continue
            for field in (
                "label",
                "original_filename",
                "source_folder",
                "workspace_video",
                "intrinsic_video",
                "extrinsic_video",
            ):
                if getattr(existing, field) is None:
                    setattr(existing, field, getattr(source, field))

        for video_dir, field in (
            (self.workspace_dir / "videos", "workspace_video"),
            (self.workspace_dir / "calibration" / "intrinsic", "intrinsic_video"),
            (self.workspace_dir / "calibration" / "extrinsic", "extrinsic_video"),
        ):
            if not video_dir.exists():
                continue
            for path in video_dir.glob("cam_*.mp4"):
                try:
                    cam_id = int(path.stem.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue

                source = sources.setdefault(cam_id, CameraSource(cam_id=cam_id, label=f"cam_{cam_id}"))
                if getattr(source, field) is None:
                    setattr(source, field, path.relative_to(self.workspace_dir).as_posix())

                metadata = read_gopro_metadata(path)
                if metadata.serial_number and source.serial_number is None:
                    source.serial_number = metadata.serial_number
                    source.casn_source = "mp4:gpmf"
                if metadata.model and source.model is None:
                    source.model = metadata.model
                if metadata.firmware and source.firmware is None:
                    source.firmware = metadata.firmware

        if save:
            self.save(sources)
        return sources
