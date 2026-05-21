"""
Repository layer: persistence gateways for domain objects.

Repositories handle load/save operations to TOML/CSV files. They are the
boundary between domain logic and storage, converting between domain objects
and their serialized representations.
"""

from caliscope.repositories.camera_array_repository import CameraArrayRepository
from caliscope.repositories.camera_sources_repository import CameraSource, CameraSourcesRepository
from caliscope.repositories.calibration_targets_repository import CalibrationTargetsRepository
from caliscope.repositories.project_settings_repository import ProjectSettingsRepository
from caliscope.repositories.capture_volume_repository import CaptureVolumeRepository

__all__ = [
    "CameraArrayRepository",
    "CameraSource",
    "CameraSourcesRepository",
    "CalibrationTargetsRepository",
    "ProjectSettingsRepository",
    "CaptureVolumeRepository",
]
