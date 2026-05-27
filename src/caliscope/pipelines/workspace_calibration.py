"""Automated non-GUI calibration pipeline for a Caliscope workspace.

Run with::

    python -m caliscope.pipelines.workspace_calibration \
        --workspace /path/to/workspace \
        --intrinsics-library /path/to/intrinsics_library.toml

The pipeline mirrors the GUI workflow without importing Qt: discover cameras,
reuse/adapt intrinsics, synchronize the extrinsic recording by audio, extract
Charuco points, run bootstrap plus two-stage bundle adjustment, and persist the
resulting camera array/capture volume.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rtoml

from caliscope.api import calibrate_intrinsics, extract_image_points
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationReport
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.charuco import Charuco
from caliscope.core.point_data import ImagePoints
from caliscope.core.process_synchronized_recording import process_synchronized_recording
from caliscope.persistence import _safe_write_csv, _safe_write_toml
from caliscope.recording.audio_sync import load_sync_summary, synchronize_recording_timeline
from caliscope.recording.gopro_metadata import read_gopro_metadata
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
from caliscope.recording.video_utils import read_video_properties
from caliscope.repositories.intrinsic_report_repository import IntrinsicReportRepository
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceCalibrationConfig:
    workspace: Path
    intrinsics_library: Path
    extrinsic_frame_step: int = 5
    intrinsic_frame_step: int = 5
    resume: bool = True
    reuse_existing_sync: bool = False
    reuse_image_points: bool = False
    force_sync: bool = False
    force_image_points: bool = False
    force_capture_volume: bool = False
    source_cam_id_fallback: bool = True
    read_metadata: bool = False
    calibrate_missing: bool = True
    parallel: bool = True
    workers: int | None = None
    opencv_threads: int | None = None
    show_progress: bool = True
    filter_percentile: float = 2.5
    max_nfev: int = 1000
    scipy_verbose: int = 0
    align_to_object: bool = False
    plan_only: bool = False
    allow_partial_extrinsics: bool = False


def run_workspace_calibration(config: WorkspaceCalibrationConfig) -> dict[str, Any]:
    """Run the full non-GUI workspace calibration pipeline."""
    workspace = config.workspace.expanduser().resolve()
    intrinsics_library = config.intrinsics_library.expanduser().resolve()
    _validate_config(config, workspace, intrinsics_library)

    intrinsic_dir = workspace / "calibration" / "intrinsic"
    extrinsic_dir = workspace / "calibration" / "extrinsic"
    tracker_dir = extrinsic_dir / "CHARUCO"
    capture_volume_dir = extrinsic_dir / "capture_volume"
    reports_dir = intrinsic_dir / "reports"

    intrinsic_videos = _discover_camera_videos(intrinsic_dir)
    extrinsic_videos = _discover_camera_videos(extrinsic_dir)
    if set(intrinsic_videos) != set(extrinsic_videos):
        raise ValueError(
            "Intrinsic and extrinsic camera sets differ: "
            f"intrinsic={sorted(intrinsic_videos)}, extrinsic={sorted(extrinsic_videos)}"
        )

    cam_ids = sorted(extrinsic_videos)
    logger.info("Discovered %d cameras: %s", len(cam_ids), cam_ids)
    runtime_report = _configure_runtime(config, camera_count=len(cam_ids))

    charuco = _load_extrinsic_charuco(workspace)
    report_repo = IntrinsicReportRepository(reports_dir)
    camera_array, metadata_report = _build_camera_array(
        workspace=workspace,
        intrinsic_videos=intrinsic_videos,
        extrinsic_videos=extrinsic_videos,
        read_metadata=config.read_metadata,
        show_progress=config.show_progress,
    )

    profiles = _load_intrinsics_profiles(intrinsics_library)
    intrinsic_matches = _apply_intrinsics_profiles(
        camera_array=camera_array,
        profiles=profiles,
        library_path=intrinsics_library,
        source_cam_id_fallback=config.source_cam_id_fallback,
        report_repo=report_repo,
        persist_reports=not config.plan_only,
    )

    intrinsic_recalibrations: list[dict[str, Any]] = []
    if not camera_array.all_intrinsics_calibrated():
        if config.plan_only:
            missing = [cid for cid, cam in camera_array.cameras.items() if cam.matrix is None or cam.distortions is None]
            intrinsic_recalibrations = [{"cam_id": cam_id, "status": "would_calibrate"} for cam_id in missing]
        elif not config.calibrate_missing:
            missing = [cid for cid, cam in camera_array.cameras.items() if cam.matrix is None or cam.distortions is None]
            raise ValueError(f"Cameras {missing} still lack intrinsics and --no-calibrate-missing was used")
        else:
            intrinsic_recalibrations = _calibrate_missing_intrinsics(
                camera_array=camera_array,
                intrinsic_videos=intrinsic_videos,
                charuco=charuco,
                frame_step=config.intrinsic_frame_step,
                report_repo=report_repo,
            )

    if not config.plan_only and not camera_array.all_intrinsics_calibrated():
        missing = [cid for cid, cam in camera_array.cameras.items() if cam.matrix is None or cam.distortions is None]
        raise ValueError(f"Cameras {missing} still lack intrinsics after reuse/recalibration")

    reuse_sync = not config.force_sync and (config.resume or config.reuse_existing_sync)
    reuse_points = not config.force_image_points and (config.resume or config.reuse_image_points)
    reuse_capture_volume = not config.force_capture_volume and config.resume

    stage_plan = _build_stage_plan(
        extrinsic_dir=extrinsic_dir,
        tracker_dir=tracker_dir,
        capture_volume_dir=capture_volume_dir,
        cam_ids=cam_ids,
        frame_step=config.extrinsic_frame_step,
        reuse_sync=reuse_sync,
        reuse_points=reuse_points,
        require_image_points_manifest=not config.reuse_image_points,
        reuse_capture_volume=reuse_capture_volume,
        allow_partial_extrinsics=config.allow_partial_extrinsics,
    )
    _log_stage_plan(stage_plan)
    if config.plan_only:
        return {
            "pipeline_schema_version": 1,
            "workspace": str(workspace),
            "intrinsics_library": str(intrinsics_library),
            "runtime": runtime_report,
            "camera_count": len(cam_ids),
            "cam_ids": cam_ids,
            "metadata": metadata_report,
            "intrinsics": {
                "profile_count": len(profiles),
                "matches": intrinsic_matches,
                "recalibrations": intrinsic_recalibrations,
            },
            "stage_plan": stage_plan,
        }

    sync_report = _synchronize_extrinsic_recording(extrinsic_dir, cam_ids, reuse_existing=reuse_sync)
    image_points = _extract_or_load_extrinsic_points(
        extrinsic_dir=extrinsic_dir,
        tracker_dir=tracker_dir,
        cameras=camera_array.cameras,
        charuco=charuco,
        frame_step=config.extrinsic_frame_step,
        reuse_existing=reuse_points,
        require_manifest=not config.reuse_image_points,
        parallel=config.parallel,
        max_workers=runtime_report["workers"],
        show_progress=config.show_progress,
    )

    capture_volume_reused = False
    capture_report, capture_volume = (
        _load_completed_capture_volume(
            capture_volume_dir,
            allow_partial_extrinsics=config.allow_partial_extrinsics,
        )
        if reuse_capture_volume
        else (None, None)
    )
    if capture_volume is not None and capture_report is not None:
        capture_volume_reused = True
        logger.info("Reusing completed capture volume: %s", capture_volume_dir)
    else:
        capture_report, capture_volume = _calibrate_capture_volume(
            image_points=image_points,
            camera_array=camera_array,
            filter_percentile=config.filter_percentile,
            max_nfev=config.max_nfev,
            scipy_verbose=config.scipy_verbose,
            align_to_object=config.align_to_object,
            show_progress=config.show_progress,
            allow_partial_extrinsics=config.allow_partial_extrinsics,
        )

        capture_volume.save(capture_volume_dir)
        _safe_write_csv(
            capture_volume.reprojection_report.raw_errors,
            capture_volume_dir / "reprojection_errors.csv",
            index=False,
        )

    capture_volume.camera_array.to_toml(workspace / "camera_array.toml")
    capture_volume.camera_array.to_aniposelib_toml(workspace / "camera_array_aniposelib.toml")

    run_report = {
        "pipeline_schema_version": 1,
        "workspace": str(workspace),
        "intrinsics_library": str(intrinsics_library),
        "runtime": runtime_report,
        "stage_plan": stage_plan,
        "resume": {
            "enabled": config.resume,
            "sync_reused": sync_report.get("mode") == "reused",
            "image_points_reused": bool(getattr(image_points, "_resumed_from_cache", False)),
            "capture_volume_reused": capture_volume_reused,
        },
        "camera_count": len(cam_ids),
        "cam_ids": cam_ids,
        "metadata": metadata_report,
        "intrinsics": {
            "profile_count": len(profiles),
            "matches": intrinsic_matches,
            "recalibrations": intrinsic_recalibrations,
        },
        "sync": sync_report,
        "extrinsic_points": _summarize_image_points(image_points),
        "capture_volume": capture_report,
        "outputs": {
            "camera_array": str(workspace / "camera_array.toml"),
            "camera_array_aniposelib": str(workspace / "camera_array_aniposelib.toml"),
            "image_points": str(tracker_dir / "image_points.csv"),
            "capture_volume_dir": str(capture_volume_dir),
            "reprojection_errors": str(capture_volume_dir / "reprojection_errors.csv"),
            "run_report": str(capture_volume_dir / "calibration_report.toml"),
        },
    }
    _safe_write_toml(_toml_clean(run_report), capture_volume_dir / "calibration_report.toml")
    logger.info("Calibration complete. Overall reprojection RMSE: %.4f px", capture_report["final_rmse"])
    return run_report

def _validate_config(config: WorkspaceCalibrationConfig, workspace: Path, intrinsics_library: Path) -> None:
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    if not intrinsics_library.exists():
        raise FileNotFoundError(f"Intrinsics library not found: {intrinsics_library}")
    if config.extrinsic_frame_step < 1:
        raise ValueError("extrinsic_frame_step must be >= 1")
    if config.intrinsic_frame_step < 1:
        raise ValueError("intrinsic_frame_step must be >= 1")
    if not (0 < config.filter_percentile <= 100):
        raise ValueError("filter_percentile must be in (0, 100]")
    if config.max_nfev < 1:
        raise ValueError("max_nfev must be >= 1")
    if config.workers is not None and config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.opencv_threads is not None and config.opencv_threads < 1:
        raise ValueError("opencv_threads must be >= 1")
    if config.scipy_verbose not in (0, 1, 2):
        raise ValueError("scipy_verbose must be 0, 1, or 2")


def _configure_runtime(config: WorkspaceCalibrationConfig, *, camera_count: int) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    workers = _resolve_workers(config, camera_count)
    if config.opencv_threads is not None:
        opencv_threads = config.opencv_threads
    elif config.parallel:
        opencv_threads = max(1, cpu_count // max(1, workers))
    else:
        opencv_threads = cpu_count

    cv2_cuda_devices: int | None = None
    cv2_cuda_enabled = False
    actual_opencv_threads: int | None = None
    try:
        import cv2

        cv2.setNumThreads(opencv_threads)
        actual_opencv_threads = int(cv2.getNumThreads())
        if hasattr(cv2, "cuda"):
            cv2_cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
            cv2_cuda_enabled = cv2_cuda_devices > 0
    except Exception as e:
        logger.warning("Could not configure OpenCV threading/CUDA introspection: %s", e)

    report = {
        "cpu_count": cpu_count,
        "parallel": config.parallel,
        "workers": workers,
        "opencv_threads_requested": opencv_threads,
        "opencv_threads_actual": actual_opencv_threads,
        "cv2_cuda_devices": cv2_cuda_devices,
        "cv2_cuda_enabled": cv2_cuda_enabled,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    logger.info(
        "Runtime: cpus=%s, workers=%s, OpenCV threads=%s, OpenCV CUDA devices=%s",
        cpu_count,
        workers,
        actual_opencv_threads,
        cv2_cuda_devices,
    )
    if not cv2_cuda_enabled:
        logger.info("OpenCV CUDA is unavailable; Charuco detection and bundle adjustment will run on CPU.")
    return report


def _resolve_workers(config: WorkspaceCalibrationConfig, camera_count: int) -> int:
    if not config.parallel:
        return 1
    requested = config.workers if config.workers is not None else camera_count
    return max(1, min(camera_count, requested))


def _discover_camera_videos(directory: Path) -> dict[int, Path]:
    videos: dict[int, Path] = {}
    for path in sorted(directory.glob("cam_*.mp4")):
        cam_id = _cam_id_from_name(path.stem)
        if cam_id is not None:
            videos[cam_id] = path
    if not videos:
        raise FileNotFoundError(f"No cam_N.mp4 videos found in {directory}")
    return videos


def _cam_id_from_name(name: str) -> int | None:
    if not name.startswith("cam_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def _load_extrinsic_charuco(workspace: Path) -> Charuco:
    targets_dir = workspace / "calibration" / "targets"
    extrinsic_path = targets_dir / "extrinsic_charuco.toml"
    if not extrinsic_path.exists():
        extrinsic_path = targets_dir / "intrinsic_charuco.toml"
    return Charuco.from_toml(extrinsic_path)


def _load_camera_name_mapping(workspace: Path) -> dict[int, dict[str, str]]:
    path = workspace / "camera_name_mapping.csv"
    if not path.exists():
        return {}

    mapping: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cam_id = _cam_id_from_name(row.get("camera", ""))
            if cam_id is not None:
                mapping[cam_id] = row
    return mapping


def _build_camera_array(
    *,
    workspace: Path,
    intrinsic_videos: dict[int, Path],
    extrinsic_videos: dict[int, Path],
    read_metadata: bool,
    show_progress: bool,
) -> tuple[CameraArray, list[dict[str, Any]]]:
    mapping = _load_camera_name_mapping(workspace)
    existing = _load_existing_camera_array(workspace / "camera_array.toml")
    cameras: dict[int, CameraData] = {}
    metadata_report: list[dict[str, Any]] = []
    items = list(sorted(extrinsic_videos.items()))
    progress_bar = _make_progress_bar(show_progress, total=len(items), desc="Camera metadata", unit="cam")

    try:
        for cam_id, extrinsic_path in items:
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"cam_{cam_id}")
            logger.info("Loading camera metadata for cam_%s", cam_id)
            props = read_video_properties(extrinsic_path)
            old = existing.cameras.get(cam_id) if existing is not None else None
            row = mapping.get(cam_id, {})

            camera = CameraData(
                cam_id=cam_id,
                size=props["size"],
                rotation_count=old.rotation_count if old is not None else 0,
                exposure=old.exposure if old is not None else None,
                ignore=old.ignore if old is not None else False,
                label=old.label if old is not None else None,
                original_filename=row.get("original_filename") or (old.original_filename if old is not None else None),
                intrinsic_video=_workspace_relative(workspace, intrinsic_videos[cam_id]),
                extrinsic_video=_workspace_relative(workspace, extrinsic_path),
                serial_number=old.serial_number if old is not None else None,
                model=old.model if old is not None else None,
                fisheye=old.fisheye if old is not None else False,
            )

            metadata_source = _metadata_source_path(workspace, row, extrinsic_path)
            metadata_status: dict[str, Any] = {
                "cam_id": cam_id,
                "path": str(metadata_source),
                "serial_number": camera.serial_number,
                "model": camera.model,
                "status": "preserved" if camera.serial_number else "not_read",
            }
            if read_metadata:
                try:
                    logger.info("Reading GoPro metadata for cam_%s from %s", cam_id, metadata_source)
                    metadata = read_gopro_metadata(metadata_source)
                    camera.serial_number = metadata.serial_number or camera.serial_number
                    camera.model = metadata.model or camera.model
                    metadata_status.update(
                        {
                            "serial_number": camera.serial_number,
                            "model": camera.model,
                            "firmware": metadata.firmware,
                            "status": "found" if metadata.serial_number else "missing_serial",
                        }
                    )
                except Exception as e:
                    metadata_status.update({"status": "error", "reason": str(e)})

            cameras[cam_id] = camera
            metadata_report.append(metadata_status)
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return CameraArray(cameras), metadata_report


def _load_existing_camera_array(path: Path) -> CameraArray | None:
    if not path.exists():
        return None
    try:
        return CameraArray.from_toml(path)
    except Exception as e:
        logger.warning("Could not load existing camera array %s: %s", path, e)
        return None


def _metadata_source_path(workspace: Path, row: dict[str, str], fallback: Path) -> Path:
    raw = row.get("video_path")
    if raw:
        candidate = workspace / raw
        if candidate.exists():
            return candidate
    return fallback


def _workspace_relative(workspace: Path, path: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def _load_intrinsics_profiles(library_path: Path) -> dict[str, dict[str, Any]]:
    if library_path.is_dir():
        workspace_camera_array = library_path / "camera_array.toml"
        if workspace_camera_array.exists():
            return _load_intrinsics_profiles_from_camera_array(workspace_camera_array)

        profiles: dict[str, dict[str, Any]] = {}
        for path in sorted(library_path.glob("*.toml")):
            profiles.update(_load_intrinsics_profiles_from_file(path))
        return profiles

    return _load_intrinsics_profiles_from_file(library_path)


def _load_intrinsics_profiles_from_file(path: Path) -> dict[str, dict[str, Any]]:
    data = rtoml.load(path)
    if "cameras" in data:
        return _load_intrinsics_profiles_from_camera_array(path)

    raw_profiles = data.get("profiles", data)
    profiles: dict[str, dict[str, Any]] = {}
    for key, value in raw_profiles.items():
        if not isinstance(value, dict):
            continue
        if not all(value.get(field) is not None for field in ("size", "matrix", "distortions")):
            continue
        serial = str(value.get("serial") or value.get("serial_number") or key)
        profile = dict(value)
        profile.setdefault("serial", serial)
        profile.setdefault("source", str(path))
        profiles[serial] = profile
    return profiles


def _load_intrinsics_profiles_from_camera_array(path: Path) -> dict[str, dict[str, Any]]:
    camera_array = CameraArray.from_toml(path)
    profiles: dict[str, dict[str, Any]] = {}
    for cam_id, camera in camera_array.cameras.items():
        if camera.serial_number is None or camera.matrix is None or camera.distortions is None:
            continue
        profiles[camera.serial_number] = {
            "serial": camera.serial_number,
            "size": list(camera.size),
            "fisheye": camera.fisheye,
            "matrix": camera.matrix.tolist(),
            "distortions": camera.distortions.ravel().tolist(),
            "rmse": camera.error,
            "grid_count": camera.grid_count,
            "source": str(path),
            "source_cam_id": cam_id,
        }
    return profiles


def _apply_intrinsics_profiles(
    *,
    camera_array: CameraArray,
    profiles: dict[str, dict[str, Any]],
    library_path: Path,
    source_cam_id_fallback: bool,
    report_repo: IntrinsicReportRepository,
    persist_reports: bool = True,
) -> list[dict[str, Any]]:
    by_source_cam_id = _profiles_by_source_cam_id(profiles)
    results: list[dict[str, Any]] = []

    for cam_id, camera in sorted(camera_array.cameras.items()):
        profile, method, profile_serial = _select_intrinsics_profile(camera, profiles, by_source_cam_id, source_cam_id_fallback)
        if profile is None:
            reason = "missing camera serial and no source_cam_id fallback"
            if camera.serial_number is not None:
                reason = "no matching serial or source_cam_id profile"
            results.append(_intrinsics_result(camera, "skipped", reason=reason))
            continue

        try:
            matrix, distortions, adaptation = _adapt_profile_intrinsics(profile, camera.size)
        except ValueError as e:
            results.append(
                _intrinsics_result(
                    camera,
                    "skipped",
                    method=method,
                    profile=profile,
                    profile_serial=profile_serial,
                    reason=str(e),
                )
            )
            continue

        camera.matrix = matrix
        camera.distortions = distortions
        camera.fisheye = bool(profile.get("fisheye", False))
        camera.error = _optional_float(profile.get("rmse", profile.get("error")))
        camera.grid_count = _optional_int(profile.get("frames_used", profile.get("grid_count")))
        camera.intrinsics_source = f"{library_path}#{profile_serial}:{adaptation}"

        if persist_reports:
            report_repo.save(cam_id, _intrinsic_report_from_profile(profile))
        results.append(
            _intrinsics_result(
                camera,
                "matched",
                method=method,
                profile=profile,
                profile_serial=profile_serial,
                adaptation=adaptation,
            )
        )

    matched = [item["cam_id"] for item in results if item["status"] == "matched"]
    logger.info("Applied library intrinsics to %d/%d cameras", len(matched), len(camera_array.cameras))
    return results


def _profiles_by_source_cam_id(profiles: dict[str, dict[str, Any]]) -> dict[int, tuple[str, dict[str, Any]]]:
    by_source: dict[int, tuple[str, dict[str, Any]]] = {}
    for serial, profile in profiles.items():
        source_cam_id = _optional_int(profile.get("source_cam_id"))
        if source_cam_id is not None and source_cam_id not in by_source:
            by_source[source_cam_id] = (serial, profile)
    return by_source


def _select_intrinsics_profile(
    camera: CameraData,
    profiles: dict[str, dict[str, Any]],
    by_source_cam_id: dict[int, tuple[str, dict[str, Any]]],
    source_cam_id_fallback: bool,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if camera.serial_number is not None and camera.serial_number in profiles:
        return profiles[camera.serial_number], "serial", camera.serial_number
    if source_cam_id_fallback and camera.cam_id in by_source_cam_id:
        serial, profile = by_source_cam_id[camera.cam_id]
        return profile, "source_cam_id", serial
    return None, None, None


def _adapt_profile_intrinsics(profile: dict[str, Any], target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    matrix, distortions = _validated_intrinsics_arrays(profile)
    source_size = tuple(int(v) for v in profile["size"])
    target_w, target_h = target_size
    source_w, source_h = source_size

    if source_size == target_size:
        return matrix, distortions, "exact_size"

    if _same_aspect(source_size, target_size):
        sx = target_w / source_w
        sy = target_h / source_h
        return _scale_intrinsics(matrix, sx, sy), distortions, f"scaled:{sx:.8g}x,{sy:.8g}y"

    if _same_aspect((source_h, source_w), target_size):
        sx = target_w / source_h
        sy = target_h / source_w
        rotated_matrix = _rotate_intrinsics_90_ccw(matrix, source_w)
        rotated_distortions = _rotate_distortions_90_ccw(distortions, bool(profile.get("fisheye", False)))
        adaptation = f"rotated_90ccw_scaled:{sx:.8g}x,{sy:.8g}y"
        return _scale_intrinsics(rotated_matrix, sx, sy), rotated_distortions, adaptation

    raise ValueError(f"profile size {source_size} is not compatible with camera size {target_size}")


def _same_aspect(source_size: tuple[int, int], target_size: tuple[int, int]) -> bool:
    source_w, source_h = source_size
    target_w, target_h = target_size
    return abs((source_w / source_h) - (target_w / target_h)) < 1e-6


def _scale_intrinsics(matrix: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = matrix.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def _rotate_intrinsics_90_ccw(matrix: np.ndarray, source_width: int) -> np.ndarray:
    rotated = np.eye(3, dtype=np.float64)
    rotated[0, 0] = matrix[1, 1]
    rotated[1, 1] = matrix[0, 0]
    rotated[0, 2] = matrix[1, 2]
    rotated[1, 2] = source_width - matrix[0, 2]
    return rotated


def _rotate_distortions_90_ccw(distortions: np.ndarray, fisheye: bool) -> np.ndarray:
    if fisheye or distortions.size != 5:
        return distortions.copy()
    k1, k2, p1, p2, k3 = distortions.tolist()
    return np.array([k1, k2, -p2, p1, k3], dtype=np.float64)


def _validated_intrinsics_arrays(profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(profile["matrix"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"invalid matrix shape {matrix.shape}")

    distortions = np.asarray(profile["distortions"], dtype=np.float64).ravel()
    expected = 4 if bool(profile.get("fisheye", False)) else 5
    if distortions.shape != (expected,):
        raise ValueError(f"invalid distortion count {distortions.size}; expected {expected}")
    return matrix, distortions


def _intrinsics_result(
    camera: CameraData,
    status: str,
    *,
    method: str | None = None,
    profile: dict[str, Any] | None = None,
    profile_serial: str | None = None,
    adaptation: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "cam_id": camera.cam_id,
        "status": status,
        "method": method,
        "serial_number": camera.serial_number,
        "profile_serial": profile_serial,
        "source_cam_id": _optional_int(profile.get("source_cam_id")) if profile is not None else None,
        "camera_size": list(camera.size),
        "profile_size": profile.get("size") if profile is not None else None,
        "adaptation": adaptation,
        "rmse": _optional_float(profile.get("rmse", profile.get("error"))) if profile is not None else None,
        "reason": reason,
    }


def _intrinsic_report_from_profile(profile: dict[str, Any]) -> IntrinsicCalibrationReport:
    return IntrinsicCalibrationReport(
        rmse=float(profile.get("rmse", profile.get("error", 0.0)) or 0.0),
        frames_used=int(profile.get("frames_used", profile.get("grid_count", 0)) or 0),
        coverage_fraction=float(profile.get("coverage_fraction", 0.0) or 0.0),
        edge_coverage_fraction=float(profile.get("edge_coverage_fraction", 0.0) or 0.0),
        corner_coverage_fraction=float(profile.get("corner_coverage_fraction", 0.0) or 0.0),
        orientation_sufficient=bool(profile.get("orientation_sufficient", False)),
        orientation_count=int(profile.get("orientation_count", 0) or 0),
        selected_frames=tuple(int(v) for v in profile.get("selected_frames", ())),
    )


def _calibrate_missing_intrinsics(
    *,
    camera_array: CameraArray,
    intrinsic_videos: dict[int, Path],
    charuco: Charuco,
    frame_step: int,
    report_repo: IntrinsicReportRepository,
) -> list[dict[str, Any]]:
    tracker = CharucoTracker(charuco)
    results: list[dict[str, Any]] = []
    for cam_id, camera in sorted(camera_array.cameras.items()):
        if camera.matrix is not None and camera.distortions is not None:
            continue
        logger.info("Calibrating missing intrinsics for cam_%s from %s", cam_id, intrinsic_videos[cam_id])
        points = extract_image_points(intrinsic_videos[cam_id], cam_id, tracker, frame_step=frame_step)
        output = calibrate_intrinsics(points, camera)
        output.camera.intrinsics_source = f"intrinsic_video:{intrinsic_videos[cam_id]}"
        camera_array.cameras[cam_id] = output.camera
        report_repo.save(cam_id, output.report)
        results.append(
            {
                "cam_id": cam_id,
                "status": "calibrated",
                "frame_step": frame_step,
                "rmse": output.report.rmse,
                "frames_used": output.report.frames_used,
                "coverage_fraction": output.report.coverage_fraction,
            }
        )
    return results


def _synchronize_extrinsic_recording(extrinsic_dir: Path, cam_ids: list[int], *, reuse_existing: bool) -> dict[str, Any]:
    if reuse_existing:
        existing = load_sync_summary(extrinsic_dir)
        timestamps_path = extrinsic_dir / "timestamps.csv"
        if existing is not None and timestamps_path.exists():
            logger.info("Reusing existing audio synchronization: %s", existing.summary_path)
            return {"mode": "reused", **existing.to_dict()}

    logger.info("Synchronizing extrinsic recording by audio")
    sync_info = synchronize_recording_timeline(extrinsic_dir, cam_ids)
    return {"mode": "computed", **sync_info.to_dict()}


def _build_stage_plan(
    *,
    extrinsic_dir: Path,
    tracker_dir: Path,
    capture_volume_dir: Path,
    cam_ids: list[int],
    frame_step: int,
    reuse_sync: bool,
    reuse_points: bool,
    require_image_points_manifest: bool,
    reuse_capture_volume: bool,
    allow_partial_extrinsics: bool,
) -> dict[str, Any]:
    sync_available = load_sync_summary(extrinsic_dir) is not None and (extrinsic_dir / "timestamps.csv").exists()
    image_points_path = tracker_dir / "image_points.csv"
    image_points_available = image_points_path.exists()
    manifest_matches = False
    if image_points_available:
        manifest_matches = _image_points_manifest_matches(
            image_points_path,
            extrinsic_dir=extrinsic_dir,
            cam_ids=cam_ids,
            frame_step=frame_step,
        )
    image_points_reusable = image_points_available and reuse_points and (
        manifest_matches or not require_image_points_manifest
    )
    capture_available, capture_complete, capture_unposed = _completed_capture_volume_status(capture_volume_dir)
    capture_reusable = reuse_capture_volume and capture_available and (capture_complete or allow_partial_extrinsics)

    return {
        "sync": {
            "action": "reuse" if reuse_sync and sync_available else "compute",
            "available": sync_available,
            "timestamps": str(extrinsic_dir / "timestamps.csv"),
            "summary": str(extrinsic_dir / "sync_offsets.toml"),
        },
        "image_points": {
            "action": "reuse" if image_points_reusable else "compute",
            "available": image_points_available,
            "manifest_matches": manifest_matches,
            "requires_manifest": require_image_points_manifest,
            "path": str(image_points_path),
            "manifest": str(_image_points_manifest_path(image_points_path)),
        },
        "capture_volume": {
            "action": "reuse" if capture_reusable else "compute",
            "available": capture_available,
            "complete_extrinsics": capture_complete,
            "unposed_cameras": capture_unposed,
            "path": str(capture_volume_dir),
            "report": str(capture_volume_dir / "calibration_report.toml"),
        },
    }


def _log_stage_plan(plan: dict[str, Any]) -> None:
    logger.info(
        "Stage plan: sync=%s, image_points=%s, capture_volume=%s",
        plan["sync"]["action"],
        plan["image_points"]["action"],
        plan["capture_volume"]["action"],
    )


def _completed_capture_volume_status(capture_volume_dir: Path) -> tuple[bool, bool, list[int]]:
    report_path = capture_volume_dir / "calibration_report.toml"
    required_files = [
        capture_volume_dir / "camera_array.toml",
        capture_volume_dir / "image_points.csv",
        capture_volume_dir / "world_points.csv",
    ]
    if not report_path.exists() or not all(path.exists() for path in required_files):
        return False, False, []
    try:
        data = rtoml.load(report_path)
        camera_array = CameraArray.from_toml(capture_volume_dir / "camera_array.toml")
    except Exception:
        return False, False, []
    if int(data.get("pipeline_schema_version", 0)) != 1:
        return False, False, []
    return True, camera_array.all_extrinsics_calibrated(), sorted(camera_array.unposed_cameras)


def _extract_or_load_extrinsic_points(
    *,
    extrinsic_dir: Path,
    tracker_dir: Path,
    cameras: dict[int, CameraData],
    charuco: Charuco,
    frame_step: int,
    reuse_existing: bool,
    require_manifest: bool,
    parallel: bool,
    max_workers: int,
    show_progress: bool,
) -> ImagePoints:
    image_points_path = tracker_dir / "image_points.csv"
    if reuse_existing and image_points_path.exists():
        if not require_manifest or _image_points_manifest_matches(
            image_points_path,
            extrinsic_dir=extrinsic_dir,
            cam_ids=sorted(cameras),
            frame_step=frame_step,
        ):
            logger.info("Reusing existing extrinsic image points: %s", image_points_path)
            image_points = ImagePoints.from_csv(image_points_path)
            setattr(image_points, "_resumed_from_cache", True)
            return image_points
        logger.info("Existing image_points.csv has no matching pipeline manifest; recomputing")

    synced_timestamps = SynchronizedTimestamps.load(extrinsic_dir, sorted(cameras))
    progress_interval = max(1, len(synced_timestamps.sync_indices[::frame_step]) // 20)
    progress_bar = _make_progress_bar(
        show_progress,
        total=len(synced_timestamps.sync_indices[::frame_step]),
        desc="Charuco extraction",
        unit="sync",
    )
    last_progress = 0

    def on_progress(current: int, total: int) -> None:
        nonlocal last_progress
        if progress_bar is not None:
            progress_bar.update(max(0, current - last_progress))
            last_progress = current
        elif current == total or current == 1 or current % progress_interval == 0:
            logger.info("Extrinsic point extraction progress: %d/%d sync indices", current, total)

    logger.info(
        "Extracting synchronized Charuco points every %d sync index(es) with %d worker(s)",
        frame_step,
        max_workers,
    )
    try:
        image_points = process_synchronized_recording(
            recording_dir=extrinsic_dir,
            cameras=cameras,
            tracker=CharucoTracker(charuco),
            synced_timestamps=synced_timestamps,
            subsample=frame_step,
            parallel=parallel,
            max_workers=max_workers,
            on_progress=on_progress,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if image_points.df.empty:
        raise ValueError("No extrinsic Charuco points were detected")
    image_points.to_csv(image_points_path)
    _safe_write_toml(
        _toml_clean(
            _image_points_manifest(
                image_points,
                extrinsic_dir=extrinsic_dir,
                cam_ids=sorted(cameras),
                frame_step=frame_step,
            )
        ),
        _image_points_manifest_path(image_points_path),
    )
    setattr(image_points, "_resumed_from_cache", False)
    return image_points


def _image_points_manifest_path(image_points_path: Path) -> Path:
    return image_points_path.with_name(f"{image_points_path.stem}.meta.toml")


def _image_points_manifest(
    image_points: ImagePoints,
    *,
    extrinsic_dir: Path,
    cam_ids: list[int],
    frame_step: int,
) -> dict[str, Any]:
    timestamps_path = extrinsic_dir / "timestamps.csv"
    summary = _summarize_image_points(image_points)
    return {
        "schema_version": 1,
        "stage": "extrinsic_image_points",
        "cam_ids": cam_ids,
        "frame_step": frame_step,
        "timestamps_path": str(timestamps_path),
        "timestamps_mtime_ns": timestamps_path.stat().st_mtime_ns if timestamps_path.exists() else None,
        "summary": summary,
    }


def _image_points_manifest_matches(
    image_points_path: Path,
    *,
    extrinsic_dir: Path,
    cam_ids: list[int],
    frame_step: int,
) -> bool:
    manifest_path = _image_points_manifest_path(image_points_path)
    timestamps_path = extrinsic_dir / "timestamps.csv"
    if not manifest_path.exists() or not timestamps_path.exists():
        return False
    try:
        data = rtoml.load(manifest_path)
    except Exception as e:
        logger.warning("Could not read image-points manifest %s: %s", manifest_path, e)
        return False

    return (
        int(data.get("schema_version", 0)) == 1
        and data.get("stage") == "extrinsic_image_points"
        and [int(v) for v in data.get("cam_ids", [])] == cam_ids
        and int(data.get("frame_step", -1)) == frame_step
        and _optional_int(data.get("timestamps_mtime_ns")) == timestamps_path.stat().st_mtime_ns
    )


def _calibrate_capture_volume(
    *,
    image_points: ImagePoints,
    camera_array: CameraArray,
    filter_percentile: float,
    max_nfev: int,
    scipy_verbose: int,
    align_to_object: bool,
    show_progress: bool,
    allow_partial_extrinsics: bool,
) -> tuple[dict[str, Any], CaptureVolume]:
    step_count = 5 if align_to_object else 4
    progress_bar = _make_progress_bar(show_progress, total=step_count, desc="Capture volume", unit="stage")

    def finish_stage(label: str) -> None:
        if progress_bar is not None:
            progress_bar.set_postfix_str(label)
            progress_bar.update(1)

    try:
        logger.info("Bootstrapping capture volume")
        volume = CaptureVolume.bootstrap(image_points, camera_array)
        if not volume.camera_array.all_extrinsics_calibrated():
            unposed = sorted(volume.camera_array.unposed_cameras)
            message = (
                "Extrinsic bootstrap did not connect all cameras. "
                f"Unposed cameras: {unposed}. "
                "Record additional calibration frames where the ChArUco board is visible "
                "to at least one posed camera and one unposed camera with >=4 detected corners each."
            )
            if not allow_partial_extrinsics:
                raise ValueError(message)
            logger.warning("%s Continuing because allow_partial_extrinsics=True.", message)
        finish_stage("bootstrap")

        bootstrap_report = _summarize_reprojection_report(volume)
        logger.info("Initial reprojection RMSE: %.4f px", bootstrap_report["overall_rmse"])

        first = volume.optimize(max_nfev=max_nfev, verbose=scipy_verbose, strict=False)
        finish_stage("first BA")
        first_report = _summarize_reprojection_report(first)
        logger.info("First bundle adjustment RMSE: %.4f px", first_report["overall_rmse"])

        filtered = first.filter_by_percentile_error(filter_percentile) if filter_percentile > 0 else first
        finish_stage("filter")
        filtered_report = _summarize_reprojection_report(filtered)
        logger.info("After %.2f%% reprojection filtering RMSE: %.4f px", filter_percentile, filtered_report["overall_rmse"])

        final = filtered.optimize(max_nfev=max_nfev, verbose=scipy_verbose, strict=False)
        finish_stage("second BA")
        alignment_sync_index: int | None = None
        if align_to_object:
            alignment_sync_index = _choose_alignment_sync_index(final)
            if alignment_sync_index is not None:
                logger.info("Aligning capture volume to object at sync_index %d", alignment_sync_index)
                final = final.align_to_object(alignment_sync_index)
            else:
                logger.warning("Could not find a suitable sync_index for object alignment")
            finish_stage("alignment")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    final_report = _summarize_reprojection_report(final)
    return (
        {
            "bootstrap": bootstrap_report,
            "first_optimization": first_report,
            "filtered": filtered_report,
            "final": final_report,
            "final_rmse": final_report["overall_rmse"],
            "filter_percentile": filter_percentile,
            "alignment_sync_index": alignment_sync_index,
            "optimization_status": _optimization_status_dict(final),
        },
        final,
    )


def _load_completed_capture_volume(
    capture_volume_dir: Path,
    *,
    allow_partial_extrinsics: bool,
) -> tuple[dict[str, Any] | None, CaptureVolume | None]:
    report_path = capture_volume_dir / "calibration_report.toml"
    available, complete, unposed = _completed_capture_volume_status(capture_volume_dir)
    if not available:
        return None, None
    if not complete and not allow_partial_extrinsics:
        logger.info("Existing capture volume is partial; recomputing because partial extrinsics are not allowed: %s", unposed)
        return None, None

    try:
        report_data = rtoml.load(report_path)
        capture_volume = CaptureVolume.load(capture_volume_dir)
        capture_report = report_data.get("capture_volume") or {}
        if "final_rmse" not in capture_report:
            final_report = _summarize_reprojection_report(capture_volume)
            capture_report = {"final": final_report, "final_rmse": final_report["overall_rmse"]}
        return capture_report, capture_volume
    except Exception as e:
        logger.warning("Could not reuse completed capture volume %s: %s", capture_volume_dir, e)
        return None, None


def _choose_alignment_sync_index(volume: CaptureVolume) -> int | None:
    img_df = volume.image_points.df
    world_sync_indices = set(int(v) for v in volume.world_points.df["sync_index"].unique())
    candidates = img_df[img_df["sync_index"].isin(world_sync_indices)]
    candidates = candidates.dropna(subset=["obj_loc_x", "obj_loc_y"])
    if candidates.empty:
        return None

    summary = candidates.groupby("sync_index").agg(
        n_points=("point_id", "nunique"),
        n_cameras=("cam_id", "nunique"),
        n_observations=("point_id", "size"),
    )
    summary = summary[summary["n_points"] >= 3]
    if summary.empty:
        return None
    summary = summary.sort_values(["n_points", "n_cameras", "n_observations"], ascending=False)
    return int(summary.index[0])


def _summarize_image_points(image_points: ImagePoints) -> dict[str, Any]:
    df = image_points.df
    by_camera = df.groupby("cam_id").size().to_dict() if not df.empty else {}
    return {
        "rows": len(df),
        "sync_indices": int(df["sync_index"].nunique()) if not df.empty else 0,
        "cameras": int(df["cam_id"].nunique()) if not df.empty else 0,
        "points": int(df["point_id"].nunique()) if not df.empty else 0,
        "observations_by_camera": {int(k): int(v) for k, v in by_camera.items()},
    }


def _summarize_reprojection_report(volume: CaptureVolume) -> dict[str, Any]:
    report = volume.reprojection_report
    return {
        "overall_rmse": float(report.overall_rmse),
        "by_camera": {int(k): float(v) for k, v in report.by_camera.items()},
        "n_observations_matched": int(report.n_observations_matched),
        "n_observations_total": int(report.n_observations_total),
        "n_unmatched_observations": int(report.n_unmatched_observations),
        "unmatched_rate": float(report.unmatched_rate),
        "n_cameras": int(report.n_cameras),
        "n_points": int(report.n_points),
    }


def _optimization_status_dict(volume: CaptureVolume) -> dict[str, Any] | None:
    status = volume.optimization_status
    if status is None:
        return None
    return {
        "converged": status.converged,
        "termination_reason": status.termination_reason,
        "iterations": status.iterations,
        "final_cost": status.final_cost,
    }


def _make_progress_bar(enabled: bool, *, total: int, desc: str, unit: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.info("tqdm is not installed; falling back to log-only progress")
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _toml_clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda item: _toml_sort_key(item[1])):
            cleaned_item = _toml_clean(item)
            if cleaned_item is not None:
                cleaned[str(key)] = cleaned_item
        return cleaned
    if isinstance(value, (list, tuple)):
        cleaned_list = []
        for item in value:
            cleaned_item = _toml_clean(item)
            if cleaned_item is not None:
                cleaned_list.append(cleaned_item)
        return cleaned_list
    return value


def _toml_sort_key(value: Any) -> tuple[int, str]:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return (2, "")
    if isinstance(value, dict):
        return (1, "")
    return (0, "")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a non-GUI Caliscope workspace calibration pipeline.")
    parser.add_argument("--workspace", type=Path, required=True, help="Caliscope workspace root")
    parser.add_argument("--intrinsics-library", type=Path, required=True, help="Intrinsics library TOML/folder/workspace")
    parser.add_argument("--extrinsic-frame-step", type=int, default=5, help="Process every Nth synced frame")
    parser.add_argument("--intrinsic-frame-step", type=int, default=5, help="Process every Nth intrinsic video frame")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from complete pipeline stage outputs when available",
    )
    parser.add_argument(
        "--reuse-existing-sync",
        action="store_true",
        help="Reuse timestamps.csv/sync_offsets.toml even when --no-resume is set",
    )
    parser.add_argument(
        "--reuse-image-points",
        action="store_true",
        help="Reuse existing extrinsic image_points.csv even without a matching pipeline manifest",
    )
    parser.add_argument("--force-sync", action="store_true", help="Recompute audio synchronization")
    parser.add_argument("--force-image-points", action="store_true", help="Recompute extrinsic Charuco image points")
    parser.add_argument("--force-capture-volume", action="store_true", help="Recompute capture volume even if complete")
    parser.add_argument(
        "--source-cam-id-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to profile source_cam_id when serial metadata is missing",
    )
    parser.add_argument(
        "--read-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attempt GoPro serial/model metadata extraction before fallback matching",
    )
    parser.add_argument(
        "--calibrate-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run intrinsic calibration for cameras not covered by the library",
    )
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel per-camera point extraction")
    parser.add_argument("--workers", type=int, default=None, help="Maximum camera worker threads for extraction")
    parser.add_argument("--opencv-threads", type=int, default=None, help="OpenCV internal thread count per worker process")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars for long stages",
    )
    parser.add_argument("--filter-percentile", type=float, default=2.5, help="Worst reprojection error percentile to remove")
    parser.add_argument("--max-nfev", type=int, default=1000, help="Maximum function evaluations per BA pass")
    parser.add_argument("--scipy-verbose", type=int, default=0, choices=(0, 1, 2), help="SciPy least_squares verbosity")
    parser.add_argument("--align-to-object", action="store_true", help="Align final volume to a visible Charuco board frame")
    parser.add_argument("--plan-only", action="store_true", help="Print planned reuse/recompute stages and exit")
    parser.add_argument(
        "--allow-partial-extrinsics",
        action="store_true",
        help="Allow saving a capture volume when some cameras remain unposed",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    config = WorkspaceCalibrationConfig(
        workspace=args.workspace,
        intrinsics_library=args.intrinsics_library,
        extrinsic_frame_step=args.extrinsic_frame_step,
        intrinsic_frame_step=args.intrinsic_frame_step,
        resume=args.resume,
        reuse_existing_sync=args.reuse_existing_sync,
        reuse_image_points=args.reuse_image_points,
        force_sync=args.force_sync,
        force_image_points=args.force_image_points,
        force_capture_volume=args.force_capture_volume,
        source_cam_id_fallback=args.source_cam_id_fallback,
        read_metadata=args.read_metadata,
        calibrate_missing=args.calibrate_missing,
        parallel=not args.no_parallel,
        workers=args.workers,
        opencv_threads=args.opencv_threads,
        show_progress=args.progress,
        filter_percentile=args.filter_percentile,
        max_nfev=args.max_nfev,
        scipy_verbose=args.scipy_verbose,
        align_to_object=args.align_to_object,
        plan_only=args.plan_only,
        allow_partial_extrinsics=args.allow_partial_extrinsics,
    )
    report = run_workspace_calibration(config)
    if args.plan_only:
        print(rtoml.dumps(_toml_clean(report)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
