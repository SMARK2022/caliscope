"""Caliscope 工作区的非 GUI 自动标定管线。

该模块刻意只做“工作流编排”，不把相机模型、ChArUco 检测、BA 优化等核心算法
复制到脚本里。这样既能在集群/远程服务器上运行，又能复用 GUI 已验证过的底层
能力：发现相机、同步外参视频、复用/适配内参、提取 ChArUco 点、初始化外参图、
两轮 bundle adjustment、保存 Caliscope 和 aniposelib 两种输出。

命令行示例::

    python -m caliscope.pipelines.workspace_calibration \
        --workspace /path/to/workspace \
        --intrinsics-library /path/to/intrinsics_library.toml

默认行为强调可恢复性和安全性：已有同步结果和本管线生成的 image_points manifest
会被复用；外参图不完整时默认失败，只有显式传入 ``--allow-partial-extrinsics`` 才会
保存部分相机外参。
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
    """非 GUI 标定管线的全部运行参数。

    这个 dataclass 是 CLI 和 Python API 之间的稳定边界。字段尽量保持扁平，避免
    为了配置再引入多层对象；这样在 VS Code、notebook 或命令行里调用时都容易看懂。

    Attributes:
        workspace: Caliscope 工作区根目录。
        intrinsics_library: 内参库 TOML、内参库目录或可读取的 Caliscope 工作区。
        extrinsic_frame_step: 外参同步视频按多少 sync index 抽帧检测。
        intrinsic_frame_step: 内参缺失时，内参视频按多少帧抽样检测。
        resume: 是否默认复用已完成且有效的阶段产物。
        reuse_existing_sync: 即使关闭 resume，也允许复用已有同步文件。
        reuse_image_points: 允许复用没有 manifest 的旧 image_points.csv。
        force_sync: 强制重做音频同步。
        force_image_points: 强制重做外参 ChArUco 点检测。
        force_capture_volume: 强制重做外参 bootstrap 和 BA。
        source_cam_id_fallback: 缺少序列号时，用内参库 source_cam_id 匹配相机。
        read_metadata: 是否读取 GoPro 大文件元数据；默认关闭以避免启动阶段卡顿。
        calibrate_missing: 内参库无法匹配时是否用内参视频补标定。
        parallel: 是否并行处理每个 sync index 上的多相机帧。
        workers: 相机处理线程上限；None 表示最多使用相机数。
        opencv_threads: OpenCV 每进程内部线程数；None 表示自动按 CPU/worker 均分。
        show_progress: 是否显示 tqdm 进度条。
        filter_percentile: 第一轮 BA 后剔除的最差重投影误差百分比。
        filter_scope: 剔除范围；sync_index 表示按整帧剔除。
        filter_sigma: sync_index 整帧剔除的可选 robust sigma 阈值；None 表示只按百分比剔除。
        max_nfev: 每轮 SciPy least_squares 最大函数评估次数。
        scipy_verbose: SciPy 优化器日志级别。
        align_to_object: 是否把最终坐标系对齐到某一帧 ChArUco 板。
        optitrack_csv: 可选 Motive/OptiTrack CSV；提供时在外参完成后运行 12D 坐标系对齐。
        plan_only: 只打印复用/重算计划，不写入标定结果。
        allow_partial_extrinsics: 允许外参图不完整时继续保存 partial 结果。
    """

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
    filter_scope: str = "per_camera"
    filter_sigma: float | None = None
    max_nfev: int = 1000
    scipy_verbose: int = 0
    align_to_object: bool = False
    optitrack_csv: Path | None = None
    optitrack_alignment_output_dir: Path | None = None
    optitrack_lambda_xy_list: str = "0,0.1,0.2,0.5,1,10,100"
    optitrack_select_lambda: str = "0.2"
    optitrack_offset_min: float = -20.0
    optitrack_offset_max: float = 20.0
    optitrack_coarse_offset_step: float = 0.25
    optitrack_max_world_grid_rmse_m: float = 0.005
    optitrack_min_points_per_fit_frame: int = 15
    optitrack_min_overlap_frames: int = 40
    optitrack_max_coarse_frames: int = 120
    optitrack_top_candidates_to_refine: int = 8
    optitrack_equal_height_maxiter: int = 260
    optitrack_offset_12d_maxiter: int = 500
    optitrack_allow_global_scale: bool = True
    optitrack_test_ratio: float = 0.33
    optitrack_seed: int = 20260521
    optitrack_write_plots: bool = True
    plan_only: bool = False
    allow_partial_extrinsics: bool = False


def run_workspace_calibration(config: WorkspaceCalibrationConfig) -> dict[str, Any]:
    """执行完整的非 GUI 工作区标定流程。

    流程顺序保持和 GUI 一致：先建立相机数组并导入内参，再同步外参视频时间线，
    然后提取同步 ChArUco 点，最后进行外参初始化和两轮 BA。函数返回的字典会被
    写入 ``calibration/extrinsic/capture_volume/calibration_report.toml``，也便于测试
    或 notebook 调用时直接检查。

    Args:
        config: 管线运行配置，包含路径、复用策略、并行参数和 BA 参数。

    Returns:
        可 TOML 序列化的运行报告。plan_only=True 时只包含阶段计划，不会写结果。

    Raises:
        FileNotFoundError: 工作区、内参库或视频目录缺失。
        ValueError: 配置无效、内参不完整或外参图无法连通所有相机。
    """
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
            "optitrack_alignment": _optitrack_alignment_plan(config, extrinsic_dir, capture_volume_dir),
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
            filter_scope=config.filter_scope,
            filter_sigma=config.filter_sigma,
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
    optitrack_alignment_report = _run_optitrack_alignment_stage(config, extrinsic_dir, capture_volume_dir)

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
        "optitrack_alignment": optitrack_alignment_report,
        "outputs": {
            "camera_array": str(workspace / "camera_array.toml"),
            "camera_array_aniposelib": str(workspace / "camera_array_aniposelib.toml"),
            "image_points": str(tracker_dir / "image_points.csv"),
            "capture_volume_dir": str(capture_volume_dir),
            "reprojection_errors": str(capture_volume_dir / "reprojection_errors.csv"),
            "run_report": str(capture_volume_dir / "calibration_report.toml"),
        },
    }
    if optitrack_alignment_report.get("enabled"):
        outputs = optitrack_alignment_report.get("outputs", {})
        run_report["outputs"]["optitrack_alignment_dir"] = optitrack_alignment_report["output_dir"]
        if isinstance(outputs, dict):
            run_report["outputs"].update({f"optitrack_{key}": value for key, value in outputs.items()})
    _safe_write_toml(_toml_clean(run_report), capture_volume_dir / "calibration_report.toml")
    logger.info("Calibration complete. Overall reprojection RMSE: %.4f px", capture_report["final_rmse"])
    return run_report

def _validate_config(config: WorkspaceCalibrationConfig, workspace: Path, intrinsics_library: Path) -> None:
    """校验路径和数值参数，尽早暴露配置错误。

    Args:
        config: 用户传入的运行配置。
        workspace: 已展开和 resolve 后的工作区路径。
        intrinsics_library: 已展开和 resolve 后的内参库路径。

    Raises:
        FileNotFoundError: 必要路径不存在。
        ValueError: 抽帧步长、worker 数或优化参数不在合法范围内。
    """

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
    if config.filter_scope not in {"per_camera", "overall", "sync_index"}:
        raise ValueError("filter_scope must be one of: per_camera, overall, sync_index")
    if config.filter_scope == "sync_index" and config.filter_percentile > 30:
        raise ValueError("sync_index frame filtering is capped at 30%")
    if config.filter_sigma is not None and config.filter_sigma <= 0:
        raise ValueError("filter_sigma must be positive when provided")
    if config.max_nfev < 1:
        raise ValueError("max_nfev must be >= 1")
    if config.workers is not None and config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.opencv_threads is not None and config.opencv_threads < 1:
        raise ValueError("opencv_threads must be >= 1")
    if config.scipy_verbose not in (0, 1, 2):
        raise ValueError("scipy_verbose must be 0, 1, or 2")
    if config.optitrack_csv is not None:
        optitrack_csv = config.optitrack_csv.expanduser().resolve()
        if not optitrack_csv.exists():
            raise FileNotFoundError(f"OptiTrack CSV not found: {optitrack_csv}")
        _parse_float_csv(config.optitrack_lambda_xy_list, "optitrack_lambda_xy_list")
        if config.optitrack_select_lambda not in {"min_test", "min_all"}:
            float(config.optitrack_select_lambda)
        if config.optitrack_offset_min >= config.optitrack_offset_max:
            raise ValueError("optitrack_offset_min must be < optitrack_offset_max")
        if config.optitrack_coarse_offset_step <= 0:
            raise ValueError("optitrack_coarse_offset_step must be positive")
        if config.optitrack_max_world_grid_rmse_m <= 0:
            raise ValueError("optitrack_max_world_grid_rmse_m must be positive")
        if config.optitrack_min_points_per_fit_frame < 3:
            raise ValueError("optitrack_min_points_per_fit_frame must be >= 3")
        if config.optitrack_min_overlap_frames < 3:
            raise ValueError("optitrack_min_overlap_frames must be >= 3")
        if config.optitrack_max_coarse_frames < 1:
            raise ValueError("optitrack_max_coarse_frames must be >= 1")
        if config.optitrack_top_candidates_to_refine < 1:
            raise ValueError("optitrack_top_candidates_to_refine must be >= 1")
        if config.optitrack_equal_height_maxiter < 1:
            raise ValueError("optitrack_equal_height_maxiter must be >= 1")
        if config.optitrack_offset_12d_maxiter < 1:
            raise ValueError("optitrack_offset_12d_maxiter must be >= 1")
        if not (0 <= config.optitrack_test_ratio < 1):
            raise ValueError("optitrack_test_ratio must be in [0, 1)")


def _parse_float_csv(value: str, field_name: str) -> list[float]:
    """解析逗号分隔浮点数配置。"""

    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    return values


def _optitrack_alignment_output_dir(config: WorkspaceCalibrationConfig, extrinsic_dir: Path) -> Path:
    if config.optitrack_alignment_output_dir is not None:
        return config.optitrack_alignment_output_dir.expanduser().resolve()
    return extrinsic_dir / "optitrack_alignment_12d"


def _optitrack_alignment_plan(
    config: WorkspaceCalibrationConfig,
    extrinsic_dir: Path,
    capture_volume_dir: Path,
) -> dict[str, Any]:
    """返回 OptiTrack 对齐阶段的计划/禁用状态。"""

    if config.optitrack_csv is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "world_points": str(capture_volume_dir / "world_points.csv"),
        "optitrack_csv": str(config.optitrack_csv.expanduser().resolve()),
        "output_dir": str(_optitrack_alignment_output_dir(config, extrinsic_dir)),
        "lambda_xy_list": config.optitrack_lambda_xy_list,
        "select_lambda": config.optitrack_select_lambda,
        "allow_global_scale": config.optitrack_allow_global_scale,
    }


def _run_optitrack_alignment_stage(
    config: WorkspaceCalibrationConfig,
    extrinsic_dir: Path,
    capture_volume_dir: Path,
) -> dict[str, Any]:
    """可选运行 12D OptiTrack/world_points 坐标系对齐。"""

    plan = _optitrack_alignment_plan(config, extrinsic_dir, capture_volume_dir)
    if not plan["enabled"]:
        return plan
    world_points_csv = capture_volume_dir / "world_points.csv"
    if not world_points_csv.exists():
        raise FileNotFoundError(f"world_points.csv not found for OptiTrack alignment: {world_points_csv}")

    from caliscope.pipelines.optitrack_alignment_12d import OptitrackAlignmentConfig, run_optitrack_alignment_12d

    output_dir = _optitrack_alignment_output_dir(config, extrinsic_dir)
    transform = run_optitrack_alignment_12d(
        OptitrackAlignmentConfig(
            world_points_csv=world_points_csv,
            optitrack_csv=config.optitrack_csv.expanduser().resolve(),
            output_dir=output_dir,
            offset_min=config.optitrack_offset_min,
            offset_max=config.optitrack_offset_max,
            coarse_offset_step=config.optitrack_coarse_offset_step,
            lambda_xy_list=config.optitrack_lambda_xy_list,
            select_lambda=config.optitrack_select_lambda,
            test_ratio=config.optitrack_test_ratio,
            seed=config.optitrack_seed,
            max_world_grid_rmse_m=config.optitrack_max_world_grid_rmse_m,
            min_points_per_fit_frame=config.optitrack_min_points_per_fit_frame,
            min_overlap_frames=config.optitrack_min_overlap_frames,
            max_coarse_frames=config.optitrack_max_coarse_frames,
            top_candidates_to_refine=config.optitrack_top_candidates_to_refine,
            equal_height_maxiter=config.optitrack_equal_height_maxiter,
            offset_12d_maxiter=config.optitrack_offset_12d_maxiter,
            allow_global_scale=config.optitrack_allow_global_scale,
            write_plots=config.optitrack_write_plots,
        )
    )
    error_summary = transform.get("fit_error_all_refit", {})
    report = {
        **plan,
        "status": "completed",
        "schema_version": transform.get("schema_version"),
        "model_type": transform.get("model_type"),
        "chosen_lambda_xy": transform.get("chosen_lambda_xy"),
        "time_offset_seconds": transform.get("time_offset_seconds"),
        "scale_opti_to_camera_world": transform.get("scale_opti_to_camera_world"),
        "rmse_mm": None if "rmse_m" not in error_summary else float(error_summary["rmse_m"]) * 1000.0,
        "mean_error_mm": None if "mean_m" not in error_summary else float(error_summary["mean_m"]) * 1000.0,
        "p95_error_mm": None if "p95_m" not in error_summary else float(error_summary["p95_m"]) * 1000.0,
        "outputs": {
            "alignment_transform_12d_summary": str(output_dir / "alignment_transform_12d_summary.json"),
            "alignment_transform_summary": str(output_dir / "alignment_transform_summary.json"),
            "transform_only": str(output_dir / "transform_only.json"),
            "report": str(output_dir / "REPORT.md"),
        },
    }
    logger.info("OptiTrack alignment complete. RMSE: %.4f mm", report["rmse_mm"])
    return report


def _configure_runtime(config: WorkspaceCalibrationConfig, *, camera_count: int) -> dict[str, Any]:
    """配置本进程的并行度，并记录 CPU/OpenCV/CUDA 运行时信息。

    OpenCV 的 ChArUco 检测主要吃 CPU。并行处理 19 台相机时，如果每个 worker 都
    使用 OpenCV 默认线程数，容易在集群节点上过度订阅 CPU。因此默认按
    ``cpu_count / workers`` 给 OpenCV 分配内部线程数。

    Args:
        config: 运行配置。
        camera_count: 当前工作区的相机数量。

    Returns:
        运行时信息字典，会写入标定报告。
    """

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
    """计算实际用于多相机检测的 worker 数量。

    Args:
        config: 运行配置。
        camera_count: 当前工作区相机数量。

    Returns:
        至少为 1，且不超过相机数量的 worker 数。
    """

    if not config.parallel:
        return 1
    requested = config.workers if config.workers is not None else camera_count
    return max(1, min(camera_count, requested))


def _discover_camera_videos(directory: Path) -> dict[int, Path]:
    """扫描目录中的 ``cam_N.mp4`` 文件并按 cam_id 建立映射。

    Args:
        directory: 内参或外参视频目录。

    Returns:
        ``{cam_id: video_path}`` 字典。

    Raises:
        FileNotFoundError: 目录下没有任何符合命名规则的视频。
    """

    videos: dict[int, Path] = {}
    for path in sorted(directory.glob("cam_*.mp4")):
        cam_id = _cam_id_from_name(path.stem)
        if cam_id is not None:
            videos[cam_id] = path
    if not videos:
        raise FileNotFoundError(f"No cam_N.mp4 videos found in {directory}")
    return videos


def _cam_id_from_name(name: str) -> int | None:
    """从 ``cam_N`` 文件名或标签中解析相机 ID。

    Args:
        name: 不带扩展名的文件名或 CSV 中的 camera 字段。

    Returns:
        解析成功时返回整数 cam_id，否则返回 None。
    """

    if not name.startswith("cam_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def _load_extrinsic_charuco(workspace: Path) -> Charuco:
    """加载外参标定使用的 ChArUco 板配置。

    Args:
        workspace: Caliscope 工作区根目录。

    Returns:
        ChArUco target 对象；优先使用 extrinsic 配置，缺失时回退到 intrinsic 配置。
    """

    targets_dir = workspace / "calibration" / "targets"
    extrinsic_path = targets_dir / "extrinsic_charuco.toml"
    if not extrinsic_path.exists():
        extrinsic_path = targets_dir / "intrinsic_charuco.toml"
    return Charuco.from_toml(extrinsic_path)


def _load_camera_name_mapping(workspace: Path) -> dict[int, dict[str, str]]:
    """读取工作区里的相机文件名映射表。

    Args:
        workspace: Caliscope 工作区根目录。

    Returns:
        以 cam_id 为 key 的 CSV 行数据；文件不存在时返回空字典。
    """

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
    """根据视频元数据、旧 camera_array 和映射表构造当前相机数组。

    这里有意不复用旧外参：每次运行都应由当前外参视频重新求解 pose，避免把旧结果
    混入新的同步/检测产物。旧的 rotation_count、label、ignore 等展示/管理字段会保留。

    Args:
        workspace: 工作区根目录。
        intrinsic_videos: 内参视频路径映射。
        extrinsic_videos: 外参视频路径映射。
        read_metadata: 是否尝试读取 GoPro 序列号和型号。
        show_progress: 是否显示相机元数据读取进度。

    Returns:
        新的 CameraArray，以及可写入报告的 metadata 读取记录。
    """

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
    """尝试读取已有 camera_array.toml，失败时退化为 None。

    Args:
        path: camera_array.toml 路径。

    Returns:
        读取成功时返回 CameraArray，否则返回 None。
    """

    if not path.exists():
        return None
    try:
        return CameraArray.from_toml(path)
    except Exception as e:
        logger.warning("Could not load existing camera array %s: %s", path, e)
        return None


def _metadata_source_path(workspace: Path, row: dict[str, str], fallback: Path) -> Path:
    """选择读取 GoPro 元数据的源视频路径。

    Args:
        workspace: 工作区根目录。
        row: camera_name_mapping.csv 中对应 cam_id 的行。
        fallback: 映射表缺失或源视频不存在时使用的外参视频路径。

    Returns:
        优先返回原始视频路径，否则返回 fallback。
    """

    raw = row.get("video_path")
    if raw:
        candidate = workspace / raw
        if candidate.exists():
            return candidate
    return fallback


def _workspace_relative(workspace: Path, path: Path) -> str:
    """把路径尽量转换成相对工作区的字符串。

    Args:
        workspace: 工作区根目录。
        path: 需要写入 TOML 的文件路径。

    Returns:
        能相对化时返回相对路径字符串，否则返回绝对路径字符串。
    """

    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def _load_intrinsics_profiles(library_path: Path) -> dict[str, dict[str, Any]]:
    """加载内参 profile，兼容文件、目录和 Caliscope 工作区三种输入。

    Args:
        library_path: 内参库 TOML、包含 TOML 的目录，或含 camera_array.toml 的工作区。

    Returns:
        以 serial/profile key 为 key 的 profile 字典。
    """

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
    """从单个 TOML 文件读取内参 profile。

    Args:
        path: 内参库 TOML 或 camera_array.toml。

    Returns:
        过滤掉无 matrix/size/distortions 的有效 profile 字典。
    """

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
    """把 camera_array.toml 中带序列号的相机转换为内参 profile。

    Args:
        path: Caliscope camera_array.toml 路径。

    Returns:
        以相机 serial_number 为 key 的 profile 字典。
    """

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
    """把内参库 profile 应用到当前 CameraArray。

    匹配优先级为 serial_number，其次是显式启用的 source_cam_id fallback。fallback 会在
    报告中标明，避免把无序列号匹配伪装成真实相机序列号匹配。

    Args:
        camera_array: 需要补齐内参的相机数组，会被原地更新。
        profiles: 已加载的内参 profile。
        library_path: 内参库路径，用于写入 intrinsics_source。
        source_cam_id_fallback: 是否允许按 source_cam_id 匹配。
        report_repo: 内参质量报告仓库。
        persist_reports: plan-only 模式下为 False，避免写文件。

    Returns:
        每台相机的匹配/跳过记录。
    """

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
    """按 source_cam_id 为内参 profile 建索引。

    Args:
        profiles: 以 serial 或 profile key 索引的内参 profile。

    Returns:
        ``{source_cam_id: (profile_serial, profile)}`` 字典。
    """

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
    """为单台相机选择最合适的内参 profile。

    Args:
        camera: 当前目标相机。
        profiles: 以 serial/profile key 索引的内参库。
        by_source_cam_id: source_cam_id 索引。
        source_cam_id_fallback: 是否允许缺少序列号时按 cam_id/source_cam_id 匹配。

    Returns:
        ``(profile, method, profile_serial)``；无法匹配时三者均为 None。
    """

    if camera.serial_number is not None and camera.serial_number in profiles:
        return profiles[camera.serial_number], "serial", camera.serial_number
    if source_cam_id_fallback and camera.cam_id in by_source_cam_id:
        serial, profile = by_source_cam_id[camera.cam_id]
        return profile, "source_cam_id", serial
    return None, None, None


def _adapt_profile_intrinsics(profile: dict[str, Any], target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    """把内参 profile 适配到目标视频分辨率。

    支持三种情况：尺寸完全一致、同宽高比缩放、横竖屏 90 度旋转后再缩放。畸变系数
    不随尺度变化；标准针孔模型旋转 90 度时需要同步旋转切向畸变 ``p1/p2``。

    Args:
        profile: 内参库中的单个 profile。
        target_size: 目标相机视频尺寸，格式为 ``(width, height)``。

    Returns:
        ``(matrix, distortions, adaptation_label)``。

    Raises:
        ValueError: profile 数据无效，或尺寸无法通过缩放/旋转兼容。
    """

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
    """判断两个分辨率是否具有相同宽高比。

    Args:
        source_size: 源尺寸 ``(width, height)``。
        target_size: 目标尺寸 ``(width, height)``。

    Returns:
        宽高比在浮点误差范围内一致时返回 True。
    """

    source_w, source_h = source_size
    target_w, target_h = target_size
    return abs((source_w / source_h) - (target_w / target_h)) < 1e-6


def _scale_intrinsics(matrix: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """按图像缩放比例调整相机矩阵中的焦距和主点。

    Args:
        matrix: 3x3 相机内参矩阵。
        sx: x 方向缩放比例。
        sy: y 方向缩放比例。

    Returns:
        缩放后的新矩阵；不会修改输入矩阵。
    """

    scaled = matrix.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def _rotate_intrinsics_90_ccw(matrix: np.ndarray, source_width: int) -> np.ndarray:
    """把横屏内参旋转为 90 度逆时针后的竖屏内参。

    Args:
        matrix: 源图像的 3x3 内参矩阵。
        source_width: 源图像宽度，用于变换 y 方向主点。

    Returns:
        旋转后的 3x3 内参矩阵。
    """

    rotated = np.eye(3, dtype=np.float64)
    rotated[0, 0] = matrix[1, 1]
    rotated[1, 1] = matrix[0, 0]
    rotated[0, 2] = matrix[1, 2]
    rotated[1, 2] = source_width - matrix[0, 2]
    return rotated


def _rotate_distortions_90_ccw(distortions: np.ndarray, fisheye: bool) -> np.ndarray:
    """旋转 90 度逆时针对畸变参数的影响。

    径向畸变不变；标准针孔模型的切向畸变 ``p1/p2`` 会随坐标轴旋转变换。fisheye
    模型这里保持原样，因为它只使用径向项。

    Args:
        distortions: 一维畸变系数数组。
        fisheye: 是否为 fisheye 模型。

    Returns:
        旋转后的畸变系数副本。
    """

    if fisheye or distortions.size != 5:
        return distortions.copy()
    k1, k2, p1, p2, k3 = distortions.tolist()
    return np.array([k1, k2, -p2, p1, k3], dtype=np.float64)


def _validated_intrinsics_arrays(profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """校验并转换内参 profile 中的矩阵和畸变系数。

    Args:
        profile: 内参 profile 字典。

    Returns:
        ``(matrix, distortions)``，均为 float64 numpy 数组。

    Raises:
        ValueError: 矩阵形状或畸变系数数量不符合模型要求。
    """

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
    """构造单台相机的内参导入报告行。

    Args:
        camera: 当前相机。
        status: ``matched``、``skipped`` 或其他状态文本。
        method: 匹配方法，例如 serial 或 source_cam_id。
        profile: 匹配到的内参 profile。
        profile_serial: profile 对应的序列号/key。
        adaptation: 尺寸适配方式说明。
        reason: 跳过或失败原因。

    Returns:
        可写入 TOML 的报告字典。
    """

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
    """把内参库 profile 转成 Caliscope 的 IntrinsicCalibrationReport。

    Args:
        profile: 内参库中的单个 profile。

    Returns:
        可由现有报告仓库保存的 IntrinsicCalibrationReport。
    """

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
    """对内参库未覆盖的相机执行视频内参标定。

    正常情况下本项目会全部从内参库复用；这个函数是兜底路径，避免新增相机或库缺失时
    管线直接中断。

    Args:
        camera_array: 需要补齐内参的相机数组，会被原地更新。
        intrinsic_videos: 内参视频路径映射。
        charuco: 内参标定使用的 ChArUco 板。
        frame_step: 内参视频抽帧步长。
        report_repo: 内参报告仓库。

    Returns:
        实际重标定相机的报告列表。
    """

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
    """同步外参视频时间线，或复用已有同步文件。

    Args:
        extrinsic_dir: 包含 ``cam_N.mp4`` 的外参视频目录。
        cam_ids: 参与同步的相机 ID。
        reuse_existing: 是否允许复用 ``timestamps.csv`` 和 ``sync_offsets.toml``。

    Returns:
        同步报告字典，并额外包含 ``mode`` 字段表示 computed/reused。
    """

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
    """生成断点复用计划，不执行任何重计算。

    Args:
        extrinsic_dir: 外参视频目录。
        tracker_dir: ChArUco 点输出目录。
        capture_volume_dir: capture volume 输出目录。
        cam_ids: 当前工作区相机 ID。
        frame_step: 外参点检测抽帧步长。
        reuse_sync: 是否允许复用同步文件。
        reuse_points: 是否允许复用 image_points.csv。
        require_image_points_manifest: 是否要求 image_points manifest 匹配。
        reuse_capture_volume: 是否允许复用完整 capture volume。
        allow_partial_extrinsics: 是否允许复用 partial capture volume。

    Returns:
        sync、image_points、capture_volume 三个阶段的 action/available 明细。
    """

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
    """把阶段复用计划写入日志，便于远程运行时快速判断是否卡住。

    Args:
        plan: ``_build_stage_plan`` 返回的计划字典。
    """

    logger.info(
        "Stage plan: sync=%s, image_points=%s, capture_volume=%s",
        plan["sync"]["action"],
        plan["image_points"]["action"],
        plan["capture_volume"]["action"],
    )


def _completed_capture_volume_status(capture_volume_dir: Path) -> tuple[bool, bool, list[int]]:
    """检查已有 capture volume 是否由本管线生成且外参完整。

    Args:
        capture_volume_dir: capture volume 输出目录。

    Returns:
        ``(available, complete_extrinsics, unposed_cam_ids)``。
    """

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
    """加载或提取外参阶段使用的同步 ChArUco 点。

    只有 manifest 与当前 timestamps、cam_ids 和 frame_step 匹配时，默认 resume 才会复用
    image_points.csv。这样可以避免把旧抽帧参数或旧同步结果下的检测点误用于新标定。

    Args:
        extrinsic_dir: 外参视频目录。
        tracker_dir: ChArUco 检测结果目录。
        cameras: 当前相机数据，提供 rotation_count。
        charuco: ChArUco target。
        frame_step: 同步帧抽样步长。
        reuse_existing: 是否允许复用已有 image_points.csv。
        require_manifest: 是否要求 manifest 完全匹配。
        parallel: 是否并行处理同一 sync index 的多相机帧。
        max_workers: 相机 worker 上限。
        show_progress: 是否显示 tqdm 进度条。

    Returns:
        ImagePoints；对象上会附加 ``_resumed_from_cache`` 标记供报告使用。
    """

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
        """把底层处理进度转发到 tqdm 或日志。"""

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
    progress_callback = on_progress if show_progress else None

    try:
        image_points = process_synchronized_recording(
            recording_dir=extrinsic_dir,
            cameras=cameras,
            tracker=CharucoTracker(charuco),
            synced_timestamps=synced_timestamps,
            subsample=frame_step,
            parallel=parallel,
            max_workers=max_workers,
            on_progress=progress_callback,
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
    """返回 image_points.csv 对应的 manifest 路径。

    Args:
        image_points_path: ChArUco image_points.csv 路径。

    Returns:
        同目录下的 ``image_points.meta.toml`` 路径。
    """

    return image_points_path.with_name(f"{image_points_path.stem}.meta.toml")


def _image_points_manifest(
    image_points: ImagePoints,
    *,
    extrinsic_dir: Path,
    cam_ids: list[int],
    frame_step: int,
) -> dict[str, Any]:
    """构造 image_points.csv 的复用 manifest。

    Args:
        image_points: 刚提取完成的点数据。
        extrinsic_dir: 外参视频目录。
        cam_ids: 当前相机 ID 列表。
        frame_step: 本次提取使用的同步帧抽样步长。

    Returns:
        包含同步文件 mtime 和点数据摘要的 manifest 字典。
    """

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
    """判断已有 image_points manifest 是否匹配当前运行参数。

    Args:
        image_points_path: 待复用的 image_points.csv 路径。
        extrinsic_dir: 当前外参视频目录。
        cam_ids: 当前相机 ID 列表。
        frame_step: 当前抽样步长。

    Returns:
        manifest 存在且 schema、相机集合、抽样步长、timestamps mtime 均一致时返回 True。
    """

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
    filter_scope: str,
    filter_sigma: float | None,
    max_nfev: int,
    scipy_verbose: int,
    align_to_object: bool,
    show_progress: bool,
    allow_partial_extrinsics: bool,
) -> tuple[dict[str, Any], CaptureVolume]:
    """执行外参 bootstrap、两轮 BA 和可选坐标系对齐。

    默认要求所有相机都获得外参；如果外参图被 ChArUco 观测分裂，会在 BA 前失败，避免
    把只覆盖部分相机的低 RMSE 结果误认为完整标定。

    Args:
        image_points: 外参阶段同步 ChArUco 检测点。
        camera_array: 已有内参的相机数组。
        filter_percentile: 第一轮 BA 后剔除的最差百分比。
        filter_scope: per_camera/overall 按观测点剔除；sync_index 按整帧剔除。
        filter_sigma: sync_index 整帧剔除的可选 robust sigma 阈值；None 表示只按百分比剔除。
        max_nfev: 每轮 BA 最大函数评估次数。
        scipy_verbose: SciPy least_squares 日志级别。
        align_to_object: 是否对齐到 ChArUco 板坐标系。
        show_progress: 是否显示阶段进度条。
        allow_partial_extrinsics: 是否允许部分相机未 pose 时继续。

    Returns:
        ``(capture_report, capture_volume)``。
    """

    step_count = 5 if align_to_object else 4
    progress_bar = _make_progress_bar(show_progress, total=step_count, desc="Capture volume", unit="stage")

    def finish_stage(label: str) -> None:
        """更新 capture volume 阶段进度条。"""

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

        filter_stats: dict[str, Any] | None = None
        if filter_percentile > 0:
            if filter_scope == "sync_index":
                filter_stats = first.describe_sync_index_error_filter(
                    filter_percentile,
                    sigma_multiplier=filter_sigma,
                )
                filtered = first.filter_by_percentile_error(
                    filter_percentile,
                    scope=filter_scope,
                    sigma_multiplier=filter_sigma,
                )
            else:
                percentile_before = first.reprojection_report.n_observations_matched
                percentile_filtered = first.filter_by_percentile_error(filter_percentile, scope=filter_scope)
                percentile_after = percentile_filtered.reprojection_report.n_observations_matched
                filter_stats = {
                    "mode": "point_percentile_then_sigma" if filter_sigma is not None else "point_percentile",
                    "scope": filter_scope,
                    "percentile": {
                        "percentile": float(filter_percentile),
                        "observations_before": int(percentile_before),
                        "observations_after": int(percentile_after),
                        "actual_drop_observations": int(percentile_before - percentile_after),
                        "actual_drop_percent": float(
                            ((percentile_before - percentile_after) / percentile_before) * 100.0
                        )
                        if percentile_before > 0
                        else 0.0,
                    },
                }
                if filter_sigma is not None:
                    sigma_stats = percentile_filtered.describe_sigma_error_filter(
                        filter_sigma, scope=filter_scope
                    )
                    filtered = percentile_filtered.filter_by_sigma_error(
                        filter_sigma, scope=filter_scope
                    )
                    sigma_after = filtered.reprojection_report.n_observations_matched
                    sigma_stats["observations_after"] = int(sigma_after)
                    filter_stats["sigma"] = sigma_stats
                else:
                    filtered = percentile_filtered
        else:
            filtered = first
        finish_stage("filter")
        filtered_report = _summarize_reprojection_report(filtered)
        if filter_stats is not None:
            if filter_scope == "sync_index":
                sigma_suffix = ""
                if filter_sigma is not None and filter_stats.get("sigma_threshold") is not None:
                    sigma_suffix = (
                        f" above median+{filter_sigma:g}sigma threshold "
                        f"{filter_stats['sigma_threshold']:.4f}px"
                    )
                logger.info(
                    "After %.2f%% %s reprojection filtering RMSE: %.4f px; dropped %d/%d frames%s",
                    filter_percentile,
                    filter_scope,
                    filtered_report["overall_rmse"],
                    filter_stats["actual_drop_frames"],
                    filter_stats["total_frames"],
                    sigma_suffix,
                )
            else:
                percentile_stats = filter_stats.get("percentile", {})
                sigma_stats = filter_stats.get("sigma")
                sigma_msg = ""
                if sigma_stats is not None:
                    sigma_msg = (
                        f" then {sigma_stats.get('actual_drop_observations', 0)} sigma points "
                        f"above {filter_sigma:g}sigma"
                    )
                logger.info(
                    "After %.2f%% %s point filtering%s RMSE: %.4f px; dropped %d percentile points%s",
                    filter_percentile,
                    filter_scope,
                    f" + {filter_sigma:g}sigma" if filter_sigma is not None else "",
                    filtered_report["overall_rmse"],
                    percentile_stats.get("actual_drop_observations", 0),
                    sigma_msg,
                )

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
            "filter_scope": filter_scope,
            "filter_sigma": filter_sigma,
            "filter_stats": filter_stats,
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
    """在 resume 模式下加载已完成的 capture volume。

    Args:
        capture_volume_dir: capture volume 输出目录。
        allow_partial_extrinsics: 是否允许复用 partial capture volume。

    Returns:
        可复用时返回 ``(capture_report, capture_volume)``，否则返回 ``(None, None)``。
    """

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
    """选择用于坐标系对齐的 ChArUco 可见帧。

    Args:
        volume: 已完成 BA 的 capture volume。

    Returns:
        能提供最多 ChArUco 点和相机观测的 sync_index；找不到时返回 None。
    """

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
    """汇总二维检测点规模和各相机观测数量。

    Args:
        image_points: ChArUco 或其他 tracker 输出的二维点。

    Returns:
        可写入报告的观测摘要。
    """

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
    """把 CaptureVolume 的重投影报告压缩成 TOML 友好的字典。

    Args:
        volume: 已经可计算 reprojection_report 的 capture volume。

    Returns:
        包含总体 RMSE、逐相机 RMSE 和匹配观测数量的摘要。
    """

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
    """提取最后一次 BA 的优化状态。

    Args:
        volume: BA 后的 capture volume。

    Returns:
        优化状态字典；如果该 volume 不是 optimize() 结果则返回 None。
    """

    status = volume.optimization_status
    if status is None:
        return None
    return {
        "converged": status.converged,
        "termination_reason": status.termination_reason,
        "iterations": status.iterations,
        "final_cost": status.final_cost,
    }


def _make_progress_bar(enabled: bool, *, total: int, desc: str, unit: str) -> Any | None:
    """按需创建 tqdm 进度条。

    Args:
        enabled: 是否启用进度条。
        total: 进度条总量。
        desc: 进度条标题。
        unit: 进度单位。

    Returns:
        tqdm 实例；未启用或未安装 tqdm 时返回 None。
    """

    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.info("tqdm is not installed; falling back to log-only progress")
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _optional_float(value: Any) -> float | None:
    """把可选数值转换为 float。

    Args:
        value: None 或可转换为 float 的值。

    Returns:
        None 或 float。
    """

    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    """把可选数值转换为 int。

    Args:
        value: None 或可转换为 int 的值。

    Returns:
        None 或 int。
    """

    if value is None:
        return None
    return int(value)


def _toml_clean(value: Any) -> Any:
    """递归清理对象，确保 rtoml 可以稳定序列化。

    该函数会去掉 None、转换 numpy scalar 和 Path，并把 list-of-table 字段排到普通
    table 后面，避免 rtoml 抛出 ``values must be emitted before tables``。

    Args:
        value: 任意待写入 TOML 的 Python 对象。

    Returns:
        只包含 TOML 可序列化类型的对象。
    """

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
    """为 TOML 字典字段提供稳定排序权重。

    Args:
        value: 字典中的字段值。

    Returns:
        排序权重；标量优先，普通 table 其次，array-of-table 最后。
    """

    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return (2, "")
    if isinstance(value, dict):
        return (1, "")
    return (0, "")


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        配置好所有 CLI 开关的 ArgumentParser。
    """

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
    parser.add_argument(
        "--filter-scope",
        choices=("per_camera", "overall", "sync_index"),
        default="per_camera",
        help="How to filter outliers after first BA; sync_index removes whole synchronized frames and is capped at 30%%",
    )
    parser.add_argument(
        "--filter-sigma",
        type=float,
        default=None,
        help=(
            "Apply robust median + N * upper-sigma filtering after percentile filtering. "
            "For point scopes this removes additional point observations; for sync_index it gates frame removal."
        ),
    )
    parser.add_argument("--max-nfev", type=int, default=1000, help="Maximum function evaluations per BA pass")
    parser.add_argument("--scipy-verbose", type=int, default=0, choices=(0, 1, 2), help="SciPy least_squares verbosity")
    parser.add_argument("--align-to-object", action="store_true", help="Align final volume to a visible Charuco board frame")
    parser.add_argument(
        "--optitrack-csv",
        type=Path,
        default=None,
        help="Optional Motive/OptiTrack CSV; enables 12D world_points-to-OptiTrack alignment after extrinsic calibration",
    )
    parser.add_argument(
        "--optitrack-alignment-output-dir",
        type=Path,
        default=None,
        help="Output directory for 12D OptiTrack alignment; defaults to calibration/extrinsic/optitrack_alignment_12d",
    )
    parser.add_argument(
        "--optitrack-lambda-xy-list",
        default="0,0.1,0.2,0.5,1,10,100",
        help="Comma-separated lambda_xy values for 12D marker offset regularization",
    )
    parser.add_argument("--optitrack-select-lambda", default="0.2", help="Numeric lambda, min_test, or min_all")
    parser.add_argument("--optitrack-offset-min", type=float, default=-20.0, help="Minimum time offset search bound in seconds")
    parser.add_argument("--optitrack-offset-max", type=float, default=20.0, help="Maximum time offset search bound in seconds")
    parser.add_argument("--optitrack-coarse-offset-step", type=float, default=0.25, help="Coarse time offset grid step in seconds")
    parser.add_argument(
        "--optitrack-max-world-grid-rmse-m",
        type=float,
        default=0.005,
        help="Maximum per-frame 5x3 world grid fit RMSE used for OptiTrack alignment",
    )
    parser.add_argument("--optitrack-min-points-per-fit-frame", type=int, default=15, help="Minimum world_points in each fit frame")
    parser.add_argument("--optitrack-min-overlap-frames", type=int, default=40, help="Minimum overlapping frames for alignment")
    parser.add_argument("--optitrack-max-coarse-frames", type=int, default=120, help="Maximum frames used in coarse seed search")
    parser.add_argument("--optitrack-top-candidates-to-refine", type=int, default=8, help="Equal-height seed candidates to refine")
    parser.add_argument("--optitrack-equal-height-maxiter", type=int, default=260, help="Nelder-Mead iterations for equal-height seed")
    parser.add_argument("--optitrack-offset-12d-maxiter", type=int, default=500, help="L-BFGS-B iterations for 12D offsets")
    parser.add_argument("--optitrack-allow-global-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optitrack-test-ratio", type=float, default=0.33, help="Train/test split ratio for lambda comparison")
    parser.add_argument("--optitrack-seed", type=int, default=20260521, help="Random seed for train/test split")
    parser.add_argument("--optitrack-write-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true", help="Print planned reuse/recompute stages and exit")
    parser.add_argument(
        "--allow-partial-extrinsics",
        action="store_true",
        help="Allow saving a capture volume when some cameras remain unposed",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口函数。

    Args:
        argv: 可选命令行参数；None 表示使用 sys.argv。

    Returns:
        进程退出码，成功时为 0。
    """

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
        filter_scope=args.filter_scope,
        filter_sigma=args.filter_sigma,
        max_nfev=args.max_nfev,
        scipy_verbose=args.scipy_verbose,
        align_to_object=args.align_to_object,
        optitrack_csv=args.optitrack_csv,
        optitrack_alignment_output_dir=args.optitrack_alignment_output_dir,
        optitrack_lambda_xy_list=args.optitrack_lambda_xy_list,
        optitrack_select_lambda=args.optitrack_select_lambda,
        optitrack_offset_min=args.optitrack_offset_min,
        optitrack_offset_max=args.optitrack_offset_max,
        optitrack_coarse_offset_step=args.optitrack_coarse_offset_step,
        optitrack_max_world_grid_rmse_m=args.optitrack_max_world_grid_rmse_m,
        optitrack_min_points_per_fit_frame=args.optitrack_min_points_per_fit_frame,
        optitrack_min_overlap_frames=args.optitrack_min_overlap_frames,
        optitrack_max_coarse_frames=args.optitrack_max_coarse_frames,
        optitrack_top_candidates_to_refine=args.optitrack_top_candidates_to_refine,
        optitrack_equal_height_maxiter=args.optitrack_equal_height_maxiter,
        optitrack_offset_12d_maxiter=args.optitrack_offset_12d_maxiter,
        optitrack_allow_global_scale=args.optitrack_allow_global_scale,
        optitrack_test_ratio=args.optitrack_test_ratio,
        optitrack_seed=args.optitrack_seed,
        optitrack_write_plots=args.optitrack_write_plots,
        plan_only=args.plan_only,
        allow_partial_extrinsics=args.allow_partial_extrinsics,
    )
    report = run_workspace_calibration(config)
    if args.plan_only:
        print(rtoml.dumps(_toml_clean(report)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
