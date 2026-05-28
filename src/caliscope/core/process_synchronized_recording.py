"""同步多相机视频的批处理入口。

本模块只负责按 ``SynchronizedTimestamps`` 对齐后的帧索引读取视频、调用 tracker、
汇总二维点数据；不做实时播放，也不做标定优化。这样 GUI、CLI 和测试都可以复用同一
套同步帧处理逻辑。
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraData
from caliscope.core.point_data import ImagePoints
from caliscope.packets import PointPacket
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.video_utils import _open_video_capture_no_auto_rotation
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """单台相机在某个 sync index 上的帧和检测结果。"""

    frame: NDArray[np.uint8]
    points: PointPacket | None
    frame_index: int


def process_synchronized_recording(
    recording_dir: Path,
    cameras: dict[int, CameraData],
    tracker: Tracker,
    synced_timestamps: SynchronizedTimestamps,
    *,
    subsample: int = 1,
    parallel: bool = True,
    max_workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_frame_data: Callable[[int, dict[int, FrameData]], None] | None = None,
    token: CancellationToken | None = None,
) -> ImagePoints:
    """从已同步的多相机视频中提取二维 landmark。

    函数按 sync index 遍历时间线，再为每台相机读取对应原始帧。并行模式只并行同一
    sync index 内的不同相机，回调仍在主循环线程触发，避免 GUI/调用方处理共享状态时
    需要额外加锁。

    Args:
        recording_dir: 包含 ``cam_N.mp4`` 的录制目录。
        cameras: 以 cam_id 为 key 的相机数据，主要提供 rotation_count。
        tracker: 用于二维点检测的 tracker。
        synced_timestamps: 已构造好的时间同步对象。
        subsample: 每隔多少个 sync index 处理一帧，1 表示全部处理。
        parallel: 是否并行处理同一 sync index 的多相机帧。
        max_workers: 相机 worker 数上限；超过相机数量时自动截断。
        on_progress: 进度回调，参数为 ``(current, total)``。
        on_frame_data: 可选帧数据回调，供 GUI 预览或调试使用。
        token: 取消令牌，用于后台任务优雅中断。

    Returns:
        包含所有二维观测的 ImagePoints。
    """
    all_sync_indices = synced_timestamps.sync_indices[::subsample]
    total = len(all_sync_indices)

    logger.info(
        f"Processing {total} sync indices "
        f"(subsample={subsample}, total available={len(synced_timestamps.sync_indices)})"
    )

    frame_sources = _create_frame_sources(recording_dir, cameras, max_workers=max_workers)
    point_rows: list[dict] = []

    try:
        use_pool = parallel and len(frame_sources) > 1

        if use_pool:
            camera_pool = ThreadPoolExecutor(max_workers=_bounded_worker_count(len(frame_sources), max_workers))
        else:
            camera_pool = None

        try:
            for i, sync_index in enumerate(all_sync_indices):
                if token is not None and token.is_cancelled:
                    logger.info("Processing cancelled")
                    break

                frame_data: dict[int, FrameData] = {}

                if use_pool and camera_pool is not None:
                    # --- Parallel path ---
                    futures: dict[int, Future[tuple[int, FrameData | None, list[dict]]]] = {}
                    for cam_id in synced_timestamps.cam_ids:
                        frame_index = synced_timestamps.frame_for(sync_index, cam_id)
                        if frame_index is None:
                            continue
                        if cam_id not in frame_sources:
                            continue
                        camera = cameras[cam_id]
                        frame_time = synced_timestamps.time_for(cam_id, frame_index)
                        futures[cam_id] = camera_pool.submit(
                            _process_one_camera,
                            cam_id,
                            sync_index,
                            frame_index,
                            frame_sources[cam_id],
                            camera,
                            tracker,
                            frame_time,
                        )

                    for cam_id, future in futures.items():
                        _, fd, rows = future.result()
                        if fd is not None:
                            frame_data[cam_id] = fd
                        point_rows.extend(rows)
                else:
                    # --- Serial path (original logic) ---
                    for cam_id in synced_timestamps.cam_ids:
                        frame_index = synced_timestamps.frame_for(sync_index, cam_id)
                        if frame_index is None:
                            continue
                        if cam_id not in frame_sources:
                            continue
                        camera = cameras[cam_id]
                        frame = frame_sources[cam_id].read_frame_at(frame_index)
                        if frame is None:
                            logger.warning(
                                f"Failed to read frame: sync={sync_index}, cam_id={cam_id}, frame_index={frame_index}"
                            )
                            continue
                        points = tracker.get_points(frame, cam_id, camera.rotation_count)
                        frame_data[cam_id] = FrameData(frame, points, frame_index)
                        frame_time = synced_timestamps.time_for(cam_id, frame_index)
                        _accumulate_points(point_rows, sync_index, cam_id, frame_index, frame_time, points)

                # Threading contract: callbacks are always invoked from this
                # thread (the worker thread that owns the sync-index loop),
                # never from pool threads. Presenters rely on this guarantee
                # for unsynchronized accumulator state.
                if on_frame_data is not None:
                    on_frame_data(sync_index, frame_data)
                if on_progress is not None:
                    on_progress(i + 1, total)
        finally:
            if camera_pool is not None:
                camera_pool.shutdown(wait=False)

    finally:
        for source in frame_sources.values():
            source.close()

    return _build_image_points(point_rows)


def get_initial_thumbnails(
    recording_dir: Path,
    cameras: dict[int, CameraData],
) -> dict[int, NDArray[np.uint8]]:
    """快速读取每台相机的首帧缩略图。

    这里只用 OpenCV 打开视频并解码第一帧，不构造 FrameSource，适合 GUI 初始化缩略图。

    Args:
        recording_dir: 包含 ``cam_N.mp4`` 的目录。
        cameras: 以 cam_id 为 key 的相机数据。

    Returns:
        ``{cam_id: BGR_frame}`` 字典。
    """
    thumbnails: dict[int, NDArray[np.uint8]] = {}

    for cam_id in cameras:
        video_path = recording_dir / f"cam_{cam_id}.mp4"
        if not video_path.exists():
            logger.warning(f"Video file not found for cam_id {cam_id}")
            continue

        try:
            capture = _open_video_capture_no_auto_rotation(video_path)
            try:
                ok, frame = capture.read()
                if ok and frame is not None:
                    thumbnails[cam_id] = frame
            finally:
                capture.release()
        except Exception as e:
            logger.warning(f"Error reading first frame for cam_id {cam_id}: {e}")

    return thumbnails


def _create_frame_sources(
    recording_dir: Path,
    cameras: dict[int, CameraData],
    *,
    max_workers: int | None = None,
) -> dict[int, FrameSource]:
    """为每台相机创建 FrameSource。

    这里并行初始化多相机视频源，避免串行打开容器拖慢启动。

    Args:
        recording_dir: 包含 ``cam_N.mp4`` 的目录。
        cameras: 需要创建 FrameSource 的相机集合。
        max_workers: 初始化线程上限。

    Returns:
        成功打开的视频源字典；缺失或无法打开的视频会被跳过并写日志。
    """

    def _init_one(cam_id: int) -> tuple[int, FrameSource | None]:
        """初始化单台相机的视频源，失败时返回 None。"""

        try:
            return cam_id, FrameSource(recording_dir, cam_id)
        except FileNotFoundError:
            logger.warning(f"Video file not found for cam_id {cam_id}, skipping")
            return cam_id, None
        except ValueError as e:
            logger.warning(f"Error opening video for cam_id {cam_id}: {e}")
            return cam_id, None

    cam_ids = list(cameras.keys())

    with ThreadPoolExecutor(max_workers=_bounded_worker_count(len(cam_ids), max_workers)) as pool:
        results = pool.map(_init_one, cam_ids)

    return {cam_id: source for cam_id, source in results if source is not None}


def _bounded_worker_count(item_count: int, max_workers: int | None) -> int:
    """把用户给出的 worker 上限约束到安全范围。

    Args:
        item_count: 需要并行处理的项目数量。
        max_workers: 用户指定的 worker 上限；None 表示不额外限制。

    Returns:
        至少为 1，且不超过 item_count/max_workers 的线程数。
    """

    if item_count < 1:
        return 1
    if max_workers is None:
        return item_count
    return max(1, min(item_count, max_workers))


def _accumulate_points(
    point_rows: list[dict],
    sync_index: int,
    cam_id: int,
    frame_index: int,
    frame_time: float,
    points: PointPacket | None,
) -> None:
    """把单帧 PointPacket 展平追加到行列表。

    Args:
        point_rows: 输出行累加器。
        sync_index: 当前同步帧索引。
        cam_id: 相机 ID。
        frame_index: 原始视频帧索引。
        frame_time: 同步后的帧时间。
        points: tracker 返回的点包；为空或无点时不追加。
    """
    if points is None:
        return

    point_count = len(points.point_id)
    if point_count == 0:
        return

    obj_loc_x, obj_loc_y, obj_loc_z = points.obj_loc_list

    for i in range(point_count):
        point_rows.append(
            {
                "sync_index": sync_index,
                "cam_id": cam_id,
                "frame_index": frame_index,
                "frame_time": frame_time,
                "point_id": int(points.point_id[i]),
                "img_loc_x": float(points.img_loc[i, 0]),
                "img_loc_y": float(points.img_loc[i, 1]),
                "obj_loc_x": obj_loc_x[i],
                "obj_loc_y": obj_loc_y[i],
                "obj_loc_z": obj_loc_z[i],
            }
        )


def _process_one_camera(
    cam_id: int,
    sync_index: int,
    frame_index: int,
    frame_source: FrameSource,
    camera: CameraData,
    tracker: Tracker,
    frame_time: float,
) -> tuple[int, FrameData | None, list[dict]]:
    """处理某个 sync index 上的单台相机帧。

    并发安全依赖一个约束：同一时刻不会有两个线程处理同一个 cam_id。每台相机独占
    FrameSource，PointPacket 行数据也在局部列表里构造后返回，因此不会共享可变状态。

    Args:
        cam_id: 相机 ID。
        sync_index: 当前同步帧索引。
        frame_index: 该相机对应的原始视频帧索引。
        frame_source: 该相机专属 FrameSource。
        camera: 相机数据，提供 rotation_count。
        tracker: 二维点检测器。
        frame_time: 同步后的时间戳。

    Returns:
        ``(cam_id, frame_data_or_none, point_rows)``。
    """
    frame = frame_source.read_frame_at(frame_index)

    if frame is None:
        logger.warning(f"Failed to read frame: sync={sync_index}, cam_id={cam_id}, frame_index={frame_index}")
        return cam_id, None, []

    points = tracker.get_points(frame, cam_id, camera.rotation_count)
    fd = FrameData(frame, points, frame_index)

    local_rows: list[dict] = []
    _accumulate_points(local_rows, sync_index, cam_id, frame_index, frame_time, points)

    return cam_id, fd, local_rows


def _build_image_points(point_rows: list[dict]) -> ImagePoints:
    """把累积的行数据转换为 ImagePoints。

    Args:
        point_rows: `_accumulate_points` 生成的行字典列表。

    Returns:
        经过 ImagePoints schema 校验的对象；无检测点时返回空表对象。
    """
    if not point_rows:
        df = pd.DataFrame(
            columns=[
                "sync_index",
                "cam_id",
                "frame_index",
                "frame_time",
                "point_id",
                "img_loc_x",
                "img_loc_y",
                "obj_loc_x",
                "obj_loc_y",
                "obj_loc_z",
            ]
        )
        return ImagePoints(df)

    df = pd.DataFrame(point_rows)
    return ImagePoints(df)
