# Caliscope 本地非 GUI 标定管线手册

本文档记录 `/home/btsun/project/Calibration/dependencies/caliscope` 当前本地版本的完整非 GUI 标定流程。
它不是上游 Caliscope 通用 README，而是当前 GoPro 多相机标定工作区的使用教程、参数说明和实现约定。

主入口：

```bash
python -m caliscope.pipelines.workspace_calibration
```

核心流程：发现 `cam_N.mp4` 视频、音频同步、复用或补算内参、同步抽取 ChArUco 点、外参初始化、两轮 bundle adjustment、重投影误差过滤、导出 Caliscope 与 aniposelib 相机数组。

## 1. 当前必须遵守的原则

| 原则 | 说明 |
| --- | --- |
| 内参优先按 GoPro 序列号匹配 | `cam_N` 会随工作区变化，不能当作稳定相机身份。 |
| 真实 GoPro 运行使用 `--read-metadata --no-source-cam-id-fallback` | 读取序列号并禁止按 `source_cam_id` 兜底，避免错配内参。 |
| 全流程使用 encoded raster 坐标 | 当前内参库是 no-auto-rotation 的 `1920x1080` 坐标。 |
| 更换视频读取、旋转处理、标定板或抽帧参数后必须重抽点 | 旧 `image_points.csv` 可能不再兼容。 |
| partial 外参只能当作 partial | 低 RMSE 不能代表所有相机都有外参。 |
| 当前推荐过滤参数为 `--filter-scope overall --filter-percentile 10 --filter-sigma 2` | 已用于当前 Desk、Kitchen partial、60x40 输出。 |

## 2. 环境和运行方式

使用 `cali` 环境直接调用 Python。长任务不要用 `conda run`，否则输出可能被缓冲。

推荐命令模板包含 `PYTHONPATH`，确保从源码 checkout 运行当前本地版本：

```bash
PATH="/home/btsun/.conda/envs/cali/bin:$PATH" \
PYTHONPATH="/home/btsun/project/Calibration/dependencies/caliscope/src" \
PYTHONUNBUFFERED=1 \
/home/btsun/.conda/envs/cali/bin/python -u -m caliscope.pipelines.workspace_calibration --help
```

常用验证：`pytest tests/test_frame_source.py tests/test_frame_source_sequential.py tests/test_api.py tests/test_process_synchronized_recording.py -q`。

## 3. 工作区目录规范

整理前真正必须准备的是视频和标定板参数；同步结果、抽点 CSV、capture volume 和最终相机数组都由管线生成，可在复跑时作为缓存复用。

```text
└── 📁<workspace>
    ├── camera_name_mapping.csv                         # 可选但推荐：原始文件映射和 metadata 源路径
    ├── camera_array.toml                               # 自动生成：最终 Caliscope 相机数组
    ├── camera_array_aniposelib.toml                    # 自动生成：aniposelib 输出
    └── 📁calibration
        ├── 📁targets                                   # 必需：至少有下面一个 ChArUco TOML
        │   ├── intrinsic_charuco.toml                  # 输入：标定板参数；可作为 fallback
        │   └── extrinsic_charuco.toml                  # 输入：外参优先使用的标定板参数
        ├── 📁intrinsic                                 # 必需：cam_N 集合必须和 extrinsic 一致
        │   └── cam_N.mp4                               # 输入：内参视频；复用内参时通常不读取，但当前管线仍要求存在
        └── 📁extrinsic                                 # 必需：外参视频目录
            ├── cam_N.mp4                               # 输入：外参视频，命名必须是 cam_N.mp4
            ├── timestamps.csv                          # 自动生成/可复用：音频同步时间线
            ├── sync_offsets.toml                       # 自动生成/可复用：音频同步摘要
            ├── 📁optitrack_alignment_12d               # 可选：world_points 与 OptiTrack 的 12D 坐标系对齐
            ├── 📁CHARUCO
            │   ├── image_points.csv                    # 自动生成/可复用：同步 ChArUco 2D 点
            │   └── image_points.meta.toml              # 自动生成/可复用：抽点 cache manifest
            └── 📁capture_volume
                ├── camera_array.toml                   # 自动生成：capture volume 内部相机数组
                ├── image_points.csv                    # 自动生成：最终使用的图像点
                ├── world_points.csv                    # 自动生成：优化后的 3D 点
                ├── reprojection_errors.csv             # 自动生成：最终重投影误差
                └── calibration_report.toml             # 自动生成：运行报告
```

要求：管线只发现 `cam_N.mp4`，且 `calibration/intrinsic` 与 `calibration/extrinsic` 的 `cam_N` 集合必须完全一致；`videos/` 和 `camera_name_mapping.csv` 可选但推荐，用于读取 GoPro metadata 和回溯原始文件。

`--intrinsics-library` 是命令行必需参数，不放在工作区树里；当前使用 `/home/btsun/project/Calibration/intrinsics_library.toml`。

## 4. ChArUco 标定板配置

外参阶段优先读取：

```text
calibration/targets/extrinsic_charuco.toml
```

如果不存在，则回退到：

```text
calibration/targets/intrinsic_charuco.toml
```

当前使用的两个板只差尺寸，公共字段为：`columns=6`、`rows=4`、`dictionary="DICT_4X4_250"`、`units="cm"`、`aruco_scale=0.7`、`inverted=false`、`legacy_pattern=false`。

| 标定板 | `board_width` | `board_height` | `square_size_override_cm` |
| --- | ---: | ---: | ---: |
| 36x24 cm，6 cm 方格 | `36.0` | `24.0` | `6.0` |
| 60x40 cm，10 cm 方格 | `60.0` | `40.0` | `10.0` |

## 5. 推荐完整运行命令

首次完整计算或改过抽点相关逻辑时使用：

```bash
PATH="/home/btsun/.conda/envs/cali/bin:$PATH" \
PYTHONPATH="/home/btsun/project/Calibration/dependencies/caliscope/src" \
PYTHONUNBUFFERED=1 \
/home/btsun/.conda/envs/cali/bin/python -u -m caliscope.pipelines.workspace_calibration \
  --workspace <workspace> \
  --intrinsics-library /home/btsun/project/Calibration/intrinsics_library.toml \
  --read-metadata --no-source-cam-id-fallback \
  --extrinsic-frame-step 5 \
  --reuse-existing-sync --force-image-points --force-capture-volume \
  --filter-scope overall --filter-percentile 10 --filter-sigma 2 \
  --max-nfev 2000 --align-to-object \
  --no-progress --log-level INFO
```

只想查看哪些阶段会复用或重算时，在同一模板中加入 `--plan-only`。

### 5.1 可选 OptiTrack 12D 坐标系对齐

外参完成并写出 `capture_volume/world_points.csv` 后，可以在同一次 workflow 中追加 OptiTrack/Motive CSV 对齐。该阶段不改变相机外参，只新增坐标系转换输出并把摘要写入 `calibration_report.toml`。

```bash
PATH="/home/btsun/.conda/envs/cali/bin:$PATH" \
PYTHONPATH="/home/btsun/project/Calibration/dependencies/caliscope/src" \
PYTHONUNBUFFERED=1 \
/home/btsun/.conda/envs/cali/bin/python -u -m caliscope.pipelines.workspace_calibration \
  --workspace <workspace> \
  --intrinsics-library /home/btsun/project/Calibration/intrinsics_library.toml \
  --read-metadata --no-source-cam-id-fallback \
  --filter-scope overall --filter-percentile 10 --filter-sigma 2 \
  --optitrack-csv /home/btsun/project/Calibration/src/mocap_rgb_projection/data/calibration-5-21.csv \
  --optitrack-lambda-xy-list "0,0.1,0.2,0.5,1,10,100" \
  --optitrack-select-lambda 0.2 \
  --no-progress --log-level INFO
```

默认输出目录：

```text
calibration/extrinsic/optitrack_alignment_12d
```

主要输出：

| 文件 | 作用 |
| --- | --- |
| `optitrack_to_camera_world_alignment.toml` | 唯一的可复用结果文件；包含时间偏移、OptiTrack 到 camera-world 的正反 Sim(3) 变换，以及标定板 12D marker-to-corner 修正参数。 |
| `optitrack_to_camera_world_alignment_report.md` | 人看的质量报告；包含最终误差、lambda sweep、marker offset 表和诊断 CSV 说明。 |
| `diagnostics/optitrack_to_camera_world_alignment_point_errors.csv` | 可选排查文件；每一行是一个标定板内角点观测的 3D 对齐残差。 |
| `diagnostics/optitrack_to_camera_world_alignment_frame_errors.csv` | 可选排查文件；每一行是一个 world_points 帧的 mean/RMSE/p95/max 残差。 |
| `calibration_report.toml` 的 `[optitrack_alignment]` | workflow 汇总：RMSE、`lambda_xy`、时间偏移、输出路径。 |

当前推荐 `--optitrack-select-lambda 0.2`。普通 OptiTrack 点使用 `optitrack_to_camera_world_alignment.toml` 中的全局 Sim(3) 与时间偏移；`calibration_board_marker_correction` 只描述该标定板四个 marker 中心到纸面角点的局部修正，不应套到其他物体。

## 6. 完整阶段说明

| 阶段 | 主要代码 | 输入 | 输出 |
| --- | --- | --- | --- |
| 发现视频 | `_discover_camera_videos` | `calibration/intrinsic`、`calibration/extrinsic` | 当前 `cam_id` 列表 |
| 读取标定板 | `_load_extrinsic_charuco` | target TOML | `Charuco` 对象 |
| 建立相机数组 | `_build_camera_array` | 视频尺寸、旧 `camera_array.toml`、mapping CSV | 内存中的 `CameraArray` |
| 应用内参 | `_apply_intrinsics_profiles` | 内参库、GoPro 序列号 | 每台相机的矩阵和畸变 |
| 音频同步 | `_synchronize_extrinsic_recording` | 外参视频 | `timestamps.csv`、`sync_offsets.toml` |
| 同步抽点 | `_extract_or_load_extrinsic_points` | 同步时间线、ChArUco | `CHARUCO/image_points.csv` |
| 外参求解 | `_calibrate_capture_volume` | 图像点、内参 | pose、3D 点、误差报告 |
| 导出 | `to_toml`、`to_aniposelib_toml` | 最终 capture volume | 根目录相机数组 TOML |

最终运行报告写入：

```text
calibration/extrinsic/capture_volume/calibration_report.toml
```

## 7. 音频同步

外参视频通过音频同步，成功后生成：

| 文件 | 作用 |
| --- | --- |
| `calibration/extrinsic/timestamps.csv` | 同步后的帧时间线，抽 ChArUco 点时使用。 |
| `calibration/extrinsic/sync_offsets.toml` | 同步摘要，包含参考相机、重叠时间段、各相机 offset。 |

复用规则：

| 参数 | 行为 |
| --- | --- |
| `--resume` | 默认开启。已有同步文件时复用。 |
| `--reuse-existing-sync` | 即使用 `--no-resume`，也允许复用已有同步文件。 |
| `--force-sync` | 强制重做音频同步。 |

Kitchen 当前应复用已有同步文件，除非源视频或同步算法变化。

## 8. 内参库复用

当前内参库：

```text
/home/btsun/project/Calibration/intrinsics_library.toml
```

支持的 `--intrinsics-library` 输入：

| 类型 | 说明 |
| --- | --- |
| 单个 TOML | 当前常用方式。 |
| TOML 目录 | 读取目录下所有 TOML profile。 |
| Caliscope 工作区 | 如果目录含 `camera_array.toml`，从其中提取内参。 |

匹配优先级：

| 优先级 | 方法 | 当前建议 |
| --- | --- | --- |
| 1 | `serial_number` | 必须优先使用。 |
| 2 | `source_cam_id` fallback | 只用于无序列号的临时场景。真实 GoPro 运行禁用。 |

真实运行必须使用：

```bash
--read-metadata --no-source-cam-id-fallback
```

报告中理想匹配类似：

```toml
[[intrinsics.matches]]
cam_id = 4
status = "matched"
method = "serial"
serial_number = "C3531325678402"
profile_serial = "C3531325678402"
adaptation = "exact_size"
camera_size = [1920, 1080]
profile_size = [1920, 1080]
```

内参适配方式：

| adaptation | 含义 |
| --- | --- |
| `exact_size` | profile 尺寸与视频尺寸完全一致，当前本地正常情况。 |
| `scaled:<sx>x,<sy>y` | 同宽高比缩放，焦距和主点同步缩放。 |
| `rotated_90ccw_scaled:<sx>x,<sy>y` | 横竖屏互换时先旋转内参再缩放。 |

如果无法匹配且 `--calibrate-missing` 开启，管线会用 `calibration/intrinsic/cam_N.mp4` 补算内参。当前 curated GoPro 数据不应依赖该兜底，出现 missing 应先查 metadata 和内参库。

## 9. 外参 ChArUco 抽点

默认抽帧：

```bash
--extrinsic-frame-step 5
```

输出：

| 文件 | 作用 |
| --- | --- |
| `calibration/extrinsic/CHARUCO/image_points.csv` | 同步后的 ChArUco 2D 检测点。 |
| `calibration/extrinsic/CHARUCO/image_points.meta.toml` | cache manifest，记录 timestamps、cam IDs、frame step。 |

缓存参数：

| 参数 | 行为 |
| --- | --- |
| `--resume` | manifest 匹配时复用 `image_points.csv`。 |
| `--reuse-image-points` | 即使没有匹配 manifest 也允许复用，需人工确认兼容。 |
| `--force-image-points` | 强制重抽并覆盖旧点。 |

更换 OpenCV/PyAV 后端、旋转处理、target TOML、`--extrinsic-frame-step` 后，必须使用 `--force-image-points`。

## 10. 外参求解和过滤阈值

外参阶段顺序：

| 步骤 | 说明 |
| --- | --- |
| Bootstrap | 用共享 ChArUco 观测建立相机位姿初值。 |
| First BA | 用全部接受点做第一轮 bundle adjustment。 |
| Filter | 按重投影误差删除异常点或异常同步帧。 |
| Second BA | 在过滤后点集上重新优化。 |
| Alignment | `--align-to-object` 时把坐标系对齐到某一帧 ChArUco 板。 |

当前推荐：

```bash
--filter-scope overall --filter-percentile 10 --filter-sigma 2
```

含义：

| 参数 | 含义 |
| --- | --- |
| `--filter-scope overall` | 全局点级阈值，所有 posed cameras 共用一个误差分布。 |
| `--filter-percentile 10` | 第一轮 BA 后删除最差 10% 的点观测。 |
| `--filter-sigma 2` | 在剩余点中继续删除高于 `median + 2 * robust_upper_sigma` 的点。 |

可选 scope：

| scope | 行为 | 何时使用 |
| --- | --- | --- |
| `per_camera` | 每台相机单独删除最差百分位，代码默认值。 | 某些相机误差分布明显不同。 |
| `overall` | 全局点级删除，当前推荐。 | 当前本地标定默认策略。 |
| `sync_index` | 按整帧 RMS 删除同步帧。 | 某些同步帧整体模糊、遮挡或板姿态异常。 |

`sync_index` 的 `--filter-percentile` 上限为 30。配合 `--filter-sigma` 时，只删除高于 frame-level `median + N * robust_upper_sigma` 的帧，percentile 只是最大删除上限。

过滤统计写在 `calibration_report.toml` 的 `[capture_volume.filter_stats]`。

## 11. Partial 外参

默认要求所有相机都连入外参图，否则失败。需要保留 partial 结果时显式使用：

```bash
--allow-partial-extrinsics
```

partial 只用于诊断或临时下游，不可当作完整相机阵列。未成功求 pose 的相机会记录在：

```text
stage_plan.capture_volume.unposed_cameras
```

当前 Kitchen full 15 路未完成，`cam_12` unposed。可用结果是 partial 14/15，使用时必须排除 `cam_12`。

## 12. GoPro 旋转 metadata 和 no-auto 坐标

本地统一约定：所有内参、检测点、重投影、可视化都应使用 encoded raster 坐标，不使用 display auto-rotation 后的图像坐标。

原因：GoPro MP4 可能携带 display rotation metadata。OpenCV 默认可能自动应用该 metadata，导致解码帧坐标和内参库坐标不一致。

当前实现会在 OpenCV 打开视频后关闭自动旋转：

```python
capture = cv2.VideoCapture(str(source_path))
capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
```

风险说明：

| metadata | 风险 |
| --- | --- |
| 90 或 270 | auto 后尺寸变成 `1080x1920`，和 `1920x1080` 内参明显不一致。 |
| 180 | 尺寸仍是 `1920x1080`，但图像中心旋转，最容易静默出错。 |
| 混用读取器 | 抽点、重投影、可视化如果有的 auto、有的 no-auto，会出现错位。 |

诊断脚本如直接使用 OpenCV，也必须设置 `CAP_PROP_ORIENTATION_AUTO=0`。推荐复用 `FrameSource` 或 `read_video_properties`。

当前内参库已验证为 no-auto 坐标。InterHyper `cam_17`、序列号 `C3531325678286`、metadata `270` 的检查结果：

| 数据和内参 | 平均重投影 RMSE |
| --- | --- |
| no-auto 点 + 当前库内参 | 约 `0.473 px` |
| auto 点 + 当前库内参 | 约 `14.876 px` |
| auto 点 + 旋转后内参 | 约 `0.487 px` |

结论：不要全局旋转当前内参库。正确做法是所有下游读取都保持 no-auto。

## 13. 输出文件说明

| 文件 | 说明 |
| --- | --- |
| `<workspace>/camera_array.toml` | 最终 Caliscope 相机数组。 |
| `<workspace>/camera_array_aniposelib.toml` | aniposelib 兼容输出。 |
| `CHARUCO/image_points.csv` | 外参抽点 cache。 |
| `CHARUCO/image_points.meta.toml` | 外参抽点 cache manifest。 |
| `capture_volume/camera_array.toml` | capture volume 内部相机数组。 |
| `capture_volume/image_points.csv` | 最终 capture volume 使用的图像点。 |
| `capture_volume/world_points.csv` | 优化得到的 3D ChArUco 点。 |
| `capture_volume/reprojection_errors.csv` | 最终每个观测的重投影误差。 |
| `capture_volume/calibration_report.toml` | 最重要的运行报告和质量摘要。 |

优先检查 `calibration_report.toml` 中的 `intrinsics.matches`、`stage_plan`、`resume`、`sync`、`extrinsic_points`、`capture_volume.final_rmse`、`capture_volume.filter_stats`、`unposed_cameras`。

## 14. 复跑策略

| 变化 | 推荐参数 |
| --- | --- |
| 只看计划 | `--plan-only` |
| 源视频变化 | `--force-sync --force-image-points --force-capture-volume` |
| 同步结果要重算 | `--force-sync --force-image-points --force-capture-volume` |
| target TOML 变化 | `--reuse-existing-sync --force-image-points --force-capture-volume` |
| 视频读取或旋转处理变化 | `--reuse-existing-sync --force-image-points --force-capture-volume` |
| 只改过滤阈值 | `--reuse-existing-sync --reuse-image-points --force-capture-volume` |
| 只改 BA 迭代次数 | `--reuse-existing-sync --reuse-image-points --force-capture-volume` |
| 已有完整结果可信 | 使用默认 `--resume`，不加 force。 |

## 15. 当前工作区套用方式

把第 5 节模板中的 `<workspace>` 换成下表路径。Desk 建议强制重抽点；Kitchen 和 60x40 当前可复用已有点后只重算 capture volume。

| 工作区 | 额外说明 |
| --- | --- |
| `/home/btsun/project/Calibration/Desk_6x6_0526/Desk_6x6_0526_workspace` | 使用 `--force-image-points --force-capture-volume`。 |
| `/home/btsun/project/Calibration/Kitchen/Kitchen_workspace` | 使用 `--reuse-image-points --force-capture-volume --allow-partial-extrinsics`。 |
| `/home/btsun/project/Calibration/60x40CharUco0527/GoPro-1779872531_workspace` | 使用 `--reuse-image-points --force-capture-volume`。 |

## 16. 当前本地结果

| 工作区 | 相机 | 标定板 | Final RMSE | 状态 |
| --- | ---: | --- | ---: | --- |
| `Desk_6x6_0526/Desk_6x6_0526_workspace` | 8/8 | 36x24 cm，6 cm 方格 | `1.4661 px` | 完整 |
| `Kitchen/Kitchen_workspace` | 14/15 posed | 60x40 cm，10 cm 方格 | `1.6458 px` | partial，`cam_12` unposed |
| `60x40CharUco0527/GoPro-1779872531_workspace` | 8/8 | 60x40 cm，10 cm 方格 | `1.9839 px` | 完整 |

## 17. 实现特性和性能

| 特性 | 说明 |
| --- | --- |
| OpenCV 后端 | 当前 `FrameSource` 使用 OpenCV `VideoCapture`，并强制 no-auto rotation。 |
| 顺序读取优化 | `read_frame_at()` 对中等正向跳转使用 `grab()` 前进再 `read()`，比反复 seek 更快。 |
| manifest 保护 | 默认只在 timestamps、cam IDs、frame step 匹配时复用抽点 cache。 |
| no-progress 快路径 | `--no-progress` 配合内置 `CharucoTracker` 可走 per-camera series 抽取路径。 |
| runtime 控制 | `--workers` 控制相机 worker 数，`--opencv-threads` 控制 OpenCV 内部线程数。 |

核心文件：

| 文件 | 作用 |
| --- | --- |
| `src/caliscope/pipelines/workspace_calibration.py` | 非 GUI CLI 和工作流编排。 |
| `src/caliscope/recording/video_utils.py` | 视频元数据读取和 no-auto 打开。 |
| `src/caliscope/recording/frame_source.py` | OpenCV 帧读取。 |
| `src/caliscope/core/process_synchronized_recording.py` | 同步视频抽 ChArUco 点。 |
| `src/caliscope/core/capture_volume.py` | 外参 bootstrap、BA、误差和过滤。 |

## 18. 诊断和常见问题

| 问题 | 常见原因 | 处理 |
| --- | --- | --- |
| 找不到 `cam_N.mp4` | 文件未按规范放入 intrinsic/extrinsic 目录。 | 先链接或复制成 `cam_N.mp4`。 |
| 内参未匹配 | metadata 没读到序列号或库中没有该序列号。 | 检查 `camera_name_mapping.csv`、`videos/`、`--read-metadata`。 |
| 可视化点和图像旋转错位 | 诊断脚本用了 OpenCV 默认 auto-rotation。 | 使用 `FrameSource` 或设置 `CAP_PROP_ORIENTATION_AUTO=0`。 |
| 有相机 unposed | 标定板没有把该相机和主图连通。 | 补拍桥接视角，或只为诊断使用 partial。 |
| 改了参数但结果没变 | cache 被复用。 | 使用 `--force-image-points --force-capture-volume`。 |
| 180 度视频难发现错误 | auto 后尺寸不变但坐标中心旋转。 | 保持 no-auto 并重新抽点。 |

Kitchen worst-frame 重投影诊断输出：`.temp/kitchen_reprojection_worst_frames_10pct_2sigma/images` 和 `.temp/kitchen_reprojection_worst_frames_10pct_2sigma/selected_frames.csv`。

后续修改应继续按关注点拆分：过滤逻辑、视频读取后端、调度性能、文档更新分别处理，便于回滚和审查。
