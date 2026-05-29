"""12D OptiTrack/world_points alignment for non-GUI calibration workflows."""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rtoml
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


SQUARE_SIZE_M = 0.06
BOARD_SQUARE_COLS = 6
BOARD_SQUARE_ROWS = 4
INNER_GRID_COLS = BOARD_SQUARE_COLS - 1
INNER_GRID_ROWS = BOARD_SQUARE_ROWS - 1
BOARD_WIDTH_M = BOARD_SQUARE_COLS * SQUARE_SIZE_M
BOARD_HEIGHT_M = BOARD_SQUARE_ROWS * SQUARE_SIZE_M
EXPECTED_POINT_IDS = tuple(range(INNER_GRID_COLS * INNER_GRID_ROWS))
MARKER_TYPE_PRIORITY = ("Rigid Body Marker", "Marker")
RANDOM_SEED = 20260521
TARGET_RMSE_MM = 3.0
OUTPUT_BASENAME = "optitrack_to_camera_world_alignment"
ALIGNMENT_TOML_NAME = f"{OUTPUT_BASENAME}.toml"
REPORT_MD_NAME = f"{OUTPUT_BASENAME}_report.md"
DIAGNOSTICS_DIR_NAME = "diagnostics"


@dataclass(frozen=True)
class OptitrackAlignmentConfig:
    """Configuration for 12D OptiTrack/world_points alignment."""

    world_points_csv: Path
    optitrack_csv: Path
    output_dir: Path
    offset_min: float = -20.0
    offset_max: float = 20.0
    coarse_offset_step: float = 0.25
    shared_height_min: float = -0.03
    shared_height_max: float = 0.03
    coarse_height_count: int = 7
    offset_12d_abs_max: float = 0.03
    offset_refine_window: float = 1.0
    lambda_xy_list: str = "0,0.1,0.2,0.5,1,10,100"
    select_lambda: str = "0.2"
    test_ratio: float = 0.33
    seed: int = RANDOM_SEED
    max_world_grid_rmse_m: float = 0.005
    min_points_per_fit_frame: int = 15
    min_overlap_frames: int = 40
    max_coarse_frames: int = 120
    top_candidates_to_refine: int = 8
    equal_height_maxiter: int = 260
    offset_12d_maxiter: int = 500
    allow_global_scale: bool = True


@dataclass(frozen=True)
class WorldFrame:
    sync_index: int
    frame_time: float
    point_ids: np.ndarray
    points_world: np.ndarray
    grid_rmse_m: float


@dataclass(frozen=True)
class Candidate:
    permutation: tuple[int, int, int, int]
    offset_s: float
    marker_height_m: float
    rmse_m: float
    n_frames: int
    n_points: int
    scale: float


@dataclass(frozen=True)
class OffsetFitResult:
    lambda_xy: float
    permutation: tuple[int, int, int, int]
    offset_s: float
    offsets_4x3_m: np.ndarray
    train_rmse_m: float
    test_rmse_m: float
    all_rmse_m: float
    train_mean_m: float
    test_mean_m: float
    all_mean_m: float
    xy_rms_m: float
    z_mean_m: float
    z_std_m: float
    success: bool
    message: str
    n_iter: int


@dataclass(frozen=True)
class TransformFit:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    source_points: np.ndarray
    target_points: np.ndarray
    predicted_points: np.ndarray
    errors_m: np.ndarray
    meta: pd.DataFrame


def build_inner_grid_local() -> np.ndarray:
    points = []
    for row in range(INNER_GRID_ROWS):
        for col in range(INNER_GRID_COLS):
            points.append([(col + 1) * SQUARE_SIZE_M, (row + 1) * SQUARE_SIZE_M, 0.0])
    return np.asarray(points, dtype=np.float64)


INNER_GRID_LOCAL = build_inner_grid_local()


def fit_similarity(source: np.ndarray, target: np.ndarray, allow_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * R @ source + t with an Umeyama Sim(3)."""

    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if len(source) < 3 or len(source) != len(target):
        raise ValueError("fit_similarity needs matched arrays with at least 3 points")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / len(source)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign_matrix = np.eye(3)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        sign_matrix[-1, -1] = -1.0
    rotation = u_matrix @ sign_matrix @ vt_matrix
    if allow_scale:
        variance = np.sum(source_centered * source_centered) / len(source)
        scale = float(np.trace(np.diag(singular_values) @ sign_matrix) / variance)
    else:
        scale = 1.0
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return scale * (rotation @ points.T).T + translation[None, :]


def inverse_similarity(scale: float, rotation: np.ndarray, translation: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    inverse_scale = 1.0 / float(scale)
    inverse_rotation = rotation.T
    inverse_translation = -inverse_scale * (rotation.T @ translation)
    return inverse_scale, inverse_rotation, inverse_translation


def summarize_errors(errors_m: np.ndarray) -> dict[str, float | int]:
    errors = np.asarray(errors_m, dtype=np.float64)
    if errors.size == 0:
        return {
            "n": 0,
            "mean_m": math.nan,
            "rmse_m": math.nan,
            "median_m": math.nan,
            "p90_m": math.nan,
            "p95_m": math.nan,
            "max_m": math.nan,
        }
    return {
        "n": int(errors.size),
        "mean_m": float(errors.mean()),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.quantile(errors, 0.90)),
        "p95_m": float(np.quantile(errors, 0.95)),
        "max_m": float(errors.max()),
    }


def local_grid_fit_rmse(point_ids: np.ndarray, points_world: np.ndarray) -> float:
    if len(point_ids) < 3:
        return math.inf
    local = INNER_GRID_LOCAL[point_ids]
    try:
        scale, rotation, translation = fit_similarity(local, points_world, allow_scale=True)
    except (ValueError, np.linalg.LinAlgError):
        return math.inf
    predicted = apply_similarity(local, scale, rotation, translation)
    return float(np.sqrt(np.mean(np.linalg.norm(predicted - points_world, axis=1) ** 2)))


def load_world_frames(path: Path) -> list[WorldFrame]:
    df = pd.read_csv(path)
    required = {"sync_index", "point_id", "x_coord", "y_coord", "z_coord", "frame_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"world_points CSV is missing columns: {sorted(missing)}")

    df = df.copy()
    df["sync_index"] = df["sync_index"].astype(int)
    df["point_id"] = df["point_id"].astype(int)
    df = df[df["point_id"].isin(EXPECTED_POINT_IDS)]

    frames: list[WorldFrame] = []
    for sync_index, group in df.groupby("sync_index", sort=True):
        group = group.sort_values("point_id").drop_duplicates("point_id", keep="first")
        point_ids = group["point_id"].to_numpy(dtype=np.int64)
        points_world = group[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float64)
        frame_time = float(group["frame_time"].median())
        frames.append(WorldFrame(int(sync_index), frame_time, point_ids, points_world, local_grid_fit_rmse(point_ids, points_world)))
    frames.sort(key=lambda frame: frame.frame_time)
    return frames


def marker_number(name: str) -> int | None:
    """Return marker number 1..4 for Motive names such as Marker 001 or Marker001."""

    text = str(name or "")
    matches = re.findall(r"(?:^|[^a-z])marker[\s:_-]*0*([1-4])(?:\b|[^0-9])", text, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def load_optitrack_markers(
    path: Path,
    marker_type_priority: Sequence[str] = MARKER_TYPE_PRIORITY,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load calibration-board Marker001..004 XYZ positions from a Motive CSV."""

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.reader(file)
        try:
            header_rows = [next(reader) for _ in range(8)]
        except StopIteration as exc:
            raise ValueError(f"OptiTrack CSV does not contain the expected Motive header rows: {path}") from exc
        types = header_rows[2]
        names = header_rows[3]
        channels = header_rows[6]
        axes = header_rows[7]

        columns: list[dict[str, Any]] = []
        current_type = ""
        current_name = ""
        for index in range(len(names)):
            if index < len(types) and types[index].strip():
                current_type = types[index].strip()
            if names[index].strip():
                current_name = names[index].strip()
            columns.append(
                {
                    "index": index,
                    "type": current_type,
                    "name": current_name,
                    "channel": channels[index].strip() if index < len(channels) else "",
                    "axis": axes[index].strip() if index < len(axes) else "",
                }
            )

        selected: dict[tuple[int, str], dict[str, Any]] | None = None
        selected_type = ""
        for marker_type in marker_type_priority:
            trial: dict[tuple[int, str], dict[str, Any]] = {}
            for column in columns:
                number = marker_number(column["name"])
                if number is None or column["type"] != marker_type:
                    continue
                if column["channel"] != "Position" or column["axis"] not in {"X", "Y", "Z"}:
                    continue
                trial[(number, column["axis"])] = column
            if all((number, axis) in trial for number in range(1, 5) for axis in "XYZ"):
                selected = trial
                selected_type = marker_type
                break
        if selected is None:
            raise ValueError("OptiTrack CSV does not contain complete Marker001..004 XYZ Position columns")

        rows: list[dict[str, float | int]] = []
        for row in reader:
            if len(row) < 2:
                continue
            try:
                record: dict[str, float | int] = {"frame": int(float(row[0])), "time": float(row[1])}
            except ValueError:
                continue
            ok = True
            for number in range(1, 5):
                for axis in "XYZ":
                    column_index = selected[(number, axis)]["index"]
                    try:
                        record[f"m{number}_{axis.lower()}"] = float(row[column_index])
                    except (ValueError, IndexError):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                rows.append(record)

    opti = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    if opti.empty:
        raise ValueError("OptiTrack CSV does not contain valid marker rows")

    column_sources = {
        "selected_marker_type": selected_type,
        **{
            f"Marker{number:03d}_{axis}": f"{selected[(number, axis)]['type']}::{selected[(number, axis)]['name']}[{axis}]"
            for number in range(1, 5)
            for axis in "XYZ"
        },
    }
    return opti, column_sources


def interpolate_markers(opti: pd.DataFrame, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=np.float64)
    opti_times = opti["time"].to_numpy(dtype=np.float64)
    valid = (times >= opti_times[0]) & (times <= opti_times[-1])
    out = np.full((len(times), 4, 3), np.nan, dtype=np.float64)
    for marker_index in range(1, 5):
        for dim_index, axis in enumerate("xyz"):
            values = opti[f"m{marker_index}_{axis}"].to_numpy(dtype=np.float64)
            out[valid, marker_index - 1, dim_index] = np.interp(times[valid], opti_times, values)
    valid &= np.isfinite(out).all(axis=(1, 2))
    return out, valid


def ordered_marker_axes(ordered_markers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = ordered_markers[:, 1] - ordered_markers[:, 0]
    e_x = x_axis / np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-12)

    y_raw = ordered_markers[:, 3] - ordered_markers[:, 0]
    y_orth = y_raw - np.sum(y_raw * e_x, axis=1, keepdims=True) * e_x
    e_y = y_orth / np.maximum(np.linalg.norm(y_orth, axis=1, keepdims=True), 1e-12)

    e_z = np.cross(e_x, e_y)
    e_z = e_z / np.maximum(np.linalg.norm(e_z, axis=1, keepdims=True), 1e-12)
    return e_x, e_y, e_z


def apply_corner_offsets(ordered_markers: np.ndarray, offsets_4x3_m: np.ndarray) -> np.ndarray:
    ordered_markers = np.asarray(ordered_markers, dtype=np.float64).reshape(-1, 4, 3)
    offsets = np.asarray(offsets_4x3_m, dtype=np.float64).reshape(4, 3)
    e_x, e_y, e_z = ordered_marker_axes(ordered_markers)
    corners = ordered_markers.copy()
    for corner_id in range(4):
        dx, dy, dz = offsets[corner_id]
        corners[:, corner_id, :] = ordered_markers[:, corner_id, :] + dx * e_x + dy * e_y + dz * e_z
    return corners


def inner_grid_from_corners_batch(corners: np.ndarray) -> np.ndarray:
    p00 = corners[:, 0]
    p10 = corners[:, 1]
    p11 = corners[:, 2]
    p01 = corners[:, 3]
    grids = []
    for local in INNER_GRID_LOCAL:
        u = float(local[0] / BOARD_WIDTH_M)
        v = float(local[1] / BOARD_HEIGHT_M)
        grids.append((1.0 - u) * (1.0 - v) * p00 + u * (1.0 - v) * p10 + u * v * p11 + (1.0 - u) * v * p01)
    return np.stack(grids, axis=1)


def marker_height_offsets_4x3(marker_height_m: float) -> np.ndarray:
    offsets = np.zeros((4, 3), dtype=np.float64)
    offsets[:, 2] = float(marker_height_m)
    return offsets


def select_fit_frames(frames: list[WorldFrame], max_grid_rmse_m: float, min_points: int, min_overlap_frames: int) -> list[WorldFrame]:
    selected = [frame for frame in frames if len(frame.point_ids) >= min_points and frame.grid_rmse_m <= max_grid_rmse_m]
    if len(selected) < min_overlap_frames:
        raise ValueError(
            f"Not enough high-quality world_points frames: {len(selected)} < {min_overlap_frames}. "
            "Check the CSV or relax max_world_grid_rmse_m/min_points_per_fit_frame."
        )
    return selected


def sample_records(records: list[WorldFrame], max_records: int) -> list[WorldFrame]:
    if len(records) <= max_records:
        return records
    indexes = np.linspace(0, len(records) - 1, max_records, dtype=int)
    return [records[index] for index in indexes]


def train_test_split_records(records: list[WorldFrame], test_ratio: float, seed: int) -> tuple[list[WorldFrame], list[WorldFrame]]:
    if test_ratio <= 0:
        return records, []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(records) * test_ratio)))
    test_idx = set(idx[:n_test].tolist())
    train = [record for i, record in enumerate(records) if i not in test_idx]
    test = [record for i, record in enumerate(records) if i in test_idx]
    train.sort(key=lambda frame: frame.frame_time)
    test.sort(key=lambda frame: frame.frame_time)
    return train, test


def build_source_target(
    records: list[WorldFrame],
    opti: pd.DataFrame,
    permutation: tuple[int, int, int, int],
    offset_s: float,
    offsets_4x3_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    if not records:
        return np.empty((0, 3)), np.empty((0, 3)), pd.DataFrame(), 0
    times = np.asarray([record.frame_time + offset_s for record in records], dtype=np.float64)
    markers, valid = interpolate_markers(opti, times)
    valid_indexes = np.where(valid)[0]
    if len(valid_indexes) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), pd.DataFrame(), 0

    selected_markers = markers[valid_indexes][:, list(permutation), :]
    corners = apply_corner_offsets(selected_markers, offsets_4x3_m)
    source_grid = inner_grid_from_corners_batch(corners)

    source_blocks = []
    target_blocks = []
    meta_rows = []
    for source_index, record_index in enumerate(valid_indexes):
        record = records[record_index]
        source_points = source_grid[source_index, record.point_ids]
        target_points = record.points_world
        source_blocks.append(source_points)
        target_blocks.append(target_points)
        for local_index, point_id in enumerate(record.point_ids):
            meta_rows.append(
                {
                    "sync_index": record.sync_index,
                    "frame_time": record.frame_time,
                    "aligned_opti_time": float(times[record_index]),
                    "point_id": int(point_id),
                    "world_grid_rmse_mm": record.grid_rmse_m * 1000.0,
                    "source_opti_x": float(source_points[local_index, 0]),
                    "source_opti_y": float(source_points[local_index, 1]),
                    "source_opti_z": float(source_points[local_index, 2]),
                    "target_camera_world_x": float(target_points[local_index, 0]),
                    "target_camera_world_y": float(target_points[local_index, 1]),
                    "target_camera_world_z": float(target_points[local_index, 2]),
                }
            )

    return np.vstack(source_blocks), np.vstack(target_blocks), pd.DataFrame(meta_rows), int(len(valid_indexes))


def fit_transform_for_params(
    records: list[WorldFrame],
    opti: pd.DataFrame,
    permutation: tuple[int, int, int, int],
    offset_s: float,
    offsets_4x3_m: np.ndarray,
    allow_scale: bool,
) -> TransformFit | None:
    source, target, meta, _ = build_source_target(records, opti, permutation, offset_s, offsets_4x3_m)
    if len(source) < 3:
        return None
    try:
        scale, rotation, translation = fit_similarity(source, target, allow_scale=allow_scale)
    except (ValueError, np.linalg.LinAlgError):
        return None
    predicted = apply_similarity(source, scale, rotation, translation)
    errors = np.linalg.norm(predicted - target, axis=1)
    return TransformFit(scale, rotation, translation, source, target, predicted, errors, meta)


def evaluate_params_with_train_transform(
    train_records: list[WorldFrame],
    eval_records: list[WorldFrame],
    opti: pd.DataFrame,
    permutation: tuple[int, int, int, int],
    offset_s: float,
    offsets_4x3_m: np.ndarray,
    allow_scale: bool,
) -> tuple[np.ndarray, TransformFit | None, pd.DataFrame]:
    train_fit = fit_transform_for_params(train_records, opti, permutation, offset_s, offsets_4x3_m, allow_scale=allow_scale)
    if train_fit is None:
        return np.asarray([], dtype=np.float64), None, pd.DataFrame()
    source_eval, target_eval, meta_eval, _ = build_source_target(eval_records, opti, permutation, offset_s, offsets_4x3_m)
    if len(source_eval) == 0:
        return np.asarray([], dtype=np.float64), train_fit, meta_eval
    predicted = apply_similarity(source_eval, train_fit.scale, train_fit.rotation, train_fit.translation)
    errors = np.linalg.norm(predicted - target_eval, axis=1)
    meta_eval = meta_eval.copy()
    meta_eval["pred_camera_world_x"] = predicted[:, 0]
    meta_eval["pred_camera_world_y"] = predicted[:, 1]
    meta_eval["pred_camera_world_z"] = predicted[:, 2]
    meta_eval["error_m"] = errors
    meta_eval["error_mm"] = errors * 1000.0
    return errors, train_fit, meta_eval


def equal_height_candidate_rmse(
    records: list[WorldFrame],
    opti: pd.DataFrame,
    permutation: tuple[int, int, int, int],
    offset_s: float,
    marker_height_m: float,
    allow_scale: bool,
    min_overlap_frames: int,
) -> tuple[float, int, int, float]:
    offsets = marker_height_offsets_4x3(marker_height_m)
    source, target, _, n_frames = build_source_target(records, opti, permutation, offset_s, offsets)
    if n_frames < min_overlap_frames or len(source) < 3:
        return 1e6 + (min_overlap_frames - n_frames) * 10.0, 0, n_frames, math.nan
    try:
        scale, rotation, translation = fit_similarity(source, target, allow_scale=allow_scale)
    except (ValueError, np.linalg.LinAlgError):
        return 1e6, int(len(source)), n_frames, math.nan
    predicted = apply_similarity(source, scale, rotation, translation)
    errors = np.linalg.norm(predicted - target, axis=1)
    return float(np.sqrt(np.mean(errors**2))), int(len(source)), int(n_frames), float(scale)


def optimize_equal_height_seed(records: list[WorldFrame], opti: pd.DataFrame, config: OptitrackAlignmentConfig) -> Candidate:
    coarse_records = sample_records(records, config.max_coarse_frames)
    offsets = np.arange(config.offset_min, config.offset_max + config.coarse_offset_step * 0.5, config.coarse_offset_step)
    heights = np.linspace(config.shared_height_min, config.shared_height_max, config.coarse_height_count)
    permutations = list(itertools.permutations(range(4)))
    logger.info(
        "Coarse equal-height search: records=%d permutations=%d offsets=%d heights=%d",
        len(coarse_records),
        len(permutations),
        len(offsets),
        len(heights),
    )

    coarse_results: list[Candidate] = []
    for permutation in permutations:
        best_for_perm: Candidate | None = None
        for height_m in heights:
            for offset_s in offsets:
                rmse_m, n_points, n_frames, scale = equal_height_candidate_rmse(
                    coarse_records,
                    opti,
                    permutation,
                    float(offset_s),
                    float(height_m),
                    allow_scale=config.allow_global_scale,
                    min_overlap_frames=config.min_overlap_frames,
                )
                candidate = Candidate(
                    permutation,
                    float(offset_s),
                    float(height_m),
                    rmse_m,
                    n_frames,
                    n_points,
                    scale,
                )
                if best_for_perm is None or candidate.rmse_m < best_for_perm.rmse_m:
                    best_for_perm = candidate
        if best_for_perm is not None:
            coarse_results.append(best_for_perm)
    coarse_results.sort(key=lambda item: item.rmse_m)

    refined_results: list[Candidate] = []
    for seed in coarse_results[: config.top_candidates_to_refine]:
        def objective(x: np.ndarray) -> float:
            rmse_m, _, _, _ = equal_height_candidate_rmse(
                records,
                opti,
                seed.permutation,
                float(x[0]),
                float(x[1]),
                allow_scale=config.allow_global_scale,
                min_overlap_frames=config.min_overlap_frames,
            )
            return rmse_m

        result = minimize(
            objective,
            np.asarray([seed.offset_s, seed.marker_height_m], dtype=np.float64),
            method="Nelder-Mead",
            options={"maxiter": config.equal_height_maxiter, "xatol": 1e-10, "fatol": 1e-12},
        )
        offset_s = float(result.x[0])
        marker_height_m = float(result.x[1])
        rmse_m, n_points, n_frames, scale = equal_height_candidate_rmse(
            records,
            opti,
            seed.permutation,
            offset_s,
            marker_height_m,
            allow_scale=config.allow_global_scale,
            min_overlap_frames=config.min_overlap_frames,
        )
        refined_results.append(Candidate(seed.permutation, offset_s, marker_height_m, rmse_m, n_frames, n_points, scale))

    if not refined_results:
        raise RuntimeError("Equal-height seed search did not produce any refined candidates")
    refined_results.sort(key=lambda item: item.rmse_m)
    best = refined_results[0]
    logger.info(
        "Equal-height seed: perm=%s offset=%.9fs height=%.6fmm rmse=%.6fmm scale=%.9f",
        best.permutation,
        best.offset_s,
        best.marker_height_m * 1000.0,
        best.rmse_m * 1000.0,
        best.scale,
    )
    return best


def parse_lambda_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("lambda_xy_list is empty")
    return values


def objective_12d(
    x: np.ndarray,
    train_records: list[WorldFrame],
    opti: pd.DataFrame,
    permutation: tuple[int, int, int, int],
    lambda_xy: float,
    config: OptitrackAlignmentConfig,
) -> float:
    offset_s = float(x[0])
    offsets = np.asarray(x[1:], dtype=np.float64).reshape(4, 3)
    if not (config.offset_min <= offset_s <= config.offset_max):
        return 1e6 + abs(offset_s) * 10.0
    if not np.all(np.isfinite(offsets)):
        return 1e6
    if np.max(np.abs(offsets)) > config.offset_12d_abs_max:
        return 1e6 + float(np.max(np.abs(offsets)))

    fit = fit_transform_for_params(train_records, opti, permutation, offset_s, offsets, allow_scale=config.allow_global_scale)
    if fit is None or len(fit.errors_m) == 0:
        return 1e6
    rmse2 = float(np.mean(fit.errors_m**2))
    xy2 = float(np.mean(offsets[:, :2] ** 2))
    return float(np.sqrt(rmse2 + lambda_xy * xy2))


def optimize_12d_offsets(
    train_records: list[WorldFrame],
    test_records: list[WorldFrame],
    all_records: list[WorldFrame],
    opti: pd.DataFrame,
    seed: Candidate,
    lambda_xy_values: list[float],
    config: OptitrackAlignmentConfig,
) -> list[OffsetFitResult]:
    results: list[OffsetFitResult] = []
    x0 = np.concatenate([[seed.offset_s], marker_height_offsets_4x3(seed.marker_height_m).reshape(-1)])
    lower = [max(config.offset_min, seed.offset_s - config.offset_refine_window)] + [-config.offset_12d_abs_max] * 12
    upper = [min(config.offset_max, seed.offset_s + config.offset_refine_window)] + [config.offset_12d_abs_max] * 12
    bounds = list(zip(lower, upper))

    previous_x = x0.copy()
    for lambda_xy in lambda_xy_values:
        logger.info("Optimize 12D offsets: lambda_xy=%g", lambda_xy)
        result = minimize(
            lambda x: objective_12d(x, train_records, opti, seed.permutation, lambda_xy, config),
            previous_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.offset_12d_maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 30},
        )
        x = np.asarray(result.x, dtype=np.float64)
        previous_x = x.copy()
        offset_s = float(x[0])
        offsets = x[1:].reshape(4, 3)

        train_errors, _, _ = evaluate_params_with_train_transform(
            train_records,
            train_records,
            opti,
            seed.permutation,
            offset_s,
            offsets,
            config.allow_global_scale,
        )
        test_errors, _, _ = evaluate_params_with_train_transform(
            train_records,
            test_records,
            opti,
            seed.permutation,
            offset_s,
            offsets,
            config.allow_global_scale,
        )
        all_errors, _, _ = evaluate_params_with_train_transform(
            train_records,
            all_records,
            opti,
            seed.permutation,
            offset_s,
            offsets,
            config.allow_global_scale,
        )

        train_summary = summarize_errors(train_errors)
        test_summary = summarize_errors(test_errors) if len(test_errors) else {"rmse_m": math.nan, "mean_m": math.nan}
        all_summary = summarize_errors(all_errors)
        xy_rms = float(np.sqrt(np.mean(offsets[:, :2] ** 2)))
        z_vals = offsets[:, 2]
        fit_result = OffsetFitResult(
            lambda_xy=lambda_xy,
            permutation=seed.permutation,
            offset_s=offset_s,
            offsets_4x3_m=offsets,
            train_rmse_m=float(train_summary["rmse_m"]),
            test_rmse_m=float(test_summary["rmse_m"]),
            all_rmse_m=float(all_summary["rmse_m"]),
            train_mean_m=float(train_summary["mean_m"]),
            test_mean_m=float(test_summary["mean_m"]),
            all_mean_m=float(all_summary["mean_m"]),
            xy_rms_m=xy_rms,
            z_mean_m=float(np.mean(z_vals)),
            z_std_m=float(np.std(z_vals)),
            success=bool(result.success),
            message=str(result.message),
            n_iter=int(getattr(result, "nit", -1)),
        )
        results.append(fit_result)
        logger.info(
            "  lambda=%g: train=%.3fmm test=%.3fmm all=%.3fmm xy_rms=%.3fmm",
            lambda_xy,
            fit_result.train_rmse_m * 1000.0,
            fit_result.test_rmse_m * 1000.0,
            fit_result.all_rmse_m * 1000.0,
            fit_result.xy_rms_m * 1000.0,
        )
    return results


def select_result(results: list[OffsetFitResult], select_lambda: str) -> OffsetFitResult:
    if select_lambda == "min_test":
        return min(results, key=lambda result: (math.inf if math.isnan(result.test_rmse_m) else result.test_rmse_m, result.xy_rms_m))
    if select_lambda == "min_all":
        return min(results, key=lambda result: (result.all_rmse_m, result.xy_rms_m))
    value = float(select_lambda)
    return min(results, key=lambda result: abs(result.lambda_xy - value))


def result_frame(results: list[OffsetFitResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "lambda_xy": result.lambda_xy,
                "permutation": str(result.permutation),
                "offset_s": result.offset_s,
                "train_rmse_mm": result.train_rmse_m * 1000.0,
                "test_rmse_mm": result.test_rmse_m * 1000.0,
                "all_rmse_mm": result.all_rmse_m * 1000.0,
                "train_mean_mm": result.train_mean_m * 1000.0,
                "test_mean_mm": result.test_mean_m * 1000.0,
                "all_mean_mm": result.all_mean_m * 1000.0,
                "xy_rms_mm": result.xy_rms_m * 1000.0,
                "z_mean_mm": result.z_mean_m * 1000.0,
                "z_std_mm": result.z_std_m * 1000.0,
                "success": result.success,
                "n_iter": result.n_iter,
                "message": result.message,
            }
        )
    return pd.DataFrame(rows).sort_values("lambda_xy")


def diagnostic_output_paths(out_dir: Path) -> dict[str, Path]:
    diagnostics_dir = out_dir / DIAGNOSTICS_DIR_NAME
    return {
        "point_errors": diagnostics_dir / f"{OUTPUT_BASENAME}_point_errors.csv",
        "frame_errors": diagnostics_dir / f"{OUTPUT_BASENAME}_frame_errors.csv",
    }


def summarize_world_quality(frames: list[WorldFrame], fit_frames: list[WorldFrame]) -> dict[str, float | int]:
    grid_rmse_mm = np.asarray([frame.grid_rmse_m * 1000.0 for frame in fit_frames], dtype=np.float64)
    all_points = sum(len(frame.point_ids) for frame in frames)
    fit_points = sum(len(frame.point_ids) for frame in fit_frames)
    return {
        "frames_total": int(len(frames)),
        "frames_used_for_fit": int(len(fit_frames)),
        "points_total": int(all_points),
        "points_used_for_fit": int(fit_points),
        "fit_frame_grid_rmse_mean_mm": float(np.mean(grid_rmse_mm)),
        "fit_frame_grid_rmse_median_mm": float(np.median(grid_rmse_mm)),
        "fit_frame_grid_rmse_p95_mm": float(np.quantile(grid_rmse_mm, 0.95)),
    }


def write_error_tables(out_dir: Path, final_fit: TransformFit) -> dict[str, float | int]:
    paths = diagnostic_output_paths(out_dir)
    paths["point_errors"].parent.mkdir(parents=True, exist_ok=True)
    rows = final_fit.meta.copy()
    rows["pred_camera_world_x"] = final_fit.predicted_points[:, 0]
    rows["pred_camera_world_y"] = final_fit.predicted_points[:, 1]
    rows["pred_camera_world_z"] = final_fit.predicted_points[:, 2]
    rows["error_m"] = final_fit.errors_m
    rows["error_mm"] = final_fit.errors_m * 1000.0
    rows.to_csv(paths["point_errors"], index=False)

    frame_rows = []
    for sync_index, group in rows.groupby("sync_index", sort=True):
        summary = summarize_errors(group["error_m"].to_numpy(dtype=np.float64))
        frame_rows.append(
            {
                "sync_index": int(sync_index),
                "frame_time": float(group["frame_time"].iloc[0]),
                "aligned_opti_time": float(group["aligned_opti_time"].iloc[0]),
                "n_points": int(summary["n"]),
                "mean_error_mm": summary["mean_m"] * 1000.0,
                "rmse_error_mm": summary["rmse_m"] * 1000.0,
                "median_error_mm": summary["median_m"] * 1000.0,
                "p95_error_mm": summary["p95_m"] * 1000.0,
                "max_error_mm": summary["max_m"] * 1000.0,
                "world_grid_rmse_mm": float(group["world_grid_rmse_mm"].iloc[0]),
            }
        )
    pd.DataFrame(frame_rows).sort_values("frame_time").to_csv(paths["frame_errors"], index=False)
    return summarize_errors(final_fit.errors_m)


def write_alignment_toml(
    out_dir: Path,
    chosen: OffsetFitResult,
    final_fit: TransformFit,
) -> dict[str, Any]:
    inverse_scale, inverse_rotation, inverse_translation = inverse_similarity(final_fit.scale, final_fit.rotation, final_fit.translation)
    document: dict[str, Any] = {
        "schema_version": 1,
        "alignment_model": "optitrack_to_camera_world_with_12d_board_marker_offsets",
        "time_offset_seconds": float(chosen.offset_s),
        "optitrack_to_camera_world": {
            "scale": float(final_fit.scale),
            "rotation_matrix": final_fit.rotation.tolist(),
            "translation_m": final_fit.translation.tolist(),
        },
        "camera_world_to_optitrack": {
            "scale": float(inverse_scale),
            "rotation_matrix": inverse_rotation.tolist(),
            "translation_m": inverse_translation.tolist(),
        },
        "calibration_board_marker_correction": {
            "corner_names": ["p00", "p10", "p11", "p01"],
            "raw_marker_indices_for_corners": list(chosen.permutation),
            "corner_local_offsets_m": chosen.offsets_4x3_m.tolist(),
        },
        "calibration_board": {
            "square_size_m": SQUARE_SIZE_M,
            "board_squares_cols_rows": [BOARD_SQUARE_COLS, BOARD_SQUARE_ROWS],
            "board_size_m": [BOARD_WIDTH_M, BOARD_HEIGHT_M],
            "inner_grid_cols_rows": [INNER_GRID_COLS, INNER_GRID_ROWS],
            "expected_point_ids": list(EXPECTED_POINT_IDS),
        },
    }
    (out_dir / ALIGNMENT_TOML_NAME).write_text(rtoml.dumps(document), encoding="utf-8")
    return document


def write_markdown_report(
    out_dir: Path,
    alignment: dict[str, Any],
    comparison_df: pd.DataFrame,
    chosen: OffsetFitResult,
    equal_height_seed: Candidate,
    world_quality: dict[str, float | int],
    error_summary: dict[str, float | int],
) -> None:
    summary = error_summary
    rmse_mm = float(summary["rmse_m"]) * 1000.0
    status = "PASS" if rmse_mm <= TARGET_RMSE_MM else "CHECK"
    offsets_mm = np.asarray(alignment["calibration_board_marker_correction"]["corner_local_offsets_m"], dtype=np.float64) * 1000.0
    offsets_df = pd.DataFrame(offsets_mm, columns=["dx_mm", "dy_mm", "dz_mm"])
    offsets_df.insert(0, "corner", ["p00", "p10", "p11", "p01"])
    paths = diagnostic_output_paths(out_dir)

    lines = [
        "# OptiTrack To Camera-World Alignment Report",
        "",
        "## Result",
        f"- Final all-frame refit RMSE: `{rmse_mm:.6f} mm` (`{status}`, reference target {TARGET_RMSE_MM:.1f} mm).",
        f"- Time sync: `opti_time = world_time + {alignment['time_offset_seconds']:.10f} s`.",
        f"- Selected `lambda_xy`: `{chosen.lambda_xy}`.",
        f"- Marker order `[p00,p10,p11,p01] -> raw marker indices`: `{alignment['calibration_board_marker_correction']['raw_marker_indices_for_corners']}`.",
        f"- OptiTrack -> camera-world scale: `{alignment['optitrack_to_camera_world']['scale']:.10f}`.",
        "",
        "## Output Files",
        f"- `{ALIGNMENT_TOML_NAME}`: clean reusable coordinate transform and board marker correction parameters.",
        f"- `{REPORT_MD_NAME}`: human-readable quality report, including lambda sweep and marker offset tables.",
        f"- `{paths['point_errors'].relative_to(out_dir)}`: per-observation board inner-corner 3D errors after alignment.",
        f"- `{paths['frame_errors'].relative_to(out_dir)}`: per-frame mean/RMSE/p95/max 3D errors after alignment.",
        "",
        "## Transform Convention",
        "```text",
        "opti_time_seconds = world_points_frame_time_seconds + time_offset_seconds",
        "X_camera_world = scale * (rotation_matrix @ X_optitrack) + translation_m",
        "```",
        "",
        "## World Points Used For Fit",
        f"- Total world_points frames: `{world_quality['frames_total']}`.",
        f"- Frames used for alignment fit: `{world_quality['frames_used_for_fit']}`.",
        f"- Fit-frame grid RMSE mean/median/p95: `{world_quality['fit_frame_grid_rmse_mean_mm']:.6f}` / `{world_quality['fit_frame_grid_rmse_median_mm']:.6f}` / `{world_quality['fit_frame_grid_rmse_p95_mm']:.6f}` mm.",
        "",
        "## Board Marker 12D Offsets",
        "These offsets only correct this calibration board's OptiTrack marker centers to board paper corners. Do not apply them to other rigid bodies or skeleton points.",
        "| corner | dx mm | dy mm | dz mm |",
        "|---|---:|---:|---:|",
    ]
    for _, row in offsets_df.iterrows():
        lines.append(f"| {row['corner']} | {row['dx_mm']:.6f} | {row['dy_mm']:.6f} | {row['dz_mm']:.6f} |")

    lines.extend(
        [
            "",
            "## Lambda Sweep",
            "| lambda_xy | train RMSE mm | test RMSE mm | all RMSE mm | xy RMS mm | z mean mm | z std mm |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in comparison_df.sort_values("lambda_xy").iterrows():
        lines.append(
            f"| {row['lambda_xy']:.6g} | {row['train_rmse_mm']:.6f} | {row['test_rmse_mm']:.6f} | "
            f"{row['all_rmse_mm']:.6f} | {row['xy_rms_mm']:.6f} | {row['z_mean_mm']:.6f} | {row['z_std_mm']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Final Error Summary",
            "| n points | mean mm | RMSE mm | median mm | p90 mm | p95 mm | max mm |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {summary['n']} | {summary['mean_m'] * 1000.0:.6f} | {summary['rmse_m'] * 1000.0:.6f} | "
                f"{summary['median_m'] * 1000.0:.6f} | {summary['p90_m'] * 1000.0:.6f} | "
                f"{summary['p95_m'] * 1000.0:.6f} | {summary['max_m'] * 1000.0:.6f} |"
            ),
            "",
            "## Equal-Height Seed",
            f"- Seed offset: `{equal_height_seed.offset_s:.10f} s`.",
            f"- Seed shared marker height: `{equal_height_seed.marker_height_m * 1000.0:.6f} mm`.",
            f"- Seed RMSE: `{equal_height_seed.rmse_m * 1000.0:.6f} mm`.",
        ]
    )
    lines.append("")
    (out_dir / REPORT_MD_NAME).write_text("\n".join(lines), encoding="utf-8")


def validate_config(config: OptitrackAlignmentConfig) -> None:
    if not config.world_points_csv.exists():
        raise FileNotFoundError(f"world_points CSV not found: {config.world_points_csv}")
    if not config.optitrack_csv.exists():
        raise FileNotFoundError(f"OptiTrack CSV not found: {config.optitrack_csv}")
    if config.offset_min >= config.offset_max:
        raise ValueError("offset_min must be < offset_max")
    if config.coarse_offset_step <= 0:
        raise ValueError("coarse_offset_step must be positive")
    if config.shared_height_min >= config.shared_height_max:
        raise ValueError("shared_height_min must be < shared_height_max")
    if config.coarse_height_count < 2:
        raise ValueError("coarse_height_count must be >= 2")
    if config.offset_12d_abs_max <= 0:
        raise ValueError("offset_12d_abs_max must be positive")
    if config.offset_refine_window <= 0:
        raise ValueError("offset_refine_window must be positive")
    if not (0 <= config.test_ratio < 1):
        raise ValueError("test_ratio must be in [0, 1)")
    if config.min_points_per_fit_frame < 3:
        raise ValueError("min_points_per_fit_frame must be >= 3")
    if config.min_overlap_frames < 3:
        raise ValueError("min_overlap_frames must be >= 3")
    if config.max_coarse_frames < 1:
        raise ValueError("max_coarse_frames must be >= 1")
    if config.top_candidates_to_refine < 1:
        raise ValueError("top_candidates_to_refine must be >= 1")
    parse_lambda_list(config.lambda_xy_list)
    if config.select_lambda not in {"min_test", "min_all"}:
        float(config.select_lambda)


def run_optitrack_alignment_12d(config: OptitrackAlignmentConfig) -> dict[str, Any]:
    """Run the full 12D alignment and write all output artifacts."""

    config = OptitrackAlignmentConfig(
        world_points_csv=config.world_points_csv.expanduser().resolve(),
        optitrack_csv=config.optitrack_csv.expanduser().resolve(),
        output_dir=config.output_dir.expanduser().resolve(),
        offset_min=config.offset_min,
        offset_max=config.offset_max,
        coarse_offset_step=config.coarse_offset_step,
        shared_height_min=config.shared_height_min,
        shared_height_max=config.shared_height_max,
        coarse_height_count=config.coarse_height_count,
        offset_12d_abs_max=config.offset_12d_abs_max,
        offset_refine_window=config.offset_refine_window,
        lambda_xy_list=config.lambda_xy_list,
        select_lambda=config.select_lambda,
        test_ratio=config.test_ratio,
        seed=config.seed,
        max_world_grid_rmse_m=config.max_world_grid_rmse_m,
        min_points_per_fit_frame=config.min_points_per_fit_frame,
        min_overlap_frames=config.min_overlap_frames,
        max_coarse_frames=config.max_coarse_frames,
        top_candidates_to_refine=config.top_candidates_to_refine,
        equal_height_maxiter=config.equal_height_maxiter,
        offset_12d_maxiter=config.offset_12d_maxiter,
        allow_global_scale=config.allow_global_scale,
    )
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading world_points: %s", config.world_points_csv)
    world_frames = load_world_frames(config.world_points_csv)
    fit_frames = select_fit_frames(
        world_frames,
        config.max_world_grid_rmse_m,
        config.min_points_per_fit_frame,
        config.min_overlap_frames,
    )
    world_quality = summarize_world_quality(world_frames, fit_frames)
    logger.info(
        "world frames=%d fit_frames=%d mean_grid_rmse=%.3fmm",
        len(world_frames),
        len(fit_frames),
        np.mean([frame.grid_rmse_m for frame in fit_frames]) * 1000.0,
    )

    logger.info("Loading OptiTrack CSV: %s", config.optitrack_csv)
    opti, marker_columns = load_optitrack_markers(config.optitrack_csv)
    logger.info("OptiTrack rows=%d time=[%.3f, %.3f] marker_source=%s", len(opti), opti.time.min(), opti.time.max(), marker_columns["selected_marker_type"])

    equal_seed = optimize_equal_height_seed(fit_frames, opti, config)

    train_records, test_records = train_test_split_records(fit_frames, config.test_ratio, config.seed)
    logger.info("split: train_frames=%d test_frames=%d", len(train_records), len(test_records))
    lambda_values = parse_lambda_list(config.lambda_xy_list)
    fit_results = optimize_12d_offsets(train_records, test_records, fit_frames, opti, equal_seed, lambda_values, config)
    comparison_df = result_frame(fit_results)

    chosen = select_result(fit_results, config.select_lambda)
    logger.info(
        "chosen lambda=%g: offset=%.9fs all_rmse=%.6fmm xy_rms=%.6fmm",
        chosen.lambda_xy,
        chosen.offset_s,
        chosen.all_rmse_m * 1000.0,
        chosen.xy_rms_m * 1000.0,
    )
    final_fit = fit_transform_for_params(
        fit_frames,
        opti,
        chosen.permutation,
        chosen.offset_s,
        chosen.offsets_4x3_m,
        allow_scale=config.allow_global_scale,
    )
    if final_fit is None:
        raise RuntimeError("Final transform fitting failed")
    error_summary = write_error_tables(config.output_dir, final_fit)
    alignment = write_alignment_toml(
        config.output_dir,
        chosen,
        final_fit,
    )
    write_markdown_report(config.output_dir, alignment, comparison_df, chosen, equal_seed, world_quality, error_summary)

    logger.info(
        "FINAL rmse=%.6fmm mean=%.6fmm offset=%.9fs scale=%.9f outputs=%s",
        error_summary["rmse_m"] * 1000.0,
        error_summary["mean_m"] * 1000.0,
        chosen.offset_s,
        final_fit.scale,
        config.output_dir,
    )
    return {
        "alignment": alignment,
        "error_summary": error_summary,
        "chosen_lambda_xy": chosen.lambda_xy,
        "outputs": {
            "alignment_toml": str(config.output_dir / ALIGNMENT_TOML_NAME),
            "report_md": str(config.output_dir / REPORT_MD_NAME),
            "point_errors_csv": str(diagnostic_output_paths(config.output_dir)["point_errors"]),
            "frame_errors_csv": str(diagnostic_output_paths(config.output_dir)["frame_errors"]),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="12D alignment between Caliscope world_points.csv and OptiTrack/Motive CSV.")
    parser.add_argument("--world-points", type=Path, required=True, help="Caliscope capture_volume/world_points.csv")
    parser.add_argument("--optitrack-csv", type=Path, required=True, help="Motive/OptiTrack CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--offset-min", type=float, default=-20.0)
    parser.add_argument("--offset-max", type=float, default=20.0)
    parser.add_argument("--coarse-offset-step", type=float, default=0.25)
    parser.add_argument("--shared-height-min", type=float, default=-0.03)
    parser.add_argument("--shared-height-max", type=float, default=0.03)
    parser.add_argument("--coarse-height-count", type=int, default=7)
    parser.add_argument("--offset-12d-abs-max", type=float, default=0.03)
    parser.add_argument("--offset-refine-window", type=float, default=1.0)
    parser.add_argument("--lambda-xy-list", type=str, default="0,0.1,0.2,0.5,1,10,100")
    parser.add_argument("--select-lambda", type=str, default="0.2", help="Numeric lambda, min_test, or min_all")
    parser.add_argument("--test-ratio", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-world-grid-rmse-m", type=float, default=0.005)
    parser.add_argument("--min-points-per-fit-frame", type=int, default=15)
    parser.add_argument("--min-overlap-frames", type=int, default=40)
    parser.add_argument("--max-coarse-frames", type=int, default=120)
    parser.add_argument("--top-candidates-to-refine", type=int, default=8)
    parser.add_argument("--equal-height-maxiter", type=int, default=260)
    parser.add_argument("--offset-12d-maxiter", type=int, default=500)
    parser.add_argument("--allow-global-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    run_optitrack_alignment_12d(
        OptitrackAlignmentConfig(
            world_points_csv=args.world_points,
            optitrack_csv=args.optitrack_csv,
            output_dir=args.output_dir,
            offset_min=args.offset_min,
            offset_max=args.offset_max,
            coarse_offset_step=args.coarse_offset_step,
            shared_height_min=args.shared_height_min,
            shared_height_max=args.shared_height_max,
            coarse_height_count=args.coarse_height_count,
            offset_12d_abs_max=args.offset_12d_abs_max,
            offset_refine_window=args.offset_refine_window,
            lambda_xy_list=args.lambda_xy_list,
            select_lambda=args.select_lambda,
            test_ratio=args.test_ratio,
            seed=args.seed,
            max_world_grid_rmse_m=args.max_world_grid_rmse_m,
            min_points_per_fit_frame=args.min_points_per_fit_frame,
            min_overlap_frames=args.min_overlap_frames,
            max_coarse_frames=args.max_coarse_frames,
            top_candidates_to_refine=args.top_candidates_to_refine,
            equal_height_maxiter=args.equal_height_maxiter,
            offset_12d_maxiter=args.offset_12d_maxiter,
            allow_global_scale=args.allow_global_scale,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
