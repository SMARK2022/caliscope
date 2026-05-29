"""12D OptiTrack/world_points alignment for non-GUI calibration workflows."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
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
    write_plots: bool = True


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
    stage: str


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


def optimize_equal_height_seed(records: list[WorldFrame], opti: pd.DataFrame, config: OptitrackAlignmentConfig) -> tuple[Candidate, list[Candidate]]:
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
                    "coarse_equal_height",
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
        refined_results.append(
            Candidate(seed.permutation, offset_s, marker_height_m, rmse_m, n_frames, n_points, scale, "refined_equal_height")
        )

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
    return best, [*coarse_results, *refined_results]


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


def candidate_frame(candidates: Iterable[Candidate]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "stage": candidate.stage,
                "permutation_raw_marker_indices_for_p00_p10_p11_p01": str(candidate.permutation),
                "offset_s": candidate.offset_s,
                "marker_height_m": candidate.marker_height_m,
                "marker_height_mm": candidate.marker_height_m * 1000.0,
                "rmse_m": candidate.rmse_m,
                "rmse_mm": candidate.rmse_m * 1000.0,
                "n_frames": candidate.n_frames,
                "n_points": candidate.n_points,
                "scale": candidate.scale,
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse_m", "stage"], kind="stable")


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


def offsets_frame(result: OffsetFitResult) -> pd.DataFrame:
    rows = []
    for i, offset in enumerate(result.offsets_4x3_m):
        rows.append(
            {
                "corner_id": i,
                "dx_m": offset[0],
                "dy_m": offset[1],
                "dz_m": offset[2],
                "dx_mm": offset[0] * 1000.0,
                "dy_mm": offset[1] * 1000.0,
                "dz_mm": offset[2] * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def write_world_quality(out_dir: Path, frames: list[WorldFrame], fit_frames: list[WorldFrame]) -> None:
    fit_sync_indexes = {frame.sync_index for frame in fit_frames}
    rows = [
        {
            "sync_index": frame.sync_index,
            "frame_time": frame.frame_time,
            "n_points": int(len(frame.point_ids)),
            "world_grid_fit_rmse_m": frame.grid_rmse_m,
            "world_grid_fit_rmse_mm": frame.grid_rmse_m * 1000.0,
            "used_for_fit": int(frame.sync_index in fit_sync_indexes),
        }
        for frame in frames
    ]
    pd.DataFrame(rows).sort_values("frame_time").to_csv(out_dir / "world_points_frame_quality.csv", index=False)


def write_error_tables(out_dir: Path, final_fit: TransformFit) -> dict[str, float | int]:
    rows = final_fit.meta.copy()
    rows["pred_camera_world_x"] = final_fit.predicted_points[:, 0]
    rows["pred_camera_world_y"] = final_fit.predicted_points[:, 1]
    rows["pred_camera_world_z"] = final_fit.predicted_points[:, 2]
    rows["error_m"] = final_fit.errors_m
    rows["error_mm"] = final_fit.errors_m * 1000.0
    rows.to_csv(out_dir / "aligned_point_errors.csv", index=False)

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
    pd.DataFrame(frame_rows).sort_values("frame_time").to_csv(out_dir / "alignment_frame_error_summary.csv", index=False)

    point_rows = []
    for point_id, group in rows.groupby("point_id", sort=True):
        summary = summarize_errors(group["error_m"].to_numpy(dtype=np.float64))
        point_rows.append(
            {
                "point_id": int(point_id),
                "n": int(summary["n"]),
                "mean_error_mm": summary["mean_m"] * 1000.0,
                "rmse_error_mm": summary["rmse_m"] * 1000.0,
                "median_error_mm": summary["median_m"] * 1000.0,
                "p95_error_mm": summary["p95_m"] * 1000.0,
                "max_error_mm": summary["max_m"] * 1000.0,
            }
        )
    pd.DataFrame(point_rows).to_csv(out_dir / "alignment_point_id_error_summary.csv", index=False)
    return summarize_errors(final_fit.errors_m)


def write_plots(out_dir: Path, final_fit: TransformFit, comparison_df: pd.DataFrame) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        logger.info("matplotlib unavailable, skipping plots: %s", exc)
        return []

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    errors_mm = final_fit.errors_m * 1000.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(errors_mm, bins=70, alpha=0.85)
    ax.axvline(float(np.sqrt(np.mean(errors_mm**2))), linestyle="--", label="RMSE")
    ax.set_xlabel("3D error (mm)")
    ax.set_ylabel("count")
    ax.set_title("Point-wise 12D alignment error")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "point_error_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    frame_df = pd.read_csv(out_dir / "alignment_frame_error_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frame_df["frame_time"], frame_df["rmse_error_mm"], linewidth=1.1)
    ax.axhline(TARGET_RMSE_MM, linestyle="--", linewidth=1.0, label=f"{TARGET_RMSE_MM:.1f} mm")
    ax.set_xlabel("world_points frame time (s)")
    ax.set_ylabel("frame RMSE (mm)")
    ax.set_title("Frame-wise 12D alignment RMSE")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "frame_rmse_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(comparison_df["lambda_xy"], comparison_df["train_rmse_mm"], marker="o", label="train")
    if comparison_df["test_rmse_mm"].notna().any():
        ax.plot(comparison_df["lambda_xy"], comparison_df["test_rmse_mm"], marker="o", label="test")
    ax.plot(comparison_df["lambda_xy"], comparison_df["all_rmse_mm"], marker="o", label="all")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("lambda_xy")
    ax.set_ylabel("RMSE (mm)")
    ax.set_title("12D offset regularization sweep")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "lambda_sweep_rmse.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_transform_only_json(out_dir: Path, transform: dict[str, Any]) -> None:
    keys = [
        "schema_version",
        "model_type",
        "units",
        "convention",
        "time_offset_seconds",
        "scale_opti_to_camera_world",
        "R_opti_to_camera_world",
        "t_opti_to_camera_world_m",
        "inverse_scale_camera_world_to_opti",
        "R_camera_world_to_opti",
        "t_camera_world_to_opti_m",
        "fit_error_all_refit",
    ]
    slim = {key: transform[key] for key in keys if key in transform}
    (out_dir / "transform_only.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")


def write_transform_json(
    out_dir: Path,
    world_points_csv: Path,
    optitrack_csv: Path,
    marker_columns: dict[str, str],
    chosen: OffsetFitResult,
    final_fit: TransformFit,
    error_summary: dict[str, float | int],
    equal_height_seed: Candidate,
    train_count: int,
    test_count: int,
) -> dict[str, Any]:
    inverse_scale, inverse_rotation, inverse_translation = inverse_similarity(final_fit.scale, final_fit.rotation, final_fit.translation)
    transform: dict[str, Any] = {
        "schema_version": 2,
        "model_type": "optitrack_world_points_12d_marker_offsets_sim3",
        "units": {"length": "meter", "time": "second", "angle": "radian-free direction cosine matrix"},
        "source_world_points_csv": str(world_points_csv),
        "source_optitrack_csv": str(optitrack_csv),
        "optitrack_marker_columns": marker_columns,
        "convention": {
            "time_alignment": "opti_time_seconds = world_points_frame_time_seconds + time_offset_seconds",
            "marker_permutation": "permutation maps [p00,p10,p11,p01] to raw marker indices [Marker001..Marker004]",
            "local_12d_offsets": "corner_i = raw_marker_i + dx_i*e_x + dy_i*e_y + dz_i*e_z",
            "optitrack_to_camera_world": "X_camera_world = scale * R_opti_to_camera_world @ X_opti_12d_corrected + t",
            "final_transform_scope": "scale/R/t are fitted after applying 12D local marker offsets to board markers",
        },
        "board_model": {
            "square_size_m": SQUARE_SIZE_M,
            "board_squares_cols_rows": [BOARD_SQUARE_COLS, BOARD_SQUARE_ROWS],
            "board_size_m": [BOARD_WIDTH_M, BOARD_HEIGHT_M],
            "inner_grid_cols_rows": [INNER_GRID_COLS, INNER_GRID_ROWS],
            "expected_point_ids": list(EXPECTED_POINT_IDS),
        },
        "data_split": {"train_frames": int(train_count), "test_frames": int(test_count)},
        "equal_height_seed": {
            "offset_s": equal_height_seed.offset_s,
            "marker_height_m": equal_height_seed.marker_height_m,
            "marker_height_mm": equal_height_seed.marker_height_m * 1000.0,
            "rmse_m": equal_height_seed.rmse_m,
            "rmse_mm": equal_height_seed.rmse_m * 1000.0,
            "scale": equal_height_seed.scale,
        },
        "chosen_lambda_xy": chosen.lambda_xy,
        "time_offset_seconds": chosen.offset_s,
        "marker_permutation_raw_indices_for_p00_p10_p11_p01": list(chosen.permutation),
        "marker_corner_local_offsets_m": chosen.offsets_4x3_m.tolist(),
        "marker_corner_local_offsets_mm": (chosen.offsets_4x3_m * 1000.0).tolist(),
        "marker_offset_summary": {
            "xy_rms_m": chosen.xy_rms_m,
            "xy_rms_mm": chosen.xy_rms_m * 1000.0,
            "z_mean_m": chosen.z_mean_m,
            "z_mean_mm": chosen.z_mean_m * 1000.0,
            "z_std_m": chosen.z_std_m,
            "z_std_mm": chosen.z_std_m * 1000.0,
        },
        "scale_opti_to_camera_world": final_fit.scale,
        "R_opti_to_camera_world": final_fit.rotation.tolist(),
        "t_opti_to_camera_world_m": final_fit.translation.tolist(),
        "inverse_scale_camera_world_to_opti": inverse_scale,
        "R_camera_world_to_opti": inverse_rotation.tolist(),
        "t_camera_world_to_opti_m": inverse_translation.tolist(),
        "rotation_determinant": float(np.linalg.det(final_fit.rotation)),
        "orthogonality_frobenius_error": float(np.linalg.norm(final_fit.rotation.T @ final_fit.rotation - np.eye(3))),
        "fit_error_all_refit": error_summary,
    }
    text = json.dumps(transform, indent=2)
    (out_dir / "alignment_transform_12d_summary.json").write_text(text, encoding="utf-8")
    (out_dir / "alignment_transform_summary.json").write_text(text, encoding="utf-8")
    write_transform_only_json(out_dir, transform)
    return transform


def write_markdown_report(out_dir: Path, transform: dict[str, Any], comparison_df: pd.DataFrame, plot_paths: list[str]) -> None:
    summary = transform["fit_error_all_refit"]
    rmse_mm = float(summary["rmse_m"]) * 1000.0
    status = "PASS" if rmse_mm <= TARGET_RMSE_MM else "CHECK"
    offsets_df = pd.DataFrame(transform["marker_corner_local_offsets_mm"], columns=["dx_mm", "dy_mm", "dz_mm"])
    offsets_df.insert(0, "corner", ["p00", "p10", "p11", "p01"])

    lines = [
        "# 12D Marker Offset Alignment Report",
        "",
        "## Result",
        f"- Final all-frame refit RMSE: `{rmse_mm:.6f} mm` (`{status}`, reference target {TARGET_RMSE_MM:.1f} mm).",
        f"- Time sync: `opti_time = world_time + {transform['time_offset_seconds']:.10f} s`.",
        f"- Selected `lambda_xy`: `{transform['chosen_lambda_xy']}`.",
        f"- Marker permutation `[p00,p10,p11,p01] -> raw [Marker001..004]`: `{transform['marker_permutation_raw_indices_for_p00_p10_p11_p01']}`.",
        f"- OptiTrack -> camera-world scale: `{transform['scale_opti_to_camera_world']:.10f}`.",
        "",
        "## 12D Offset Parameters",
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
            "## Output Files",
            "- `alignment_transform_12d_summary.json`: full transform, 12D offsets, diagnostics, and source metadata.",
            "- `alignment_transform_summary.json`: same content under the legacy downstream filename.",
            "- `transform_only.json`: minimal downstream transform payload.",
            "- `fit_comparison_summary.csv`: lambda sweep train/test/all errors.",
            "- `marker_corner_offsets.csv`: four-corner 12D offsets.",
            "- `aligned_point_errors.csv`: point-wise 3D errors.",
            "- `alignment_frame_error_summary.csv`: frame-wise RMSE.",
            "- `alignment_point_id_error_summary.csv`: point-id-wise errors.",
            "- `equal_height_candidate_results.csv`: equal-height seed candidates.",
            "- `world_points_frame_quality.csv`: 5x3 grid quality of each world_points frame.",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", *[f"- `{Path(path).relative_to(out_dir)}`" for path in plot_paths]])
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
        write_plots=config.write_plots,
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
    write_world_quality(config.output_dir, world_frames, fit_frames)
    logger.info(
        "world frames=%d fit_frames=%d mean_grid_rmse=%.3fmm",
        len(world_frames),
        len(fit_frames),
        np.mean([frame.grid_rmse_m for frame in fit_frames]) * 1000.0,
    )

    logger.info("Loading OptiTrack CSV: %s", config.optitrack_csv)
    opti, marker_columns = load_optitrack_markers(config.optitrack_csv)
    logger.info("OptiTrack rows=%d time=[%.3f, %.3f] marker_source=%s", len(opti), opti.time.min(), opti.time.max(), marker_columns["selected_marker_type"])

    equal_seed, equal_candidates = optimize_equal_height_seed(fit_frames, opti, config)
    candidate_frame(equal_candidates).to_csv(config.output_dir / "equal_height_candidate_results.csv", index=False)

    train_records, test_records = train_test_split_records(fit_frames, config.test_ratio, config.seed)
    logger.info("split: train_frames=%d test_frames=%d", len(train_records), len(test_records))
    lambda_values = parse_lambda_list(config.lambda_xy_list)
    fit_results = optimize_12d_offsets(train_records, test_records, fit_frames, opti, equal_seed, lambda_values, config)
    comparison_df = result_frame(fit_results)
    comparison_df.to_csv(config.output_dir / "fit_comparison_summary.csv", index=False)

    chosen = select_result(fit_results, config.select_lambda)
    logger.info(
        "chosen lambda=%g: offset=%.9fs all_rmse=%.6fmm xy_rms=%.6fmm",
        chosen.lambda_xy,
        chosen.offset_s,
        chosen.all_rmse_m * 1000.0,
        chosen.xy_rms_m * 1000.0,
    )
    offsets_frame(chosen).to_csv(config.output_dir / "marker_corner_offsets.csv", index=False)

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
    transform = write_transform_json(
        config.output_dir,
        config.world_points_csv,
        config.optitrack_csv,
        marker_columns,
        chosen,
        final_fit,
        error_summary,
        equal_seed,
        len(train_records),
        len(test_records),
    )
    plot_paths = write_plots(config.output_dir, final_fit, comparison_df) if config.write_plots else []
    write_markdown_report(config.output_dir, transform, comparison_df, plot_paths)

    logger.info(
        "FINAL rmse=%.6fmm mean=%.6fmm offset=%.9fs scale=%.9f outputs=%s",
        error_summary["rmse_m"] * 1000.0,
        error_summary["mean_m"] * 1000.0,
        chosen.offset_s,
        final_fit.scale,
        config.output_dir,
    )
    return transform


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
    parser.add_argument("--write-plots", action=argparse.BooleanOptionalAction, default=True)
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
            write_plots=args.write_plots,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
