"""Camera-frame preparation and physically labeled calibration checks."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .assays import number, partition, tick
from .geometry import apply_homography, estimate_homography, panel_coordinates, scaled_dimensions
from .inspection import Report, Scene, make_plane, paired_plane
from .model import Point, Problem, Region, boolean, numeric, sequence
from .raster import code_statistics, heatmap, luma, masked_rgb, overlay_regions, png_bytes, rounding, side_by_side


def compare_alignment(scene: Scene, second: Scene, region: Region, options: dict) -> Report:
    opacity = numeric(options.get("mix"), "叠加图对照透明百分比", 1, 99, default=50) / 100
    first, target, overlap = paired_plane(scene, second, region, max_side=880)
    grid_size = min(7, max(4, min(first.shape) // 85))
    first_gray = first.gray
    target_gray = target.gray
    shift = float(np.median(target_gray[overlap] - first_gray[overlap]))
    delta = target_gray - first_gray - shift
    overlay = np.where(
        overlap[..., None],
        (1.0 - opacity) * first.rgb + opacity * target.rgb,
        np.where(first.observed[..., None], .60 * first.rgb, .60 * target.rgb),
    )
    row_records = []
    for cell in partition(first.shape[1], first.shape[0], grid_size, grid_size):
        keep = cell.values(overlap)
        if keep.mean() > .35:
            difference = cell.values(delta)[keep]
            score = rounding(np.median(np.abs(difference)))
        else:
            score = None
        row_records.append({
            "row": cell.row + 1,
            "column": cell.column + 1,
            "common_coverage_pct": rounding(100 * keep.mean(), 2),
            "median_abs_code_delta": score,
        })
    limit = max(.02, float(np.percentile(np.abs(delta[overlap]), 95)))
    metrics = {
        **first.info(),
        "pair_overlap_pct": rounding(100 * overlap.mean(), 3),
        "overlay_ratio": rounding(opacity),
        "estimated_global_code_shift": rounding(shift),
        "mean_absolute_code_difference": rounding(np.mean(np.abs(delta[overlap]))),
        "delta_range": [-rounding(limit), rounding(limit)],
    }
    report = Report("校准重叠检查", "在同一面板网格叠加两次采集，并记录两图确实同时可见的区域。", metrics, row_records)
    report.image("alignment-overlap.png", masked_rgb(overlay, first.observed | target.observed))
    report.image("alignment-residual.png", heatmap(delta, overlap, -limit, limit, diverging=True))
    report.table("alignment-cells.csv", row_records)
    report.warnings.append("差值含照明和拍摄设置影响，用于定位需复核区域。")
    return report


def validate_reference_geometry(source: Scene, dark: Scene, flat: Scene) -> None:
    for ref, role_name in ((dark, "镜头暗场"), (flat, "镜头平场")):
        if ref.panel["id"] != source.panel["id"]:
            raise Problem(f"{role_name}需来自同一面板记录。")
        if ref.image.size != source.image.size:
            raise Problem(f"{role_name}和采集图的朝向规范化后尺寸须一致。")
        if ref.capture["id"] == source.capture["id"]:
            raise Problem("不能以被处理图作为暗场或平场基准。")
        source_camera = source.capture["exposure"].get("camera")
        reference_camera = ref.capture["exposure"].get("camera")
        if source_camera and reference_camera and source_camera != reference_camera:
            raise Problem(f"{role_name}的相机标识和被处理图不同。")
    dark.require_role({"dark"}, "暗场基准必须标记为镜头暗场。")
    flat.require_role({"flat"}, "平场基准必须标记为镜头平场。")
    keys = ("shutter_ms", "iso")
    for key in keys:
        first_value = source.capture["exposure"].get(key)
        if first_value is None:
            continue
        for ref in (dark, flat):
            second_value = ref.capture["exposure"].get(key)
            if second_value is not None and abs(first_value - second_value) > .02 * first_value:
                raise Problem("暗平场与被处理图应使用一致的快门和 ISO。")


@dataclass
class Correction:
    rgb: np.ndarray
    valid: np.ndarray
    normalization: np.ndarray
    clipped: np.ndarray
    reference_bad: np.ndarray
    original_usable: np.ndarray

    def camera_rgba(self) -> bytes:
        values = np.rint(np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)
        values[~self.valid] = 0
        opacity = np.where(self.valid, 255, 0).astype(np.uint8)
        return png_bytes(np.dstack((values, opacity)))


def perform_flatfield(source: Scene, dark: Scene, flat: Scene, minimum: float) -> Correction:
    validate_reference_geometry(source, dark, flat)
    good = dark.image.valid & flat.image.valid
    reference_input = flat.image.rgb - dark.image.rgb
    high = (flat.image.rgb < .995).all(axis=-1)
    good &= high & (reference_input >= minimum).all(axis=-1)
    affected = source.image.valid & good
    if good.mean() < .25 or affected.sum() < 125:
        raise Problem("暗平场参考的有效支撑不足，检查亮度或最低平场差值。")
    median = np.median(reference_input[good], axis=0)
    raw_delta = source.image.rgb - dark.image.rgb
    estimate = np.zeros_like(raw_delta)
    np.divide(raw_delta * median, reference_input, out=estimate, where=affected[..., None])
    clipped = (np.min(estimate, axis=-1) < 0) | (np.max(estimate, axis=-1) > 1)
    clipped &= affected
    return Correction(np.clip(estimate, 0, 1), affected, median, clipped, ~good, source.image.valid)


def flatfield_report(scene: Scene, dark: Scene, flat: Scene, minimum: float) -> tuple[Correction, Report]:
    computed = perform_flatfield(scene, dark, flat, minimum)
    before = luma(scene.image.rgb)
    after = luma(computed.rgb)
    supported = computed.valid
    error = after - before
    lo = max(.015, float(np.percentile(np.abs(error[supported]), 98)))
    coverage = supported.sum() / max(1, computed.original_usable.sum())
    metrics = {
        "dark_capture_id": dark.capture["id"],
        "flat_capture_id": flat.capture["id"],
        "dark_title": dark.capture["title"],
        "flat_title": flat.capture["title"],
        "minimum_flat_minus_dark": rounding(minimum),
        "normalization_reference": [rounding(value) for value in computed.normalization],
        "source_covered_pct": rounding(coverage * 100, 3),
        "clipped_pct": rounding(computed.clipped.sum() / supported.sum() * 100, 3),
        "before": code_statistics(before[supported]),
        "after": code_statistics(after[supported]),
    }
    report = Report("镜头背景校正", "在未经展平的相机原像素网格上应用暗场和相机平场。", metrics)
    report.image("camera-flatfield.png", masked_rgb(computed.rgb, computed.valid))
    report.image("camera-adjustment.png", heatmap(error, supported, -lo, lo, diverging=True))
    report.warnings.append("仅当暗平场使用相同现场设置时才可归因到镜头背景。无效参考位置不填补。")
    if coverage < .98:
        report.warnings.append("一部分原图像素没有可靠暗平场参考，派生 PNG 以透明标记。")
    return computed, report


def fiducial_pairs(raw, panel: dict):
    entries = sequence(raw, "检查点", minimum=4, maximum=28)
    target = []
    measured = []
    names = []
    for index, item in enumerate(entries, start=1):
        if not isinstance(item, dict):
            raise Problem(f"检查点 {index} 缺少两组坐标。")
        pair_id = item.get("label") or str(index)
        if not isinstance(pair_id, str) or len(pair_id) > 50:
            raise Problem("检查点名称过长。")
        target.append(Point.parse(item.get("panel")).values())
        measured.append(Point.parse(item.get("camera"), lo=-.10, hi=1.10).values())
        names.append(pair_id)
    if len(set(names)) != len(names):
        raise Problem("检查点名称不能重复。")
    panel_points = np.asarray(target)
    camera_points = np.asarray(measured)
    if len(np.unique(panel_points, axis=0)) != len(panel_points):
        raise Problem("存在重合的面板检查点。")
    return names, panel_points, camera_points


def check_fiducials(scene: Scene, second, region: Region, options: dict) -> Report:
    names, panel_points, observations = fiducial_pairs(options.get("pairs"), scene.panel)
    expected = scene.calibration.panel_to_camera(panel_points)
    initial = observations - expected
    source_scale = np.array(scene.image.size) - 1
    threshold = numeric(options.get("threshold"), "像素偏差提醒值", .25, 150, default=6)
    preview_fit = boolean(options.get("test_fit"), "同时看可拟合偏差", default=False)
    study_model = estimate_homography(panel_points, observations) if preview_fit else None
    trial = apply_homography(study_model, panel_points) if study_model is not None else None
    residual = initial * source_scale
    rows = []
    boxes = []
    tolerance = int(np.clip(round(max(scene.image.size) / 190), 2, 12))
    for i, name in enumerate(names):
        sx, sy = expected[i]
        px, py = observations[i]
        error = float(np.linalg.norm(residual[i]))
        trial_error = float(np.linalg.norm((observations[i] - trial[i]) * source_scale)) if trial is not None else None
        record = {
            "label": name,
            "panel_x": rounding(panel_points[i, 0]), "panel_y": rounding(panel_points[i, 1]),
            "observed_camera_x": rounding(px), "observed_camera_y": rounding(py),
            "expected_camera_x": rounding(sx), "expected_camera_y": rounding(sy),
            "source_error_x_px": rounding(residual[i, 0], 3),
            "source_error_y_px": rounding(residual[i, 1], 3),
            "source_distance_px": rounding(error, 3),
            "in_sample_fit_error_px": rounding(trial_error, 3) if trial_error is not None else None,
        }
        rows.append(record)
        right, bottom = tolerance / source_scale
        boxes.append(((sx - right, sy - bottom, sx + right, sy + bottom), f"R{i + 1}",
                      (249, 169, 102) if error > threshold else (167, 232, 201)))
        boxes.append(((px - right, py - bottom, px + right, py + bottom), f"O{i + 1}", (105, 173, 251)))
    metrics = {
        "base_calibration": scene.revision_id,
        "point_count": len(names),
        "max_point_deviation_px": max(record["source_distance_px"] for record in rows),
        "rms_camera_deviation_px": rounding(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))), 4),
        "points_outside_reminder": sum(record["source_distance_px"] > threshold for record in rows),
    }
    report = Report("几何检查点校验", "以手工对应点检查当前四角校准的定位偏差。", metrics, rows)
    report.image("geometry-checkpoints.png", overlay_regions(masked_rgb(scene.image.rgb, scene.image.valid), boxes))
    report.table("geometry-checkpoints.csv", rows)
    if trial is not None:
        report.warnings.append("试算残差来自用于拟合的同一批点，只作匹配诊断，不修改当前标定。")
    return report


def configs():
    return {
        "alignment": {
            "title": "采集重叠检查",
            "group": "采集坐标",
            "description": "叠加同一面板的两张校准采集，并标记交集区域。",
            "comparison": "required",
            "run": compare_alignment,
            "options": [number("mix", "对照图混合 %", 50, 1, 99)],
        },
        "fiducials": {
            "title": "几何检查点校验",
            "group": "采集坐标",
            "description": "输入四组或更多面板与相机检查点，核现用校准的误差。",
            "run": check_fiducials,
            "options": [{"key": "pairs", "label": "检查点数组", "type": "json",
                         "default": [{"panel": [.15, .18], "camera": [.19, .20]},
                                     {"panel": [.77, .21], "camera": [.73, .24]},
                                     {"panel": [.81, .73], "camera": [.78, .69]},
                                     {"panel": [.19, .78], "camera": [.22, .76]}]},
                        number("threshold", "像素偏差提醒值", 6, .25, 150, .25),
                        tick("test_fit", "同时看拟合偏差")],
        },
    }
