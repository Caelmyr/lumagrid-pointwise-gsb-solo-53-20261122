"""Camera-frame preparation and physically labeled calibration checks."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

from .assays import number, partition, tick
from .geometry import (
    Calibration, apply_homography, estimate_homography, fit_internal_references, panel_coordinates,
    sampling_grid, scaled_dimensions,
)
from .inspection import Report, Scene, make_plane, paired_plane
from .model import Point, Problem, Region, boolean, numeric, sequence
from .raster import (
    code_statistics, heatmap, luma, masked_rgb, overlay_regions, png_bytes, rounding,
    side_by_side,
)


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


def correction_proposal(
    scene: Scene,
    names: list[str],
    panel_points: np.ndarray,
    observations: np.ndarray,
    source_scale: np.ndarray,
    options: dict,
) -> dict | None:
    """Robust internal-reference field on the current four-corner version."""
    if not boolean(options.get("internal_reference"), "内部参照修正", default=False):
        return None
    if scene.revision is None:
        raise Problem("内部参照修正需要先保存四角标定，再在其基础上试算。")
    if scene.uses_internal_reference:
        raise Problem("当前查看的标定版本已经使用内部参照；请切换到对应四角版本再试算。")
    base = Calibration.from_raw(scene.revision["corners"])
    threshold = numeric(options.get("threshold"), "像素偏差提醒值", .25, 150, default=6)
    reach = numeric(options.get("reach"), "内部参照影响范围", .18, .45, default=.30)
    fit = fit_internal_references(
        base, names, panel_points, observations, source_scale, threshold_px=threshold, reach=reach,
    )
    if fit.field is None:
        return {"fit": fit, "calibration": None, "excluded": True, "base": base}
    corrected = base.corrected(fit.field)
    return {
        "fit": fit,
        "calibration": corrected,
        "excluded": False,
        "base": base,
        "threshold_px": threshold,
        "reach": reach,
        "support_area_pct": corrected.reference_summary()["support_area_pct"],
        "minimum_sampling": corrected.minimum_jacobian(),
    }


def _support_overlay(photo: Image.Image, support: np.ndarray, color=(129, 240, 213)) -> Image.Image:
    width, height = photo.size
    mask = Image.fromarray(np.clip(support * 255, 0, 255).astype(np.uint8), "L")
    mask = mask.resize((width, height), Image.Resampling.BILINEAR)
    tint = Image.new("RGB", photo.size, color)
    return Image.composite(Image.blend(photo, tint, 0.30), photo, mask)


def correction_figure(scene: Scene, proposal: dict, tolerance: int) -> Image.Image:
    """Corner-only vs corner+reference; teal marks where references are used."""
    width, height = scene.image.size
    overlay = proposal["calibration"]
    grid = panel_coordinates(96, 96, Region(0, 0, 1, 1))
    support = overlay.support(grid)
    photo = masked_rgb(scene.image.rgb, scene.image.valid)
    panels = [photo, _support_overlay(photo, support)]
    for index, calibration in enumerate((proposal["base"], overlay)):
        canvas = panels[index]
        draw = ImageDraw.Draw(canvas)
        for i in range(len(proposal["_names"])):
            for x, y, color in (
                (*calibration.panel_to_camera(proposal["_panel_points"][i:i + 1])[0], (105, 173, 251)),
                (*proposal["_observations"][i], (249, 169, 102)),
            ):
                half = tolerance / np.array([width - 1, height - 1], dtype=np.float64)
                draw.ellipse(
                    (max(0, (x - half[0]) * width), max(0, (y - half[1]) * height),
                     min(width, (x + half[0]) * width), min(height, (y + half[1]) * height)),
                    outline=color, width=2,
                )
        for i, record in enumerate(proposal["fit"].records):
            if record["included"]:
                continue
            px, py = (proposal["_observations"][i] * [width - 1, height - 1])
            span = int(np.clip(round(max(scene.image.size) / 90), 6, 26))
            draw.line((px - span, py - span, px + span, py + span), fill=(246, 120, 120), width=2)
            draw.line((px - span, py + span, px + span, py - span), fill=(246, 120, 120), width=2)
        draw.text((8, 8), "原四角" if index == 0 else "四角+内部参照（青色为参照支撑区）",
                  fill=(236, 241, 249))
        panels[index] = canvas
    return side_by_side(panels[0], panels[1])


def correction_payload(proposal: dict) -> dict | None:
    """JSON-safe summary of a proposal for the inspection snapshot."""
    if proposal is None or proposal.get("excluded"):
        return None
    field = proposal["calibration"].field
    return {
        "field": field.describe(),
        "threshold_px": proposal.get("threshold_px"),
        "reach": proposal.get("reach"),
        "support_area_pct": proposal.get("support_area_pct"),
        "minimum_sampling": proposal.get("minimum_sampling"),
        "summary": proposal["calibration"].reference_summary(),
        "fit": proposal["fit"].records,
    }


def check_fiducials(scene: Scene, second, region: Region, options: dict) -> tuple[Report, dict | None]:
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
    proposal = correction_proposal(scene, names, panel_points, observations, source_scale, options)
    if proposal is not None:
        proposal["_panel_points"] = panel_points
        proposal["_names"] = names
        proposal["_observations"] = observations
    corrected = proposal["calibration"] if proposal and not proposal["excluded"] else None
    corrected_points = corrected.panel_to_camera(panel_points) if corrected is not None else None
    for i, name in enumerate(names):
        sx, sy = expected[i]
        px, py = observations[i]
        error = float(np.linalg.norm(residual[i]))
        trial_error = float(np.linalg.norm((observations[i] - trial[i]) * source_scale)) if trial is not None else None
        after_error = (float(np.linalg.norm((observations[i] - corrected_points[i]) * source_scale))
                       if corrected_points is not None else None)
        record = {
            "label": name,
            "panel_x": rounding(panel_points[i, 0]), "panel_y": rounding(panel_points[i, 1]),
            "observed_camera_x": rounding(px), "observed_camera_y": rounding(py),
            "expected_camera_x": rounding(sx), "expected_camera_y": rounding(sy),
            "source_error_x_px": rounding(residual[i, 0], 3),
            "source_error_y_px": rounding(residual[i, 1], 3),
            "source_distance_px": rounding(error, 3),
            "in_sample_fit_error_px": rounding(trial_error, 3) if trial_error is not None else None,
            "base_reference_distance_px": rounding(error, 3),
            "corrected_distance_px": rounding(after_error, 3) if after_error is not None else None,
        }
        rows.append(record)
        right, bottom = tolerance / source_scale
        boxes.append(((sx - right, sy - bottom, sx + right, sy + bottom), f"R{i + 1}",
                      (249, 169, 102) if error > threshold else (167, 232, 201)))
        boxes.append(((px - right, py - bottom, px + right, py + bottom), f"O{i + 1}", (105, 173, 251)))
    if proposal and not proposal["excluded"]:
        for i, fit_record in enumerate(proposal["fit"].records):
            rows[i]["reference_included"] = fit_record["included"]
            rows[i]["reference_reason"] = fit_record["reason"]
            rows[i]["leave_one_out_px"] = fit_record["leave_one_out_px"]
            rows[i]["local_correction_px"] = fit_record["correction_px"]
        for i, fit_record in enumerate(proposal["fit"].records):
            if fit_record["included"]:
                continue
            px, py = observations[i]
            right, bottom = tolerance / source_scale
            boxes.append(((px - right, py - bottom, px + right, py + bottom), f"X{i + 1}", (246, 120, 120)))
    metrics = {
        "base_calibration": scene.revision_id,
        "geometry_model": "corners+internal-reference" if corrected is not None else "corners",
        "point_count": len(names),
        "max_point_deviation_px": max(record["source_distance_px"] for record in rows),
        "rms_camera_deviation_px": rounding(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))), 4),
        "points_outside_reminder": sum(record["source_distance_px"] > threshold for record in rows),
    }
    if proposal and not proposal["excluded"]:
        fit = proposal["fit"]
        included = int(fit.included.sum())
        after = np.asarray([record["corrected_distance_px"] for record in rows])
        metrics.update({
            "reference_count": included,
            "reference_excluded_count": len(names) - included,
            "reference_correction_area_pct": proposal["support_area_pct"],
            "reference_min_sampling": (
                rounding(proposal["minimum_sampling"], 6)
                if proposal["minimum_sampling"] is not None else None
            ),
            "rms_corrected_deviation_px": rounding(np.sqrt(np.mean(after ** 2)), 4),
        })
    report = Report("几何检查点校验", "以手工对应点检查当前四角校准的定位偏差。", metrics, rows)
    report.image("geometry-checkpoints.png", overlay_regions(masked_rgb(scene.image.rgb, scene.image.valid), boxes))
    report.table("geometry-checkpoints.csv", rows)
    if trial is not None:
        report.warnings.append("试算残差来自用于拟合的同一批点，只作匹配诊断，不修改当前标定。")
    if proposal is not None:
        if proposal["excluded"]:
            report.warnings.append(proposal["fit"].warning or "内部参照不足，未生成修正。")
        else:
            for record in proposal["fit"].records:
                if not record["included"]:
                    report.warnings.append(
                        f"参照 {record['label']} 未参与修正：{record['reason'] or '与周边趋势不符'}。"
                    )
            report.warnings.append(
                "内部参照修正只覆盖参照点附近；其余区域仍沿用四角标定，"
                "预览不会修改现有版本，保存后才会成为独立可选版本。"
            )
            if proposal["support_area_pct"] < 45:
                report.warnings.append(
                    f"参照集中在画面中部，仅 {proposal['support_area_pct']:.0f}% 面板获得修正；四角维持原四角结果。"
                )
            if proposal["minimum_sampling"] is not None and proposal["minimum_sampling"] <= 1e-4:
                report.warnings.append("局部修正可能造成折叠，不能保存；请缩小影响范围或复核参照点。")
            report.image("geometry-correction-preview.png", correction_figure(scene, proposal, tolerance))
    payload = correction_payload(proposal) if proposal is not None else None
    if payload is not None:
        sampling = proposal["minimum_sampling"]
        payload["savable"] = sampling is None or sampling > 1e-4
    return report, payload


def preview_correction(scene: Scene, proposal: dict) -> tuple[Report, dict]:
    """Rasterize both geometry models for the operator to compare before saving."""
    size = scaled_dimensions(*scene.image.size, maximum=720, limit_area=620_000)
    grid = panel_coordinates(size[0], size[1], Region(0, 0, 1, 1))
    base = proposal["base"]
    overlay = proposal["calibration"]
    first_rgb, first_valid = _sample_with(scene, base, grid)
    second_rgb, second_valid = _sample_with(scene, overlay, grid)
    support = overlay.support(grid)
    ridge = side_by_side(
        masked_rgb(first_rgb, first_valid),
        masked_rgb(second_rgb, second_valid),
    )
    tinted = masked_rgb(
        np.where((support >= 0.05)[..., None], second_rgb * (0.72 + 0.28 * support[..., None]), second_rgb),
        second_valid,
    )
    diff = np.linalg.norm(second_rgb - first_rgb, axis=-1)
    scale = max(0.01, float(np.percentile(diff[second_valid & first_valid], 98))) if np.any(second_valid & first_valid) else 1.0
    metrics = {
        "geometry_model": "corners+internal-reference",
        "base_calibration_id": scene.base_revision,
        "reference_points": int(len(overlay.field.panel)),
        "reference_correction_area_pct": overlay.reference_summary()["support_area_pct"],
        "corner_only_area_pct": overlay.reference_summary()["corner_only_area_pct"],
        "reference_min_sampling": (rounding(proposal["minimum_sampling"], 6)
                                   if proposal["minimum_sampling"] is not None else None),
        "max_correction_pct": overlay.reference_summary()["max_correction_pct"],
        "changed_region_p98": rounding(scale, 5),
    }
    report = Report(
        "内部参照修正预览",
        "左侧保持原四角展平，右侧叠加内部参照修正；浅色区仍有内部参照支撑，深色区沿用四角。",
        metrics,
    )
    report.image("geometry-models.png", ridge)
    report.image("geometry-where-used.png", tinted)
    report.image("geometry-difference.png", heatmap(diff, second_valid & first_valid, 0, scale))
    report.warnings.append("右图亮度随内部参照支撑减弱而降低，提示该处已回到四角结果。")
    report.warnings.append("预览不写入任何版本；确认后保存为独立可选标定版本。")
    return report, {
        "base_calibration_id": scene.base_revision,
        "field": overlay.field.describe(),
        "threshold_px": proposal.get("threshold_px"),
        "reach": proposal.get("reach"),
        "summary": overlay.reference_summary(),
        "fit": proposal["fit"].records,
    }


def _sample_with(scene: Scene, calibration: Calibration, grid: np.ndarray):
    camera_points = calibration.panel_to_camera(grid)
    coordinates, inside = sampling_grid(scene.image.valid.shape, camera_points)
    planes = [
        map_coordinates(face, coordinates, order=1, mode="nearest")
        for face in scene.image.rgb.transpose(2, 0, 1)
    ]
    return np.stack(planes, axis=-1), inside


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
                        tick("test_fit", "同时看拟合偏差"),
                        tick("internal_reference", "内部参照修正（试算）"),
                        number("reach", "内部参照影响范围", .30, .18, .45, .01)],
        },
    }
