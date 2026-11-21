"""Spatial light quality assays on aligned camera captures."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter, label

from .graycard import (
    STATUS_BRACKETED, STATUS_EXTRAPOLATED, STATUS_NEAR, STATUS_UNREFERENCED,
    correction_metrics, evidence_image, parse_gray_boxes, patch_rows, evaluate as gray_evaluate,
)
from .inspection import Report, Scene, make_plane, paired_plane, stable_fraction
from .model import Problem, Region, boolean, integral, numeric
from .raster import (
    chart_png, code_statistics, heatmap, linear_code, luma, masked_rgb, overlay_regions,
    paired_bounds, png_bytes, resize_report, rounding, side_by_side, titled_strip,
)


@dataclass
class Cell:
    row: int
    column: int
    x: slice
    y: slice
    box: tuple[float, float, float, float]

    def values(self, array: np.ndarray) -> np.ndarray:
        return array[self.y, self.x]


def partition(width: int, height: int, columns: int, rows: int, guard=0.0):
    if width < 3 * columns or height < 3 * rows:
        raise Problem("区域采样尺寸太小，无法分出所选网格。")
    for row in range(rows):
        top = round(row * height / rows)
        bottom = round((row + 1) * height / rows)
        dy = min((bottom - top - 1) // 2, round((bottom - top) * guard))
        for column in range(columns):
            left = round(column * width / columns)
            right = round((column + 1) * width / columns)
            dx = min((right - left - 1) // 2, round((right - left) * guard))
            yield Cell(
                row,
                column,
                slice(left + dx, right - dx),
                slice(top + dy, bottom - dy),
                ((left + dx) / width, (top + dy) / height,
                 (right - dx) / width, (bottom - dy) / height),
            )


def coverage_stats(cell: Cell, valid: np.ndarray, signal: np.ndarray, minimum: float) -> dict:
    keep = cell.values(valid)
    seen = float(keep.mean())
    stats = code_statistics(cell.values(signal)[keep]) if seen >= minimum else code_statistics([])
    return {
        "row": cell.row + 1,
        "column": cell.column + 1,
        "coverage_pct": rounding(seen * 100, 1),
        **stats,
    }


def spatial_uniformity(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    size = integral(options.get("grid"), "网格边数", 2, 12, default=5)
    required = numeric(options.get("coverage"), "最低单格覆盖百分比", 10, 100, default=75) / 100
    strict = boolean(options.get("linear"), "sRGB 线性化", default=False)
    plane = make_plane(scene, region)
    plane.require_coverage(.20)
    valid = plane.observed
    signal = luma(linear_code(plane.rgb)) if strict else plane.gray
    rows = []
    boxes = []
    used = np.zeros(plane.shape, dtype=bool)
    for cell in partition(signal.shape[1], signal.shape[0], size, size):
        stats = coverage_stats(cell, valid, signal, required)
        rows.append(stats)
        highlight = (236, 241, 251) if stats["median"] is not None else (246, 162, 104)
        boxes.append((cell.box, f"{cell.row + 1}.{cell.column + 1}", highlight))
        if stats["median"] is not None:
            used[cell.y, cell.x] = cell.values(valid)
    if not np.any(used):
        raise Problem("当前覆盖门槛下没有可用于统计的网格。")
    accepted = [r for r in rows if r["median"] is not None]
    medians = np.array([r["median"] for r in accepted])
    center = float(np.median(medians))
    smallest = min(accepted, key=lambda row: row["median"])
    largest = max(accepted, key=lambda row: row["median"])
    drift = np.full(plane.shape, 0.0)
    np.divide(signal - center, center, out=drift, where=abs(center) > 1e-8)
    drift_range = max(.025, float(np.percentile(np.abs(drift[used]), 96)))
    metrics = {
        **plane.info(),
        "cells_used": len(accepted),
        "cells_total": len(rows),
        "min_p50_over_max_p50": stable_fraction(smallest["median"], largest["median"]),
        "relative_reference": rounding(center),
        "dimmest_cell": [smallest["row"], smallest["column"]],
        "brightest_cell": [largest["row"], largest["column"]],
        "scale": "sRGB 假设相对线性" if strict else "归一化码值",
        "heatmap_range": [-rounding(drift_range), rounding(drift_range)],
    }
    report = Report("空间均匀性", "以可信网格的中位数对比选区内的明暗分布。", metrics, rows)
    report.image("uniformity.png", overlay_regions(
        heatmap(drift, used, -drift_range, drift_range, diverging=True), boxes
    ))
    report.table("cells.csv", rows)
    if len(accepted) != len(rows):
        report.warnings.append("橙色边框网格覆盖不足，未计入最暗／最亮格与图示的统计。")
    return report


def light_leak(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    scene.require_role({"black", "uniform"}, "低黑位工具需要低灰、黑场或均匀灰场图。")
    plane = make_plane(scene, region, max_side=720)
    plane.require_coverage(.35)
    multiplier = numeric(options.get("sigma"), "噪声倍数", 2, 12, default=4)
    tolerance = numeric(options.get("floor"), "最小抬升码值", 0, .2, default=.012)
    min_area = numeric(options.get("size_pct"), "最小区域占比", .01, 3, default=.12) / 100
    gray = plane.gray
    valid = plane.observed
    values = gray[valid]
    center = float(np.median(values))
    noise = max(.002, 1.4826 * float(np.median(np.abs(values - center))))
    sigma = max(1.0, min(gray.shape) / 130)
    clean = gaussian_filter(np.where(valid, gray, 0.0), sigma)
    support = gaussian_filter(valid.astype(np.float32), sigma)
    np.divide(clean, support, out=clean, where=support > .5)
    threshold = center + max(noise * multiplier, tolerance)
    candidates = valid & (support > .91) & (clean > threshold)
    closed = binary_closing(candidates, structure=np.ones((3, 3)))
    clusters, count = label(candidates | (closed & valid), structure=np.ones((3, 3)))
    pixels_by_cluster = np.bincount(clusters.ravel())
    cutoff = max(5, round(valid.sum() * min_area))
    chosen = np.flatnonzero(pixels_by_cluster[1:] >= cutoff) + 1
    chosen = sorted(chosen, key=lambda item: (-pixels_by_cluster[item], item))[:60]
    positive = np.isin(clusters, chosen)
    report_rows = []
    boxes = []
    height, width = gray.shape
    for index, number in enumerate(chosen, start=1):
        contour = clusters == number
        ys, xs = np.where(contour)
        box = (xs.min() / width, ys.min() / height,
               (xs.max() + 1) / width, (ys.max() + 1) / height)
        boxes.append((box, str(index), (248, 231, 155)))
        report_rows.append({
            "region_id": index, "left": rounding(box[0]), "top": rounding(box[1]),
            "right": rounding(box[2]), "bottom": rounding(box[3]),
            "coverage_pct": rounding(100 * contour.sum() / valid.sum(), 3),
            "median_excess": rounding(np.median(gray[contour]) - center),
            "peak_smoothed": rounding(np.max(clean[contour])),
            "touches_selection_edge": bool(xs.min() == 0 or ys.min() == 0
                                           or xs.max() == width - 1 or ys.max() == height - 1),
        })
    top = max(tolerance, threshold - center, float(np.percentile((clean - center)[valid], 98)))
    stats = {
        **plane.info(),
        "background_code": rounding(center),
        "robust_noise_code": rounding(noise),
        "threshold_code": rounding(threshold),
        "region_count": len(report_rows),
        "nontrivial_raw_components": int(count),
        "leak_area_pct": rounding(100 * positive.sum() / valid.sum(), 2),
    }
    report = Report("低黑位漏光图", "保留超过低灰背景噪声且面积足够的连通亮区。", stats, report_rows)
    report.image("leak.png", overlay_regions(heatmap(clean - center, valid, 0, top), boxes))
    report.table("regions.csv", report_rows)
    if center > .25:
        report.warnings.append("采集背景码值偏亮，结果仅表示该采集内的相对抬升。")
    if not chosen:
        report.warnings.append("按当前面积和噪声门槛未保留候选区域。")
    return report


_RGB_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_D65 = np.array([0.95047, 1.0, 1.08883])


def lab_values(rgb: np.ndarray) -> np.ndarray:
    xyz = rgb @ _RGB_XYZ.T / _D65
    delta = 6 / 29
    transformed = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3 * delta ** 2) + 4 / 29,
    )
    fx, fy, fz = transformed.transpose(2, 0, 1)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def _chroma_grid(first_plane, chroma_values, valid, size):
    rows, boxes = [], []
    for cell in partition(first_plane.shape[1], first_plane.shape[0], size, size):
        sample = coverage_stats(cell, valid, chroma_values, .60)
        rows.append(sample)
        boxes.append((cell.box, str(len(rows)), (235, 241, 249) if sample["samples"] else (228, 143, 103)))
    return rows, boxes


def _reference_status_counts(correction, first_linear, valid, size):
    counts = {STATUS_BRACKETED: 0, STATUS_NEAR: 0, STATUS_EXTRAPOLATED: 0, STATUS_UNREFERENCED: 0}
    for cell in partition(first_linear.shape[1], first_linear.shape[0], size, size):
        keep = cell.values(valid)
        if keep.sum() < 4:
            status = STATUS_UNREFERENCED
        else:
            cell_luma = float(np.median(luma(first_linear)[cell.y, cell.x][keep]))
            status = correction.cell_status(cell_luma)
        counts[status] += 1
    return counts


def color_shift(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    first, target, overlap = paired_plane(scene, second, region, max_side=720)
    exposure_match = boolean(options.get("neutralize"), "亮度等比例归一", default=True)
    size = integral(options.get("grid"), "网格边数", 2, 9, default=4)
    gray_boxes = parse_gray_boxes(options.get("gray_boxes"))
    valid = overlap & first.usable(avoid_saturated=True) & target.usable(avoid_saturated=True)
    if valid.sum() < 100:
        raise Problem("两张采集中共同有效且未到码值极限的采样点不足。")
    first_linear = linear_code(first.rgb)
    second_linear = linear_code(target.rgb)
    first_y = luma(first_linear)
    second_y = luma(second_linear)
    ratio = float(np.median(first_y[valid]) / max(1e-8, np.median(second_y[valid])))
    effective = ratio if exposure_match else 1.0
    scaled = second_linear * effective
    base_valid = valid & (np.max(scaled, axis=-1) <= 1)
    base_valid = np.asarray(base_valid, dtype=bool)
    if base_valid.sum() < 100:
        raise Problem("亮度归一后留下的未裁切像素不足。")
    delta = lab_values(scaled) - lab_values(first_linear)
    full = np.linalg.norm(delta, axis=-1)
    chroma = np.linalg.norm(delta[..., 1:], axis=-1)
    stats = {
        **first.info(),
        "common_coverage_pct": rounding(100 * base_valid.mean(), 2),
        "exposure_gain": rounding(effective),
        "relative_median_lightness": rounding(np.median(delta[..., 0][base_valid])),
        "median_delta_e_76": rounding(np.median(full[base_valid]), 3),
        "median_chromatic_shift": rounding(np.median(chroma[base_valid]), 3),
    }
    rows, boxes = _chroma_grid(first, chroma, base_valid, size)
    scale = max(2.0, float(np.percentile(chroma[base_valid], 97)))
    stats["heatmap_max"] = rounding(scale, 3)
    report = Report("视角色偏比较", "在共同可见的面板区域比较两次采集的色度差。", stats, rows)
    base_overlay = overlay_regions(heatmap(chroma, base_valid, 0, scale), boxes)
    report.warnings.append("按 sRGB 与 D65 假设估计相对色差，未经相机／光谱校正。")

    if not gray_boxes:
        report.image("color-delta.png", base_overlay)
        report.table("color-cells.csv", rows)
        return report

    correction = gray_evaluate(gray_boxes, scene, second)
    report.warnings.extend(correction.warnings)
    report.image(
        "graycard-evidence.png",
        evidence_image(scene, second, correction),
    )
    report.table("graycard-patches.csv", patch_rows(correction))
    if not correction.available:
        report.image("color-delta.png", base_overlay)
        report.table("color-cells.csv", rows)
        report.metrics.update(correction_metrics(correction))
        return report

    corrected = correction.apply(second_linear)
    corrected_valid = (
        valid
        & np.all(corrected > 0, axis=-1)
        & (np.max(corrected, axis=-1) <= 1)
    )
    corrected_valid = np.asarray(corrected_valid, dtype=bool)
    if corrected_valid.sum() < 100:
        report.image("color-delta.png", base_overlay)
        report.table("color-cells.csv", rows)
        report.metrics.update(correction_metrics(correction))
        report.warnings.append(
            "灰片校正参数使面板像素大面积超出码值范围，已放弃校正并保留未校正结果，请复核灰片框。"
        )
        return report
    corrected_delta = lab_values(np.clip(corrected, 0, 1)) - lab_values(first_linear)
    corrected_full = np.linalg.norm(corrected_delta, axis=-1)
    corrected_chroma = np.linalg.norm(corrected_delta[..., 1:], axis=-1)
    corrected_rows, corrected_boxes = _chroma_grid(first, corrected_chroma, corrected_valid, size)
    reference_counts = _reference_status_counts(correction, first_linear, base_valid, size)
    cells = list(partition(first.shape[1], first.shape[0], size, size))
    for index, row in enumerate(rows):
        keep = cells[index].values(base_valid)
        if keep.sum() < 4:
            status = STATUS_UNREFERENCED
        else:
            cell_luma = float(np.median(first_y[cells[index].y, cells[index].x][keep]))
            status = correction.cell_status(cell_luma)
        row["reference_status"] = status
        row["uncorrected_median_chroma"] = row["median"]
        row["median"] = corrected_rows[index]["median"]
        row["corrected_median_chroma"] = corrected_rows[index]["median"]
        row["corrected_samples"] = corrected_rows[index]["samples"]
    corrected_overlay = overlay_regions(
        heatmap(corrected_chroma, corrected_valid, 0, scale), corrected_boxes
    )
    report.image(
        "color-delta-before-after.png",
        resize_report(side_by_side(
            titled_strip(base_overlay, "Uncorrected (global gain)"),
            titled_strip(corrected_overlay, "Gray-card corrected"),
        )),
    )
    report.image("color-delta-uncorrected.png", base_overlay)
    report.image("color-delta-corrected.png", corrected_overlay)
    report.table("color-cells.csv", rows)
    report.metrics.update(correction_metrics(correction))
    report.metrics.update({
        "common_coverage_pct": rounding(100 * corrected_valid.mean(), 2),
        "corrected_median_delta_e_76": rounding(np.median(corrected_full[corrected_valid]), 3),
        "corrected_median_chromatic_shift": rounding(
            np.median(corrected_chroma[corrected_valid]), 3
        ),
        "cells_with_reference": reference_counts[STATUS_BRACKETED] + reference_counts[STATUS_NEAR],
        "cells_bracketed": reference_counts[STATUS_BRACKETED],
        "cells_near_anchor": reference_counts[STATUS_NEAR],
        "cells_extrapolated": reference_counts[STATUS_EXTRAPOLATED],
        "cells_unreferenced": reference_counts[STATUS_UNREFERENCED],
    })
    return report


def checker_contrast(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    scene.require_role({"checker"}, "请选择棋盘格采集。")
    columns = integral(options.get("columns"), "棋盘列数", 2, 24, default=6)
    rows = integral(options.get("rows"), "棋盘行数", 2, 24, default=4)
    inset = numeric(options.get("guard"), "格内边缘忽略比例", 0, .36, default=.14)
    first_light = boolean(options.get("first_light"), "左上角为亮格", default=True)
    plane = make_plane(scene, region)
    plane.require_coverage(.22)
    gray = luma(linear_code(plane.rgb))
    measured = []
    kept = {True: [], False: []}
    boxes = []
    for cell in partition(plane.shape[1], plane.shape[0], columns, rows, inset):
        light = (cell.column + cell.row) % 2 == 0
        light = light == first_light
        summary = coverage_stats(cell, plane.observed, gray, .72)
        summary["expected"] = "white" if light else "black"
        measured.append(summary)
        if summary["median"] is not None:
            kept[light].append(summary["median"])
        tint = (236, 200, 95) if summary["median"] is None else (
            (232, 247, 241) if light else (98, 164, 239)
        )
        boxes.append((cell.box, "W" if light else "B", tint))
    if not kept[True] or not kept[False]:
        raise Problem("有效覆盖中没有同时出现可统计的亮格和暗格。")
    bright = float(np.median(kept[True]))
    dark = float(np.median(kept[False]))
    metrics = {
        **plane.info(),
        "white_cells": len(kept[True]),
        "black_cells": len(kept[False]),
        "white_level": rounding(bright),
        "black_level": rounding(dark),
        "contrast_ratio": stable_fraction(bright, dark),
        "michelson": stable_fraction(bright - dark, bright + dark),
    }
    report = Report("同屏棋盘格对比", "按预期亮暗位置在格心取样，汇总两类格的相对对比。", metrics, measured)
    report.image("checker-grid.png", overlay_regions(masked_rgb(plane.rgb, plane.observed), boxes))
    report.table("checker-cells.csv", measured)
    if dark >= bright:
        report.warnings.append("暗格值不低于亮格，请复核起始颜色、行列数与棋盘边界。")
    if dark < 1e-8:
        report.warnings.append("暗格接近码值零，亮暗比无法从该曝光中估算。")
    return report


def number(key, label, default, minimum, maximum, step=1):
    return {"key": key, "label": label, "type": "number", "default": default,
            "min": minimum, "max": maximum, "step": step}


def tick(key, label, default=False):
    return {"key": key, "label": label, "type": "checkbox", "default": default}


def configs():
    return {
        "uniformity": {
            "title": "空间均匀性",
            "group": "亮度与颜色",
            "description": "对均匀采集按网格检查相对亮度，剔除不足覆盖的格子。",
            "run": spatial_uniformity,
            "options": [number("grid", "网格边数", 5, 2, 12),
                        number("coverage", "最低单格覆盖 %", 75, 10, 100),
                        tick("linear", "sRGB 线性化")],
        },
        "leak": {
            "title": "低黑位漏光",
            "group": "亮度与颜色",
            "description": "在低灰场标定背景噪声，找持续高于背景的可观测区域。",
            "run": light_leak,
            "options": [number("sigma", "噪声倍数", 4, 2, 12, .2),
                        number("floor", "最小抬升码值", .012, 0, .2, .001),
                        number("size_pct", "最小区域占比 %", .12, .01, 3, .01)],
        },
        "color": {
            "title": "视角色偏",
            "group": "亮度与颜色",
            "description": "对齐同一面板的两张采集，查看共同可见区域的色度漂移；可在相机原图上框亮暗两档灰片以扣除补光偏色。",
            "comparison": "required",
            "run": color_shift,
            "options": [tick("neutralize", "亮度等比例归一", True), number("grid", "网格边数", 4, 2, 9),
                        {"key": "gray_boxes", "label": "灰片参考框（相机原图坐标，点击右侧按钮框选）",
                         "type": "graybox", "default": {"primary": [], "comparison": []}}],
        },
        "checker": {
            "title": "棋盘格对比",
            "group": "亮度与颜色",
            "description": "从棋盘格漫反射采集提取亮暗格，避开格线邻接污染。",
            "run": checker_contrast,
            "options": [number("columns", "棋盘列数", 6, 2, 24),
                        number("rows", "棋盘行数", 4, 2, 24),
                        number("guard", "格内边缘忽略比例", .14, 0, .36, .01),
                        tick("first_light", "左上角为亮格", True)],
        },
    }
