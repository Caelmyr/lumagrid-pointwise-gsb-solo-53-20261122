"""Inspection of a known module boundary on a calibrated panel capture."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from .assays import number
from .inspection import Report, Scene, make_plane
from .model import Problem, Region, choice, integral, numeric
from .raster import chart_png, heatmap, masked_rgb, overlay_regions, rounding


@dataclass
class CrossSections:
    gray: np.ndarray
    support: np.ndarray
    distance: np.ndarray
    along_mm: np.ndarray
    native_axis: str
    scale_x: float
    scale_y: float

    @classmethod
    def orient(cls, scene: Scene, plane, axis: str):
        signal = plane.gray
        observed = plane.observed
        height, width = observed.shape
        region = plane.region
        panel_width, panel_height = scene.scale
        if axis == "vertical":
            across_len = panel_width * (region.right - region.left)
            along_span = panel_height * (region.bottom - region.top)
            px_step = across_len / width
            coord_x = region.left + (np.arange(width) + .5) / width * (region.right - region.left)
            coord_y = region.top + (np.arange(height) + .5) / height * (region.bottom - region.top)
        else:
            signal = signal.T
            observed = observed.T
            across_len = panel_height * (region.bottom - region.top)
            along_span = panel_width * (region.right - region.left)
            px_step = across_len / height
            coord_x = region.top + (np.arange(height) + .5) / height * (region.bottom - region.top)
            coord_y = region.left + (np.arange(width) + .5) / width * (region.right - region.left)
        mm = scene.panel.get("height_mm" if axis == "vertical" else "width_mm")
        along = coord_y * mm if mm is not None else np.full(len(coord_y), np.nan)
        return cls(signal, observed, coord_x, along, axis, px_step, along_span / len(coord_y))


def row_background(values, valid, positions, seam_at, near, far):
    left = valid & (positions >= seam_at - far) & (positions <= seam_at - near)
    right = valid & (positions >= seam_at + near) & (positions <= seam_at + far)
    if left.sum() < 4 or right.sum() < 4:
        return None
    left_level, right_level = np.median(values[left]), np.median(values[right])
    left_x, right_x = np.median(positions[left]), np.median(positions[right])
    slope = (right_level - left_level) / (right_x - left_x)
    estimate = left_level + slope * (positions - left_x)
    return estimate, float(left_level), float(right_level), int(left.sum()), int(right.sum())


def seam_samples(
    sections: CrossSections,
    center: float,
    window_radius: float,
    flank_near: float,
    flank_far: float,
):
    test = (sections.distance >= center - window_radius) & (sections.distance <= center + window_radius)
    if test.sum() < 6:
        raise Problem("接缝检查带太窄，需在面板采样网格上覆盖至少六个横向位置。")
    positions = sections.distance[test]
    traces = np.full((sections.gray.shape[0], int(test.sum())), np.nan)
    observed = np.zeros_like(traces, dtype=bool)
    left_samples = np.full(sections.gray.shape[0], np.nan)
    right_samples = np.full(sections.gray.shape[0], np.nan)
    neighbor_counts = np.zeros((sections.gray.shape[0], 2), dtype=int)
    for i, (value, support) in enumerate(zip(sections.gray, sections.support, strict=True)):
        valid_band = support[test]
        if valid_band.sum() < .65 * test.sum():
            continue
        flank = row_background(value, support, sections.distance, center, flank_near, flank_far)
        if flank is None:
            continue
        reference, left, right, left_count, right_count = flank
        observed[i] = valid_band
        traces[i] = np.where(valid_band, value[test] - reference[test], np.nan)
        left_samples[i], right_samples[i] = left, right
        neighbor_counts[i] = left_count, right_count
    return positions, traces, observed, left_samples, right_samples, neighbor_counts


def valid_column_summary(profile: np.ndarray, support: np.ndarray, proportion: float):
    result = np.full(profile.shape[-1], np.nan)
    counts = support.sum(axis=0)
    qualifying = np.flatnonzero(counts >= max(3, round(proportion * len(profile))))
    for j in qualifying:
        result[j] = float(np.median(profile[:, j][support[:, j]]))
    return result


def peak_notch(positions: np.ndarray, values: np.ndarray, radius: float, code_floor: float):
    accepted = np.isfinite(values)
    if accepted.sum() < max(4, .65 * len(values)):
        return None
    positions = positions[accepted]
    samples = values[accepted]
    filtered = gaussian_filter1d(samples, .75)
    index = int(np.argmin(filtered))
    depth = -float(filtered[index])
    if depth < code_floor:
        return None
    criterion = -max(code_floor * .6, depth * .50)
    left = right = index
    while left > 0 and filtered[left - 1] <= criterion:
        left -= 1
    while right + 1 < len(filtered) and filtered[right + 1] <= criterion:
        right += 1
    pitch = np.median(np.diff(positions))
    return {
        "center": float(positions[index]),
        "width_normalized": min(2 * radius, float(positions[right] - positions[left] + pitch)),
        "depth": depth,
        "clipped_by_window": bool(left == 0 or right == len(filtered) - 1),
    }


def seam_assay(scene: Scene, second, region: Region, options: dict) -> Report:
    axis = choice(options.get("axis"), {"horizontal", "vertical"}, "接缝方向", default="vertical")
    center = numeric(options.get("center"), "名义接缝位置", .02, .98, default=.50)
    radius = numeric(options.get("radius"), "接缝检查带半宽", .004, .15, default=.028)
    flanks = numeric(options.get("flank"), "参照带外边距", radius + .015, .29, default=.087)
    slices = integral(options.get("segments"), "沿线区段数", 2, 18, default=8)
    delta_min = numeric(options.get("notch"), "最小凹陷码差", .003, .25, default=.023)
    begin, end = (region.left, region.right) if axis == "vertical" else (region.top, region.bottom)
    if center - flanks < begin or center + flanks > end:
        raise Problem("框选范围需同时覆盖名义接缝和两侧参照带。")
    plane = make_plane(scene, region, max_side=1000)
    plane.require_coverage(.24)
    visibility = CrossSections.orient(scene, plane, axis)
    positions, error, valid, left, right, counts = seam_samples(visibility, center, radius, radius * 1.22, flanks)
    if valid.sum() < .26 * valid.size:
        raise Problem("两侧都有亮度参考且可见的接缝片段过少。")
    physical_dimension = scene.panel["native_width" if axis == "vertical" else "native_height"]
    segments = np.array_split(np.arange(len(error)), slices)
    index_for_segment = np.full(error.shape[0], -1)
    rows = []
    chart_series = []
    for idx, subset in enumerate(segments, start=1):
        section_error = error[subset]
        seen = valid[subset]
        index_for_segment[subset] = idx
        summary = valid_column_summary(section_error, seen, .42)
        match = peak_notch(positions, summary, radius, delta_min)
        both = np.isfinite(left[subset]) & np.isfinite(right[subset])
        gradient = (right[subset][both] - left[subset][both])
        center_mm = visibility.along_mm[subset][len(subset) // 2]
        rows.append({
            "segment": idx,
            "center_along_mm": rounding(center_mm, 3),
            "valid_band_pct": rounding(100 * seen.mean(), 2),
            "effective_flank_lines": int(both.sum()),
            "flank_difference_code": rounding(np.median(gradient), 5) if gradient.size else None,
            "dark_notch_code": rounding(match["depth"], 5) if match else None,
            "notch_center_panel_px": rounding(match["center"] * physical_dimension, 2) if match else None,
            "notch_width_panel_px": rounding(match["width_normalized"] * physical_dimension, 2) if match else None,
            "width_reaches_window": match["clipped_by_window"] if match else None,
        })
        if idx in {1, len(segments) // 2, len(segments)}:
            chart_series.append((f"Segment {idx}", (positions - center) * physical_dimension, summary))
    box_lines = []
    def normalized_x(x):
        return (x - begin) / (end - begin)
    for index, subset in enumerate(segments, start=1):
        boundary_start = subset[0] / len(error)
        boundary_end = (subset[-1] + 1) / len(error)
        a, b = normalized_x(center - radius), normalized_x(center + radius)
        if axis == "vertical":
            box = (a, boundary_start, b, boundary_end)
        else:
            box = (boundary_start, a, boundary_end, b)
        color = (246, 184, 96) if rows[index - 1]["dark_notch_code"] else (157, 239, 214)
        box_lines.append((box, str(index), color))
    limit = max(delta_min, float(np.percentile(np.abs(error[valid]), 97)))
    count = sum(row["dark_notch_code"] is not None for row in rows)
    coverage = float(valid.mean())
    report = Report("定点屏缝剖面", "以手工给定的名义模块边界为基准，复核沿线连续性和两边亮度偏移。", {
        **plane.info(),
        "seam_axis": axis,
        "nominal_seam_panel_px": rounding(center * physical_dimension, 2),
        "segments_with_notch": count,
        "measured_segments": len(rows),
        "paired_band_pct": rounding(100 * coverage, 3),
        "profile_code_scale": [-rounding(limit), rounding(limit)],
    }, rows)
    report.image("seam-inspection-regions.png", overlay_regions(masked_rgb(plane.rgb, plane.observed), box_lines))
    plotted = error if axis == "vertical" else error.T
    shown = valid if axis == "vertical" else valid.T
    report.image("seam-residual-band.png", heatmap(plotted, shown, -limit, limit, diverging=True))
    report.add("seam-cross-sections.png", chart_png(
        chart_series, x_label="Relative panel pixels across seam", y_label="Relative recorded code",
    ), "image/png")
    report.table("seam-segments.csv", rows)
    report.warnings.append("名义位置由复核员给定；参照带仅估计该点两侧局部背景。")
    if count == 0:
        report.warnings.append("当前门槛下没有可信的暗槽；仍可查看两侧亮度差。")
    return report


def configs():
    return {
        "seam": {
            "title": "定点屏缝剖面",
            "group": "采样与细节",
            "description": "沿预先给定的模块边界复核沿线亮度凹槽及两侧偏移。",
            "run": seam_assay,
            "options": [
                {"key": "axis", "label": "接缝方向", "type": "select",
                 "values": {"vertical": "垂直接缝", "horizontal": "水平接缝"}, "default": "vertical"},
                number("center", "名义位置", .5, .02, .98, .001),
                number("radius", "检查带半宽", .028, .004, .15, .001),
                number("flank", "参照带外边距", .087, .02, .29, .001),
                number("segments", "沿线区段数", 8, 2, 18),
                number("notch", "最小凹陷码差", .023, .003, .25, .001),
            ],
        },
    }
