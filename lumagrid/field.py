"""Localized panel-edge measurement and region workflow planning from optical checks."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from scipy.optimize import least_squares

from .assays import number, partition, tick
from .inspection import Report, Scene, make_plane, paired_plane
from .model import Problem, Region, boolean, choice, integral, numeric
from .raster import chart_png, code_statistics, heatmap, luma, masked_rgb, overlay_regions, rounding


@dataclass
class Margin:
    side: str
    outline: np.ndarray
    line_index: np.ndarray
    available: np.ndarray
    contrast: np.ndarray
    inner_sign: float


def margin_layout(side, raster, mask):
    if side in {"upper", "lower"}:
        plane, valid = raster.T, mask.T
    else:
        plane, valid = raster, mask
    if side in {"lower", "right"}:
        plane = plane[:, ::-1]
        valid = valid[:, ::-1]
    return plane, valid


def edge_at(values, observed, span: float, sigma: float):
    length = len(values)
    min_at = max(3, round(.025 * length))
    max_at = min(length - 4, round(span * length))
    if max_at <= min_at:
        return None
    profile = gaussian_filter(values.astype(np.float64), sigma)
    gradient = np.diff(profile)
    proof = observed[1:] & observed[:-1]
    left_right = (np.arange(len(gradient)) >= min_at) & (np.arange(len(gradient)) <= max_at)
    can_use = proof & left_right
    if not np.any(can_use):
        return None
    scores = np.abs(gradient)
    scores[~can_use] = 0
    best = int(np.argmax(scores))
    cutoff = max(3, round(.025 * length))
    if best - cutoff < 0 or best + cutoff + 1 > length:
        return None
    outs = slice(best - cutoff, best)
    ins = slice(best + 1, best + cutoff + 1)
    if min(observed[outs].mean(), observed[ins].mean()) < .85:
        return None
    outside = np.median(profile[outs])
    inside = np.median(profile[ins])
    strength = inside - outside
    if abs(strength) < .022:
        return None
    left, center, right = scores[best - 1:best + 2]
    curvature = left - 2.0 * center + right
    fractional = .5 * (left - right) / curvature if curvature < -1e-6 else 0
    return best + .5 + np.clip(fractional, -.5, .5), strength


def gather_margin(raster, observed, side: str, width_frac: float, sigma: float):
    aligned, valid = margin_layout(side, raster, observed)
    lines, width = aligned.shape
    bands = np.linspace(0, lines, min(96, lines) + 1, dtype=int)
    location = np.full(len(bands) - 1, np.nan)
    strength = np.full_like(location, np.nan)
    left_right = (bands[1:] + bands[:-1]) * .5
    for i, (start, stop) in enumerate(zip(bands[:-1], bands[1:], strict=True)):
        section = aligned[start:stop]
        supported = valid[start:stop]
        usable = supported.mean(axis=0) > .70
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.divide(np.where(supported, section, 0).sum(axis=0), supported.sum(axis=0))
        filled = np.interp(np.arange(width), np.flatnonzero(usable), means[usable]) if usable.sum() >= 9 else None
        if filled is None:
            continue
        edge = edge_at(filled, usable, width_frac, sigma)
        if edge:
            location[i], strength[i] = edge
    avail = np.isfinite(location)
    sign = float(np.sign(np.nanmedian(strength))) if avail.any() else 0
    avail &= np.sign(strength) == sign
    return Margin(side, location, left_right, avail, strength, sign)


def straightness_fit(margin: Margin):
    index = margin.line_index[margin.available]
    picked = margin.outline[margin.available]
    if len(index) < 7 or np.ptp(index) < margin.line_index[-1] * .35:
        return None
    start = float(index[0])
    span = float(np.ptp(index))
    scaled = (index - start) / span

    def prediction(xx, coefficients):
        return coefficients[0] + coefficients[1] * xx + coefficients[2] * (xx - .5) ** 2

    def mismatch(coefficients):
        return prediction(scaled, coefficients) - picked

    result = least_squares(mismatch, (np.median(picked), 0.0, 0.0),
                           loss="soft_l1", f_scale=1.8, max_nfev=140)
    if not result.success:
        return None
    predicted = prediction((margin.line_index - start) / span, result.x)
    return predicted, result.x


def projected_margin_points(margin: Margin, sample_shape: tuple[int, int], values: np.ndarray):
    height, width = sample_shape
    start = margin.line_index
    if margin.side == "upper":
        return (start / width, (values + .5) / height)
    if margin.side == "lower":
        return (start / width, 1 - (values + .5) / height)
    if margin.side == "left":
        return ((values + .5) / width, start / height)
    return (1 - (values + .5) / width, start / height)


def panel_steps(scene: Scene, region: Region, shape: tuple[int, int], side: str):
    height, width = shape
    if side in {"upper", "lower"}:
        return scene.scale[1] * (region.bottom - region.top) / height
    return scene.scale[0] * (region.right - region.left) / width


def outline_check(scene: Scene, second, region: Region, options: dict):
    span = numeric(options.get("window"), "边缘搜索窗占比", .04, .38, default=.19)
    smooth = numeric(options.get("smooth"), "沿剖面去噪", .5, 6, default=1.5)
    plane = make_plane(scene, region, max_side=980)
    plane.require_coverage(.45)
    selected = choice(
        options.get("side"),
        {"all", "upper", "lower", "left", "right"},
        "检查边界",
        default="all",
    )
    sides = ("upper", "right", "lower", "left") if selected == "all" else (selected,)
    inset_px = []
    side_rows = []
    trace = []
    lines = []
    visualization = masked_rgb(plane.rgb, plane.observed)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(visualization)
    directions = {"upper": "U", "lower": "D", "left": "L", "right": "R"}
    for side in sides:
        margin = gather_margin(plane.gray, plane.observed, side, span, smooth)
        coverage = int(margin.available.sum()) / len(margin.available)
        estimate = straightness_fit(margin)
        converter = panel_steps(scene, region, plane.shape, side)
        if estimate is None:
            side_rows.append({
                "side": directions[side], "coverage_pct": rounding(100 * coverage, 2),
                "median_inset_panel_px": None, "rms_fit_panel_px": None,
                "midpoint_curvature_panel_px": None,
            })
            continue
        predicted, terms = estimate
        error = (margin.outline - predicted) * converter
        supported = margin.available
        depth = margin.outline[supported] * converter
        middle = float(np.median(depth))
        inset_px.append(middle)
        side_rows.append({
            "side": directions[side],
            "coverage_pct": rounding(coverage * 100, 2),
            "median_inset_panel_px": rounding(middle, 3),
            "rms_fit_panel_px": rounding(np.sqrt(np.mean(error[supported] ** 2)), 3),
            "midpoint_curvature_panel_px": rounding(-.25 * terms[2] * converter, 3),
            "edge_contrast": rounding(np.median(np.abs(margin.contrast[supported])), 3),
        })
        length = margin.line_index[-1]
        relative = margin.line_index / length
        trace.append((directions[side], relative, np.where(supported, error, np.nan)))
        found_x, found_y = projected_margin_points(margin, plane.shape, margin.outline)
        guess_x, guess_y = projected_margin_points(margin, plane.shape, predicted)
        representative = list(zip(guess_x[supported], guess_y[supported], strict=True))
        colored = [(x * visualization.width, y * visualization.height) for x, y in representative]
        if len(colored) >= 2:
            draw.line(colored, fill=(129, 240, 213), width=2)
        for rx, ry, px, py, ok, err in zip(
            found_x, found_y, guess_x, guess_y, supported, error, strict=True,
        ):
            if not ok:
                continue
            x, y = rx * visualization.width, ry * visualization.height
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(246, 201, 93))
            lines.append({
                "side": directions[side],
                "observed_panel_x": rounding(region.left + rx * (region.right - region.left)),
                "observed_panel_y": rounding(region.top + ry * (region.bottom - region.top)),
                "fitted_x": rounding(region.left + px * (region.right - region.left)),
                "fitted_y": rounding(region.top + py * (region.bottom - region.top)),
                "inset_residual_panel_px": rounding(err, 3),
            })
    report = Report("有效画面边界", "沿面板选区边缘对比连续明暗变化，给出有效画面边缘误差。", {
        **plane.info(),
        "usable_sides": len(inset_px),
        "mean_edge_inset_panel_px": rounding(np.mean(inset_px), 3) if inset_px else None,
        "edge_search_fraction": span,
    }, side_rows)
    report.image("display-edge-location.png", visualization)
    if trace:
        report.add("display-edge-residual.png", chart_png(
            trace, x_label="Position along side", y_label="Boundary fit residual, panel px",
        ), "image/png")
    report.table("display-edge-observations.csv", lines)
    report.table("display-edge-summary.csv", side_rows)
    if len(inset_px) < len(sides):
        report.warnings.append("部分边没有足够连续的明暗变化；请把选区框在面板边界外侧，并复核出画位置。")
    return report


def region_transition(scene: Scene, second, region: Region, options: dict):
    first, other, overlap = paired_plane(scene, second, region, max_side=850)
    floor = numeric(options.get("floor"), "变化码值", .005, .3, default=.045)
    blur = numeric(options.get("sigma"), "背景平滑尺度", 1, 21, default=7)
    subtract = boolean(options.get("undo_ramp"), "从差值减去连续背景", default=True)
    signal = other.gray - first.gray
    revised = signal
    if subtract:
        local = gaussian_filter(np.where(overlap, signal, 0), blur)
        support = gaussian_filter(overlap.astype(float), blur)
        local = np.divide(local, support, where=support > .60, out=np.zeros_like(local))
        revised = signal - local
        overlap &= support > .85
    else:
        revised = signal - np.median(signal[overlap])
    if overlap.sum() < 80:
        raise Problem("有效的局部变化采样过少。")
    comparable = revised[overlap]
    upper = max(floor, float(np.percentile(np.abs(comparable), 98)))
    signed = np.zeros_like(revised)
    signed[(revised > floor) & overlap] = 1
    signed[(revised < -floor) & overlap] = -1
    hist, edges = np.histogram(comparable, bins=80, range=(-upper, upper))
    rows = []
    for cell in partition(revised.shape[1], revised.shape[0], 5, 5):
        seen = cell.values(overlap)
        relative = cell.values(revised)[seen]
        positive = (cell.values(signed)[seen] == 1).sum()
        negative = (cell.values(signed)[seen] == -1).sum()
        zones = {
            "row": cell.row + 1, "column": cell.column + 1,
            "coverage_pct": rounding(seen.mean() * 100, 2),
            "kept": int(len(relative)),
            "positive_count": int(positive),
            "negative_count": int(negative),
            "mean_absolute_change": rounding(np.abs(relative).mean()) if len(relative) else None,
        }
        rows.append(zones)
    report = Report("局部切换差分", "从切换前后的共同面板坐标核对亮度变化方向及占比。", {
        **first.info(),
        "common_coverage_pct": rounding(overlap.mean() * 100, 2),
        "meaningful_positive_pct": rounding(100 * np.mean(comparable > floor), 3),
        "meaningful_negative_pct": rounding(100 * np.mean(comparable < -floor), 3),
        "transition_stats": code_statistics(comparable),
        "heatmap_levels": [-rounding(upper), rounding(upper)],
    }, rows)
    report.image("panel-transition.png", heatmap(revised, overlap, -upper, upper, diverging=True))
    report.add("transition-distribution.png", chart_png(
        [("Common area", (edges[1:] + edges[:-1]) / 2, hist / max(1, hist.sum()))],
        x_label="Comparison minus primary code", y_label="Fraction of plotted samples",
        x_bounds=(-upper, upper),
        markers=[(-floor, "Lower"), (floor, "Upper")],
    ), "image/png")
    report.table("local-transition-grid.csv", rows)
    report.warnings.append("差分图为单对图像的局部变化；光照、曝光和遮挡也会影响其表现。")
    return report


def configs():
    return {
        "outline": {
            "title": "有效画面边界",
            "group": "采集坐标",
            "description": "找出框选范围四边的可见过渡和不同边段的定位偏差。",
            "run": outline_check,
            "options": [
                {"key": "side", "label": "检查边界", "type": "select",
                 "values": {"all": "四边", "upper": "上", "lower": "下", "left": "左", "right": "右"}, "default": "all"},
                number("window", "沿边搜索窗比例", .19, .04, .38, .01),
                number("smooth", "沿剖面去噪", 1.5, .5, 6, .25),
            ],
        },
        "transition": {
            "title": "局部切换差分",
            "group": "对照证据",
            "description": "以一对平面采集定位切换后的局部亮度变化。",
            "run": region_transition,
            "comparison": "required",
            "options": [
                number("floor", "变化码值", .045, .005, .3, .005),
                number("sigma", "背景平滑尺度", 7, 1, 21, .5),
                tick("undo_ramp", "从差值减去连续背景", True),
            ],
        },
    }
