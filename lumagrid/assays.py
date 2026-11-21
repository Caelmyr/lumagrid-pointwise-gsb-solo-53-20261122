"""Spatial light quality assays on aligned camera captures."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter, label

from .inspection import Report, Scene, make_plane, paired_plane, stable_fraction
from .model import Problem, Region, boolean, choice, integral, numeric
from .raster import (
    chart_png, code_statistics, heatmap, linear_code, link_regions, luma, masked_rgb,
    overlay_regions, paired_bounds, png_bytes, rounding,
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


def color_shift(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    first, target, overlap = paired_plane(scene, second, region, max_side=720)
    exposure_match = boolean(options.get("neutralize"), "亮度等比例归一", default=True)
    size = integral(options.get("grid"), "网格边数", 2, 9, default=4)
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
    valid &= np.max(scaled, axis=-1) <= 1
    if valid.sum() < 100:
        raise Problem("亮度归一后留下的未裁切像素不足。")
    delta = lab_values(scaled) - lab_values(first_linear)
    full = np.linalg.norm(delta, axis=-1)
    chroma = np.linalg.norm(delta[..., 1:], axis=-1)
    stats = {
        **first.info(),
        "common_coverage_pct": rounding(100 * valid.mean(), 2),
        "exposure_gain": rounding(effective),
        "relative_median_lightness": rounding(np.median(delta[..., 0][valid])),
        "median_delta_e_76": rounding(np.median(full[valid]), 3),
        "median_chromatic_shift": rounding(np.median(chroma[valid]), 3),
    }
    rows, boxes = [], []
    for cell in partition(first.shape[1], first.shape[0], size, size):
        sample = coverage_stats(cell, valid, chroma, .60)
        rows.append(sample)
        boxes.append((cell.box, str(len(rows)), (235, 241, 249) if sample["samples"] else (228, 143, 103)))
    scale = max(2.0, float(np.percentile(chroma[valid], 97)))
    stats["heatmap_max"] = rounding(scale, 3)
    report = Report("视角色偏比较", "在共同可见的面板区域比较两次采集的色度差。", stats, rows)
    report.image("color-delta.png", overlay_regions(heatmap(chroma, valid, 0, scale), boxes))
    report.table("color-cells.csv", rows)
    report.warnings.append("按 sRGB 与 D65 假设估计相对色差，未经相机／光谱校正。")
    return report


def blocked_cells(value, columns: int, rows: int) -> set:
    if value is None:
        return set()
    if not isinstance(value, (list, tuple)) or len(value) > columns * rows:
        raise Problem("遮挡格应为 [行, 列] 组成的数组。")
    blocked = set()
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise Problem("遮挡格每项应写作 [行, 列]。")
        blocked.add((integral(item[0], "遮挡格行号", 1, rows),
                     integral(item[1], "遮挡格列号", 1, columns)))
    return blocked


def checker_contrast(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    scene.require_role({"checker"}, "请选择棋盘格采集。")
    columns = integral(options.get("columns"), "棋盘列数", 2, 24, default=6)
    rows = integral(options.get("rows"), "棋盘行数", 2, 24, default=4)
    inset = numeric(options.get("guard"), "格内边缘忽略比例", 0, .36, default=.14)
    first_light = boolean(options.get("first_light"), "左上角为亮格", default=True)
    blocked = blocked_cells(options.get("blocked"), columns, rows)
    mode = choice(options.get("mode"), {"paired", "aggregate"}, "对比方式", default="paired")
    plane = make_plane(scene, region)
    plane.require_coverage(.22)
    gray = luma(linear_code(plane.rgb))
    cells = []
    for cell in partition(plane.shape[1], plane.shape[0], columns, rows, inset):
        light = ((cell.column + cell.row) % 2 == 0) == first_light
        summary = coverage_stats(cell, plane.observed, gray, .72)
        summary["expected"] = "white" if light else "black"
        cells.append({
            "row": cell.row, "column": cell.column, "light": light, "box": cell.box,
            "blocked": (cell.row + 1, cell.column + 1) in blocked, "summary": summary,
        })
    if mode == "aggregate":
        return checker_aggregate(plane, cells, blocked)
    return checker_paired(plane, cells, blocked)


def checker_aggregate(plane, cells: list, blocked: set) -> Report:
    measured = []
    kept = {True: [], False: []}
    boxes = []
    for cell in cells:
        light, summary = cell["light"], cell["summary"]
        if cell["blocked"]:
            summary = {**summary, "state": "夹具遮挡"}
            boxes.append((cell["box"], "挡", (228, 143, 103)))
        else:
            if summary["median"] is not None:
                kept[light].append(summary["median"])
            tint = (236, 200, 95) if summary["median"] is None else (
                (232, 247, 241) if light else (98, 164, 239)
            )
            boxes.append((cell["box"], "W" if light else "B", tint))
        measured.append(summary)
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
    if blocked:
        report.warnings.append(f"已按标记排除 {len(blocked)} 个夹具遮挡格。")
    if dark >= bright:
        report.warnings.append("暗格值不低于亮格，请复核起始颜色、行列数与棋盘边界。")
    if dark < 1e-8:
        report.warnings.append("暗格接近码值零，亮暗比无法从该曝光中估算。")
    return report


def checker_paired(plane, cells: list, blocked: set) -> Report:
    usable = {(cell["row"], cell["column"]): cell for cell in cells
              if not cell["blocked"] and cell["summary"]["median"] is not None}
    steps = ((-1, 0), (0, -1), (0, 1), (1, 0))
    neighbors = {key: [(key[0] + dr, key[1] + dc) for dr, dc in steps
                       if (key[0] + dr, key[1] + dc) in usable]
                 for key, cell in usable.items() if cell["light"]}
    matched = {}

    def extend(start, seen):
        for target in neighbors[start]:
            if target in seen:
                continue
            seen.add(target)
            if target not in matched or extend(matched[target], seen):
                matched[target] = start
                return True
        return False

    for key in sorted(neighbors):
        extend(key, set())
    pairs = sorted((bright, dark) for dark, bright in matched.items())
    if not pairs:
        raise Problem("有效格中没有相邻的亮暗对，请检查遮挡标记、行列数与棋盘边界。")

    def place(key):
        return f"{key[0] + 1},{key[1] + 1}"

    rows_out, links, used = [], [], set()
    for index, (bright_key, dark_key) in enumerate(pairs, 1):
        label = f"P{index}"
        high = usable[bright_key]["summary"]["median"]
        low = usable[dark_key]["summary"]["median"]
        rows_out.append({
            "pair": label, "state": "配对",
            "white_cell": place(bright_key), "black_cell": place(dark_key),
            "white_level": high, "black_level": low,
            "contrast_ratio": stable_fraction(high, low),
            "michelson": stable_fraction(high - low, high + low),
        })
        links.append((usable[bright_key]["box"], usable[dark_key]["box"]))
        usable[bright_key]["pair"] = usable[dark_key]["pair"] = label
        used.update((bright_key, dark_key))
    cell_rows = []
    for cell in cells:
        key, summary = (cell["row"], cell["column"]), cell["summary"]
        if cell["blocked"]:
            state = "夹具遮挡"
        elif summary["median"] is None:
            state = "覆盖不足"
        elif key in used:
            state = f"配对 {cell['pair']}"
        else:
            state = "未配对留空"
        cell_rows.append({**summary, "state": state, "pair": cell.get("pair", "")})
        if key in used:
            continue
        entry = {"pair": "", "state": state}
        entry["white_cell" if cell["light"] else "black_cell"] = place(key)
        if state == "覆盖不足":
            entry["coverage_pct"] = summary["coverage_pct"]
        elif state == "未配对留空":
            entry["white_level" if cell["light"] else "black_level"] = summary["median"]
        rows_out.append(entry)
    ratios = [row["contrast_ratio"] for row in rows_out if row.get("contrast_ratio") is not None]
    michelsons = [row["michelson"] for row in rows_out if row.get("michelson") is not None]
    metrics = {
        **plane.info(),
        "compare_mode": "就近亮暗配对",
        "blocked_cells": len(blocked),
        "measured_cells": len(usable),
        "pairs": len(pairs),
        "paired_cells": 2 * len(pairs),
        "unpaired_cells": len(usable) - 2 * len(pairs),
        "pair_contrast_median": rounding(np.median(ratios)) if ratios else None,
        "pair_contrast_min": rounding(min(ratios)) if ratios else None,
        "pair_contrast_max": rounding(max(ratios)) if ratios else None,
        "pair_michelson_median": rounding(np.median(michelsons)) if michelsons else None,
    }
    boxes = []
    for cell in cells:
        pair = cell.get("pair")
        if cell["blocked"]:
            tint, text = (228, 143, 103), "挡"
        elif pair:
            tint = (232, 247, 241) if cell["light"] else (98, 164, 239)
            text = pair
        elif cell["summary"]["median"] is None:
            tint, text = (236, 200, 95), "W" if cell["light"] else "B"
        else:
            tint, text = (196, 205, 214), "W" if cell["light"] else "B"
        boxes.append((cell["box"], text, tint))
    canvas = link_regions(masked_rgb(plane.rgb, plane.observed), links)
    summary = "每格只与一枚相邻异色格配对，配对编号标在图上；无可信邻格的格子留空，不进入汇总。"
    report = Report("棋盘格就近亮暗对比", summary, metrics, rows_out)
    report.image("checker-pairs.png", overlay_regions(canvas, boxes))
    report.table("checker-pairs.csv", rows_out)
    report.table("checker-cells.csv", cell_rows)
    if blocked:
        report.warnings.append(f"已按标记排除 {len(blocked)} 个夹具遮挡格。")
    if metrics["unpaired_cells"]:
        report.warnings.append(f"{metrics['unpaired_cells']} 个有效格没有可信的相邻异色格，已在结果中留空。")
    inverted = sum(1 for row in rows_out if row["state"] == "配对"
                   and row["black_level"] >= row["white_level"])
    if inverted:
        report.warnings.append(f"{inverted} 对格子的暗格读数不低于亮格，请复核起始颜色、行列数与棋盘边界。")
    if len(ratios) < len(pairs):
        report.warnings.append("存在暗格接近码值零的配对，对应亮暗比记为空。")
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
            "description": "对齐同一面板的两张采集，查看共同可见区域的色度漂移。",
            "comparison": "required",
            "run": color_shift,
            "options": [tick("neutralize", "亮度等比例归一", True), number("grid", "网格边数", 4, 2, 9)],
        },
        "checker": {
            "title": "棋盘格对比",
            "group": "亮度与颜色",
            "description": "从棋盘格漫反射采集提取亮暗格，可就近配对或整图汇总，支持标记夹具遮挡格。",
            "run": checker_contrast,
            "cell_marking": True,
            "options": [{"key": "mode", "label": "对比方式", "type": "select",
                         "values": {"paired": "就近亮暗配对", "aggregate": "整图亮暗汇总"},
                         "default": "paired"},
                        number("columns", "棋盘列数", 6, 2, 24),
                        number("rows", "棋盘行数", 4, 2, 24),
                        number("guard", "格内边缘忽略比例", .14, 0, .36, .01),
                        tick("first_light", "左上角为亮格", True),
                        {"key": "blocked", "label": "夹具遮挡格 [行,列]，可点「标出遮挡格」在图上点选",
                         "type": "json", "default": []}],
        },
    }
