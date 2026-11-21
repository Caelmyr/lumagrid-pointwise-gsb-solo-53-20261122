"""Use a camera flat-field to characterize falloff and decentering without modifying sources."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .assays import number, tick
from .inspection import Report, Scene
from .model import Problem, Region, boolean, integral, numeric
from .raster import chart_png, heatmap, luma, masked_rgb, rounding
from .sensor import NativeWindow


@dataclass
class FitSamples:
    x: np.ndarray
    y: np.ndarray
    measured: np.ndarray
    count: np.ndarray


def camera_coordinates(width: int, height: int, win: NativeWindow):
    normalize = max(width, height) / 2
    x = (np.arange(win.left, win.right) - (width - 1) / 2) / normalize
    y = (np.arange(win.top, win.bottom) - (height - 1) / 2) / normalize
    return np.meshgrid(x, y)


def summary_samples(signal, valid, x, y, size):
    rows, columns = signal.shape
    row_break = np.linspace(0, rows, size + 1, dtype=int)
    col_break = np.linspace(0, columns, size + 1, dtype=int)
    anchors, medians, counts = [], [], []
    for i in range(size):
        sy = slice(row_break[i], row_break[i + 1])
        for j in range(size):
            sx = slice(col_break[j], col_break[j + 1])
            full = np.s_[sy, sx]
            coverage = valid[full]
            count = int(np.count_nonzero(coverage))
            if count < 20 or count < int(.42 * coverage.size):
                continue
            positions = median_anchor(x[full][coverage], y[full][coverage])
            anchors.append(positions)
            medians.append(float(np.median(signal[full][coverage])))
            counts.append(count)
    if len(anchors) < 18:
        raise Problem("可用平场区块不足，无法拟合光学中心。")
    location = np.asarray(anchors)
    if not np.all(np.ptp(location, axis=0) > .30):
        raise Problem("平场选区应在两个轴向留出足够跨度。")
    return FitSamples(location[:, 0], location[:, 1], np.asarray(medians), np.asarray(counts))


def median_anchor(x, y):
    return float(np.median(x)), float(np.median(y))


def radial_model(x, y, params):
    cx, cy, level, second, fourth = params
    distance = (x - cx) ** 2 + (y - cy) ** 2
    return level - second * distance - fourth * distance ** 2


def center_model(samples: FitSamples, keep_center: bool, floor: float):
    weights = np.sqrt(samples.count / max(1, np.median(samples.count)))
    initial_level = np.clip(np.percentile(samples.measured, 88), .035, 1.2)

    def measured_error(params):
        difference = radial_model(samples.x, samples.y, params) - samples.measured
        return difference * weights

    if keep_center:
        def locked_position(params):
            return measured_error((0.0, 0.0, *params))
        out = least_squares(
            locked_position, (initial_level, .09, .015),
            bounds=((.025, 0, 0), (1.25, .70, .30)),
            loss="soft_l1", f_scale=floor, max_nfev=180,
        )
        params = np.concatenate((np.zeros(2), out.x))
    else:
        out = least_squares(
            measured_error, (0.0, 0.0, initial_level, .09, .015),
            bounds=((-0.65, -0.65, .025, 0, 0), (.65, .65, 1.25, .70, .30)),
            loss="soft_l1", f_scale=floor, max_nfev=200,
        )
        params = out.x
    if not out.success:
        raise Problem("平场中心拟合没有收敛，请查看图案是否为稳定平场。")
    return params


def ring_sector_table(radius, angles, qualified, values, fitted, rings, sectors):
    limit = np.percentile(radius[qualified], 99.8)
    edges = np.linspace(0, max(limit, 1e-4), rings + 1)
    angle_edges = np.linspace(-np.pi, np.pi, sectors + 1)
    rows = []
    ring_curve = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True), start=1):
        is_ring = (radius >= a) & (radius < b) & qualified
        ring_values = values[is_ring]
        ring_prediction = fitted[is_ring]
        center = (a + b) / 2
        for j, (t0, t1) in enumerate(zip(angle_edges[:-1], angle_edges[1:], strict=True), start=1):
            mask = is_ring & (angles >= t0) & (angles < t1)
            samples = values[mask]
            baseline = fitted[mask]
            rows.append({
                "ring": i,
                "sector": j,
                "radius_lower": rounding(a),
                "radius_upper": rounding(b),
                "angle_start": rounding(np.rad2deg(t0), 2),
                "angle_end": rounding(np.rad2deg(t1), 2),
                "samples": len(samples),
                "median_code": rounding(np.median(samples)) if len(samples) >= 20 else None,
                "median_residual": rounding(np.median(samples - baseline)) if len(samples) >= 20 else None,
            })
        ring_curve.append((
            center,
            float(np.median(ring_values)) if ring_values.size >= 30 else float("nan"),
            float(np.median(ring_prediction)) if ring_prediction.size >= 30 else float("nan"),
        ))
    return rows, np.asarray(ring_curve)


def lens_falloff(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    scene.require_role({"flat"}, "请选择内容均一、曝光保留平场的镜头平场图。")
    anchors = integral(options.get("anchors"), "拟合网格", 4, 18, default=9)
    rings = integral(options.get("rings"), "半径区间数", 3, 18, default=8)
    sectors = integral(options.get("sectors"), "角度区间数", 4, 16, default=8)
    keep_center = boolean(options.get("lock_center"), "固定几何中心", default=False)
    clip_level = numeric(options.get("clip"), "有效码值上限", .45, .999, default=.975)
    robust_floor = numeric(options.get("robust_floor"), "稳健拟合码差", .003, .09, default=.02)
    window = NativeWindow.select(scene, region, min_width=64, limit=2_500_000)
    measured = window.extract(scene.image.rgb)
    bright = luma(measured)
    qualified = window.extract(scene.image.valid) & (measured.max(axis=-1) < clip_level)
    if qualified.mean() < .65:
        raise Problem("该平场区域饱和或无有效像素的位置过多。")
    width, height = scene.image.size
    x, y = camera_coordinates(width, height, window)
    sample = summary_samples(bright, qualified, x, y, anchors)
    params = center_model(sample, keep_center, robust_floor)
    cx, cy, level, second_order, fourth_order = params
    fitted = radial_model(x, y, params)
    radius = np.hypot(x - cx, y - cy)
    angles = np.arctan2(y - cy, x - cx)
    rows, circle = ring_sector_table(radius, angles, qualified, bright, fitted, rings, sectors)
    residual = bright - fitted
    relative_sd = np.sqrt(np.mean(residual[qualified] ** 2)) / level
    code_range = max(.009, float(np.percentile(np.abs(residual[qualified]), 96)))
    extent = max(width, height) / 2
    xs = cx * extent + (width - 1) / 2
    ys = cy * extent + (height - 1) / 2
    low, high = np.quantile(fitted[qualified], [.05, .95])
    not_flat = (float(high) - float(low)) / level
    confident = not_flat > .025 and relative_sd < .10 and (abs(cx) < .645 and abs(cy) < .645)
    metrics = {
        "source_window_px": window.coords(),
        "shared_coordinate_assumption": "原相机像素",
        "fit_tiles_used": len(sample.measured),
        "qualified_pct": rounding(100 * qualified.mean(), 3),
        "optical_center_assumed": keep_center,
        "optical_center_px": [rounding(xs, 3), rounding(ys, 3)],
        "center_estimate_supported": bool(confident),
        "model_center_code": rounding(level),
        "second_order_falloff": rounding(second_order),
        "fourth_order_falloff": rounding(fourth_order),
        "fitted_relative_falloff": rounding(not_flat),
        "relative_fit_error": rounding(relative_sd),
        "residual_heatmap_range": [-rounding(code_range), rounding(code_range)],
    }
    report = Report("镜头平场渐晕", "按镜头像素坐标检查径向平场衰减和偏离径向模型的区域。", metrics, rows[:96])
    report.image("lens-flatfield.png", masked_rgb(measured, qualified))
    report.image("radial-model-residual.png", heatmap(residual, qualified, -code_range, code_range, diverging=True))
    report.add("radial-falloff.png", chart_png(
        [("Recorded", circle[:, 0], circle[:, 1]), ("Radial fit", circle[:, 0], circle[:, 2])],
        x_label="Normalized source radius", y_label="Median camera luma",
    ), "image/png")
    report.table("flatfield-sectors.csv", rows)
    if not confident:
        report.warnings.append("渐晕特征偏弱或模型位置靠近约束边界，图中中心需手工核对。")
    report.warnings.append("径向模型用于诊断。表格同时保留扇区观测与残差，避免将偏色或遮挡变成镜头校正系数。")
    return report


def configs():
    return {
        "vignette": {
            "title": "镜头平场渐晕",
            "group": "采集质量",
            "description": "在原始平场内测径向衰减，查看偏心与角向分布的拟合残差。",
            "run": lens_falloff,
            "coordinate_space": "camera",
            "options": [
                number("anchors", "拟合网格", 9, 4, 18),
                number("rings", "半径区间数", 8, 3, 18),
                number("sectors", "角度区间数", 8, 4, 16),
                number("clip", "有效码值上限", .975, .45, .999, .001),
                number("robust_floor", "稳健拟合码差", .02, .003, .09, .001),
                tick("lock_center", "固定几何中心", False),
            ],
        },
    }
