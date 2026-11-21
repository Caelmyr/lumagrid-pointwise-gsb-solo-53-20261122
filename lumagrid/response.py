"""Response samples, optical sharpness, and observations across the acquisition line."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .assays import coverage_stats, number, partition, tick
from .inspection import Report, Scene, make_plane, stable_fraction
from .model import Point, Problem, Region, boolean, choice, integral, numeric
from .raster import chart_png, code_statistics, luma, masked_rgb, overlay_regions, rounding


@dataclass
class StepSamples:
    code: np.ndarray
    measured: np.ndarray
    weight: np.ndarray
    rows: list[dict]
    boxes: list[tuple]


def collect_steps(plane, count: int, axis: str, dark_first: bool) -> StepSamples:
    gray = plane.gray
    if axis == "horizontal":
        columns, row_count = count, 1
    else:
        columns, row_count = 1, count
    kept_code = []
    kept_signal = []
    kept_weight = []
    rows, boxes = [], []
    for cell in partition(plane.shape[1], plane.shape[0], columns, row_count, .17):
        index = cell.column + cell.row
        code = index / (count - 1)
        if not dark_first:
            code = 1 - code
        stats = coverage_stats(cell, plane.observed, gray, .72)
        values = cell.values(gray)[cell.values(plane.observed)]
        clipped = None if values.size == 0 else float(((values <= .005) | (values >= .995)).mean())
        accepted = stats["samples"] >= 24 and clipped is not None and clipped < .2
        rows.append({
            "patch": index + 1,
            "commanded_code": rounding(code),
            "coverage_pct": stats["coverage_pct"],
            "median_camera_code": stats["median"],
            "p05": stats["p05"],
            "p95": stats["p95"],
            "used_for_fit": accepted,
            "clipped_pct": rounding(100 * clipped, 2) if clipped is not None else None,
        })
        if accepted:
            patch_spread = max(.006, stats["p95"] - stats["p05"])
            kept_code.append(code)
            kept_signal.append(stats["median"])
            kept_weight.append(np.sqrt(stats["samples"]) / patch_spread)
        color = (213, 248, 219) if accepted else (245, 174, 117)
        boxes.append((cell.box, str(index + 1), color))
    return StepSamples(
        np.asarray(kept_code), np.asarray(kept_signal), np.asarray(kept_weight), rows, boxes
    )


def gamma_response(commanded, floor, span, gamma):
    return floor + span * commanded ** gamma


def fit_gamma(samples: StepSamples):
    if len(samples.code) < 5 or np.ptp(samples.code) < .42:
        raise Problem("拟合需要五个未裁切且覆盖至少约半个输入域的灰阶。")
    weights = samples.weight / np.median(samples.weight)
    weight = np.sqrt(np.clip(weights, .20, 5))

    def residual(params):
        return (gamma_response(samples.code, *params) - samples.measured) * weight

    result = least_squares(
        residual,
        (.018, .80, 1.05),
        bounds=((0.0, .03, .20), (.35, 1.15, 6.0)),
        loss="soft_l1",
        f_scale=.022,
        max_nfev=170,
    )
    if not result.success:
        raise Problem("灰阶响应拟合未收敛，请复核选区和图案方向。")
    residual_values = gamma_response(samples.code, *result.x) - samples.measured
    return result.x, residual_values


def tone_response(scene: Scene, second, region: Region, options: dict) -> Report:
    scene.require_role({"ramp"}, "请选择已知均匀码阶的灰阶采集。")
    count = integral(options.get("steps"), "阶数", 5, 32, default=9)
    axis = choice(options.get("axis"), {"horizontal", "vertical"}, "码阶方向", default="horizontal")
    dark_first = boolean(options.get("dark_first"), "先暗后亮", default=True)
    plane = make_plane(scene, region)
    plane.require_coverage(.35)
    samples = collect_steps(plane, count, axis, dark_first)
    params, residual = fit_gamma(samples)
    floor, span, gamma = params
    estimate_domain = np.linspace(float(samples.code.min()), float(samples.code.max()), 180)
    rows = samples.rows
    residual_by_code = {rounding(code): rounding(error)
                        for code, error in zip(samples.code, residual, strict=True)}
    for row in rows:
        row["fit_residual"] = residual_by_code.get(row["commanded_code"])
    metrics = {
        **plane.info(),
        "patches_used": len(samples.code),
        "scan_axis": axis,
        "estimation_domain": [rounding(estimate_domain[0]), rounding(estimate_domain[-1])],
        "gamma": rounding(gamma, 4),
        "floor_fit_parameter": rounding(floor, 5),
        "span_fit_parameter": rounding(span, 5),
        "fit_rmse": rounding(np.sqrt(np.mean(residual ** 2))),
    }
    good = np.argsort(samples.code)
    report = Report("灰阶传递曲线", "测量相机码阶与既定输入码阶的相对响应，保留实际拟合域。", metrics, rows)
    report.image("response-patches.png", overlay_regions(masked_rgb(plane.rgb, plane.observed), samples.boxes))
    report.add("response-curve.png", chart_png([
        ("Observed patches", samples.code[good], samples.measured[good]),
        ("Fitted response", estimate_domain, gamma_response(estimate_domain, *params)),
    ], x_label="Commanded code", y_label="Recorded code", x_bounds=(0, 1), y_bounds=(0, 1)), "image/png")
    report.table("response-patches.csv", rows)
    if metrics["fit_rmse"] > .055:
        report.warnings.append("残差较大，核对阶数及顺序，或改用亮暗覆盖更完整的采集。")
    if len(samples.code) != count:
        report.warnings.append("橙色网格有覆盖不足或码值饱和，未进入拟合。")
    return report


@dataclass(frozen=True)
class EdgeLine:
    intercept: float
    slope: float
    horizontal: bool
    used_scanlines: int
    residual_px: float

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dependent, independent = (y, x) if self.horizontal else (x, y)
        return (dependent - self.intercept - self.slope * independent) / np.hypot(1, self.slope)


def edge_positions(signal: np.ndarray, valid: np.ndarray, horizontal: bool):
    gray = signal.T if horizontal else signal
    mask = valid.T if horizontal else valid
    height, width = gray.shape
    if width < 18 or height < 24:
        raise Problem("斜边选区需要至少 18×24 个有效采样单元。")
    row_indices, centers, strengths = [], [], []
    for index, values in enumerate(gray):
        row_valid = mask[index]
        if row_valid.mean() < .78:
            continue
        clean = gaussian_filter1d(values, .8)
        gradients = np.abs(np.diff(clean))
        usable = row_valid[1:] & row_valid[:-1]
        usable[:3] = False
        usable[-3:] = False
        if not np.any(usable):
            continue
        gradients[~usable] = 0.0
        peak = int(np.argmax(gradients))
        strength = float(gradients[peak])
        if strength < .014:
            continue
        left, center, right = gradients[peak - 1:peak + 2]
        curvature = left - 2 * center + right
        offset = np.clip(.5 * (left - right) / curvature, -.5, .5) if curvature < -1e-7 else 0.0
        row_indices.append(index)
        centers.append(peak + .5 + offset)
        strengths.append(strength)
    return np.asarray(row_indices), np.asarray(centers), np.asarray(strengths)


def fit_edge(gray: np.ndarray, valid: np.ndarray, horizontal: bool) -> EdgeLine:
    along, across, strength = edge_positions(gray, valid, horizontal)
    if len(along) < 18:
        raise Problem("当前方向没有足够一致的斜边扫描线。")
    median_strength = np.median(strength)
    selected = strength >= median_strength * .35
    for _ in range(4):
        if selected.sum() < 18:
            raise Problem("边线支撑不足，选区应仅包含一条明显的边。")
        slope, intercept = np.polyfit(along[selected], across[selected], 1, w=np.sqrt(strength[selected]))
        residual = across - intercept - slope * along
        mad = np.median(np.abs(residual[selected] - np.median(residual[selected])))
        selected = (np.abs(residual) < max(1.25, mad * 3.2)) & (strength >= median_strength * .35)
    if selected.sum() < 18 or np.ptp(along[selected]) < .50 * (gray.shape[1] if horizontal else gray.shape[0]):
        raise Problem("斜边未经过足够长的选区，或直线拟合不稳定。")
    if abs(slope) > 0.60 or abs(slope) < .016:
        raise Problem("请改选方向合适且与像素网格略有夹角的斜边。")
    return EdgeLine(intercept, slope, horizontal, int(selected.sum()), float(np.std(residual[selected])))


def binned_spread(signal: np.ndarray, distances: np.ndarray, valid: np.ndarray):
    fine = .25
    half_width = 7.0
    count = round(2 * half_width / fine)
    mask = valid & (distances >= -half_width) & (distances < half_width)
    index = np.floor((distances[mask] + half_width) / fine).astype(np.intp)
    weight = np.bincount(index, minlength=count)
    total = np.bincount(index, weights=signal[mask], minlength=count)
    known = weight >= max(3, valid.shape[0] // 65)
    if known.sum() < .75 * count:
        raise Problem("边线距离各区间的数据不足，建议选择夹角更明确或较长的边。")
    if not np.all([known[:6].any(), known[-6:].any()]):
        raise Problem("斜边两侧必须都留有可见的稳定像素。")
    x = -half_width + (np.arange(count) + .5) * fine
    sampled = total[known] / weight[known]
    values = np.interp(x, x[known], sampled)
    left = np.median(values[:10])
    right = np.median(values[-10:])
    if abs(right - left) < .07:
        raise Problem("边两侧亮度差过小，无法可靠估算频率响应。")
    return x, (values - left) / (right - left), weight, fine, abs(right - left)


def mtf_response(esf: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    derivative = np.gradient(esf, step)
    windowed = derivative * np.hamming(derivative.size)
    frequency = np.fft.rfftfreq(512, d=step)
    response = np.abs(np.fft.rfft(windowed, n=512))
    response /= max(1e-8, response[0])
    below_nyquist = frequency <= .5
    return frequency[below_nyquist], response[below_nyquist]


def threshold_crossing(freq: np.ndarray, response: np.ndarray, target: float):
    below = np.flatnonzero(response < target)
    if not below.size:
        return None
    right = int(below[0])
    if right == 0:
        return rounding(freq[right])
    left = right - 1
    blend = (target - response[left]) / (response[right] - response[left])
    return rounding(freq[left] + (freq[right] - freq[left]) * blend, 5)


def sharpness(scene: Scene, second, region: Region, options: dict) -> Report:
    scene.require_role({"edge"}, "请选择包含斜边图案的采集。")
    direction = choice(options.get("axis"), {"vertical", "horizontal"}, "边线方向", default="vertical")
    plane = make_plane(scene, region, max_side=1050)
    plane.require_coverage(.62)
    gray = plane.gray
    edge = fit_edge(gray, plane.observed, direction == "horizontal")
    y, x = np.indices(gray.shape)
    distances = edge.distance(x, y)
    xs, esf, weight, step, contrast = binned_spread(gray, distances, plane.observed)
    freq, response = mtf_response(esf, step)
    metrics = {
        **plane.info(),
        "direction": direction,
        "edge_slope": rounding(edge.slope, 5),
        "edge_fit_error_px": rounding(edge.residual_px, 4),
        "scanlines_used": edge.used_scanlines,
        "relative_edge_contrast": rounding(contrast),
        "mtf50_cycles_per_sample": threshold_crossing(freq, response, .50),
        "mtf10_cycles_per_sample": threshold_crossing(freq, response, .10),
    }
    rows = [{"cycles_per_sample": rounding(f), "relative_response": rounding(a)}
            for f, a in zip(freq, response, strict=True)]
    report = Report("斜边清晰度", "从适当倾斜的明暗边估算法向的空间频率响应。", metrics, rows)
    highlighted = np.where((np.abs(distances) < .65)[..., None], [.34, .91, .72], plane.rgb)
    report.image("slanted-edge.png", masked_rgb(highlighted, plane.observed))
    report.add("mtf.png", chart_png(
        [("MTF", freq, response)],
        x_label="Cycles per rectified sample", y_label="Normalized response",
        x_bounds=(0, .5), y_bounds=(0, max(1.0, float(response.max()))),
    ), "image/png")
    report.add("edge-profile.png", chart_png(
        [("Edge spread", xs, esf)], x_label="Signed distance (samples)", y_label="Normalized edge",
    ), "image/png")
    report.table("mtf.csv", rows)
    report.table("edge-samples.csv", [
        {"distance": rounding(xv), "normalized_code": rounding(yv), "samples": int(n)}
        for xv, yv, n in zip(xs, esf, weight, strict=True)
    ])
    report.warnings.append("频率单位是展平图上的采样单元，结果包含当前采集链和重采样影响。")
    if metrics["mtf50_cycles_per_sample"] is None:
        report.warnings.append("MTF 在工作图奈奎斯特频率前未跌破 50%，不外推。")
    return report


def line_profile(scene: Scene, second, region: Region, options: dict) -> Report:
    direction = choice(options.get("axis"), {"horizontal", "vertical"}, "剖面方向", default="horizontal")
    strips = integral(options.get("strips"), "分带数", 1, 8, default=3)
    plane = make_plane(scene, region, max_side=650)
    plane.require_coverage(.10)
    gray = plane.gray
    valid = plane.observed
    height, width = gray.shape
    if direction == "vertical":
        gray, valid = gray.T, valid.T
        height, width = gray.shape
    steps = np.linspace(0, 1, width)
    series, rows = [], []
    for strip, item in enumerate(np.array_split(np.arange(height), strips), start=1):
        subset = gray[item]
        covered = valid[item]
        support = np.count_nonzero(covered, axis=0)
        aggregate = np.full(width, np.nan)
        np.divide(np.where(covered, subset, 0.0).sum(axis=0), support, out=aggregate, where=support > 0)
        empty = support < max(1, round(len(item) * .20))
        aggregate[empty] = np.nan
        series.append((f"Strip {strip}", steps, aggregate))
        rows.extend({
            "strip": strip,
            "relative_position": rounding(xs),
            "relative_code": rounding(ys),
            "samples": int(available),
        } for xs, ys, available in zip(steps, aggregate, support, strict=True))
    metrics = {**plane.info(), "axis": direction, "strip_count": strips, "points_per_strip": width}
    report = Report("面板剖面", "比较平行采样带的空间亮度，不连接无有效观察的采样点。", metrics)
    report.add("line-profile.png", chart_png(
        series[:6], x_label="Relative position within region", y_label="Recorded code", x_bounds=(0, 1),
    ), "image/png")
    report.table("line-profile.csv", rows)
    report.rows = [row for i, row in enumerate(rows) if i % max(1, width // 12) == 0][:96]
    if strips > 6:
        report.warnings.append("预览展示前六条剖面，表格保留所有采样带。")
    return report


def axis_option(default="horizontal"):
    return {"key": "axis", "label": "方向", "type": "select",
            "values": {"horizontal": "水平", "vertical": "垂直"}, "default": default}


def configs():
    return {
        "tone": {
            "title": "灰阶传递曲线",
            "group": "采样与细节",
            "description": "沿已知输入码阶提取样本，拟合响应并标出拟合的有效域。",
            "run": tone_response,
            "options": [number("steps", "码阶数", 9, 5, 32), axis_option(), tick("dark_first", "先暗后亮", True)],
        },
        "mtf": {
            "title": "斜边清晰度",
            "group": "采样与细节",
            "description": "从单条斜边估算边展宽、MTF50 和 MTF10。",
            "run": sharpness,
            "options": [axis_option("vertical")],
        },
        "profile": {
            "title": "面板剖面",
            "group": "采样与细节",
            "description": "以平行采样带追踪亮度变化，导出具体采样支撑。",
            "run": line_profile,
            "options": [axis_option(), number("strips", "分带数", 3, 1, 8)],
        },
    }
