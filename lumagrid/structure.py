"""Repeated optical structure, illumination-dependent marks, and panel measurements."""
from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy.ndimage import binary_opening, gaussian_filter, gaussian_filter1d, label
from scipy.signal import find_peaks

from .assays import number, tick
from .geometry import segment_statistics
from .inspection import Report, Scene, make_plane, paired_plane
from .model import Point, Problem, Region, choice, integral, numeric, sequence
from .raster import chart_png, heatmap, luma, masked_rgb, overlay_regions, rounding
from .response import axis_option


@dataclass
class Spectrum:
    axis: str
    frequency: np.ndarray
    amplitude: np.ndarray
    samples: int
    lines: int

    @property
    def pitch_axis(self):
        return np.divide(1.0, self.frequency, out=np.zeros_like(self.frequency), where=self.frequency > 0)


def valid_scanline_profiles(rgb: np.ndarray, valid: np.ndarray, vertical: bool):
    image = rgb.transpose(1, 0, 2) if vertical else rgb
    mask = valid.T if vertical else valid
    supported = mask.mean(axis=1) >= .90
    if supported.sum() < 10:
        raise Problem("选区没有足够完整的平行采样线。")
    profiles = np.zeros_like(image[supported], dtype=np.float32)
    x = np.arange(image.shape[1])
    for i, (row, keep) in enumerate(zip(image[supported], mask[supported], strict=True)):
        for channel in range(3):
            profiles[i, :, channel] = np.interp(x, x[keep], row[keep, channel])
    return profiles


def spatial_spectrum(profiles: np.ndarray, axis: str, band: str) -> Spectrum:
    length = profiles.shape[1]
    if length < 48:
        raise Problem("频谱剖面需要沿扫描方向至少 48 个采样点。")
    source = luma(profiles) if band == "luma" else profiles[..., "rgb".index(band)]
    low = gaussian_filter1d(source, sigma=max(3, length / 17), axis=-1, mode="reflect")
    fluctuation = source - low
    window = np.hanning(length)
    power = np.abs(np.fft.rfft(fluctuation * window, axis=-1))
    robust = np.median(power, axis=0) * (2 / max(window.sum(), 1))
    return Spectrum(axis, np.fft.rfftfreq(length), robust, length, profiles.shape[0])


def peak_frequency(spectrum: Spectrum, minimum_pitch: float, maximum_pitch: float, count: int):
    freqs, power = spectrum.frequency, spectrum.amplitude
    pitch = spectrum.pitch_axis
    eligible = (pitch >= minimum_pitch) & (pitch <= maximum_pitch) & (freqs > 0)
    if eligible.sum() < 3:
        raise Problem("周期搜索窗内的频率点不足，请扩大周期范围。")
    baseline = max(1e-7, float(np.median(power[eligible])))
    peaks, _ = find_peaks(power, prominence=baseline * 1.25)
    candidates = [int(i) for i in peaks if eligible[i] and power[i] > baseline * 1.8]
    candidates.sort(key=lambda i: (-power[i], i))
    rows = []
    for index in candidates[:count]:
        local = np.log(np.clip(power[index - 1:index + 2], 1e-8, None))
        curvature = local[0] - 2 * local[1] + local[2]
        offset = np.clip(.5 * (local[0] - local[2]) / curvature, -.5, .5) if curvature < -1e-8 else 0.0
        frequency = (index + offset) / spectrum.samples
        rows.append({
            "axis": spectrum.axis,
            "cycles_per_sample": rounding(frequency, 6),
            "period": rounding(1 / frequency, 4),
            "amplitude": rounding(power[index], 7),
            "amplitude_over_floor": rounding(power[index] / baseline, 3),
            "resolution_cycles_per_sample": rounding(1 / spectrum.samples, 6),
        })
    return rows


def spatial_bands(scene: Scene, second, region: Region, options: dict) -> Report:
    minimum_period = numeric(options.get("min_period"), "最小周期", 2.1, 90, default=5)
    maximum_period = numeric(options.get("max_period"), "最大周期", minimum_period + 1, 480, default=96)
    axis = choice(options.get("axis"), {"horizontal", "vertical"}, "扫描方向", default="vertical")
    plane = make_plane(scene, region, max_side=1024)
    plane.require_coverage(.60)
    profiles = valid_scanline_profiles(plane.rgb, plane.observed, axis == "vertical")
    spectrum = spatial_spectrum(profiles, axis, "luma")
    rows = peak_frequency(spectrum, minimum_period, maximum_period, 5)
    frequency = spectrum.frequency
    power = spectrum.amplitude
    metrics = {
        **plane.info(),
        "axis": axis,
        "scanlines_kept": spectrum.lines,
        "samples_along_scanline": spectrum.samples,
        "cycles_per_sample_step": rounding(1 / spectrum.samples, 7),
        "dominant_period": rows[0]["period"] if rows else None,
    }
    report = Report("空间条纹频谱", "列出一张采集中的重复亮度条纹和可能的谐波候选。", metrics, rows)
    markers = [(row["cycles_per_sample"], f"P{i + 1}") for i, row in enumerate(rows)]
    report.add("band-spectrum.png", chart_png(
        [("Luma", frequency, power)],
        x_label="Cycles per rectified sample", y_label="Relative amplitude",
        x_bounds=(0, min(.5, max(.08, 1.4 / minimum_period))), markers=markers,
    ), "image/png")
    report.table("band-peaks.csv", rows)
    report.image("band-selection.png", masked_rgb(plane.rgb, plane.observed))
    report.warnings.append("该页输出空间周期，单张相机采集不能唯一还原面板刷新频率。")
    if not rows:
        report.warnings.append("搜索窗内没有越过噪声底的突出周期。")
    return report


def lag_support(signal: np.ndarray, pitch: float) -> float:
    position = int(round(pitch))
    if position < 1 or position >= len(signal) * .30:
        return 0.0
    first = signal[:-position]
    second = signal[position:]
    first = first - first.mean()
    second = second - second.mean()
    scale = np.sqrt(np.dot(first, first) * np.dot(second, second))
    return float(np.dot(first, second) / scale) if scale > 1e-7 else 0.0


def subpixel_layout(scene: Scene, second, region: Region, options: dict) -> Report:
    scene.require_role({"macro"}, "请选择次像素近摄采集。")
    direction = choice(options.get("axis"), {"horizontal", "vertical"}, "扫描方向", default="horizontal")
    minimum = numeric(options.get("min_period"), "最短重复宽度", 3, 25, default=4)
    maximum = numeric(options.get("max_period"), "最长重复宽度", minimum + 1, 80, default=32)
    vertical = direction == "vertical"
    x0, y0, x1, y1 = region.pixels(*scene.image.size)
    crop_rgb = scene.image.rgb[y0:y1, x0:x1]
    visible = scene.image.valid[y0:y1, x0:x1]
    if max(visible.shape) > 1800:
        raise Problem("请选择较小的原像素近摄区域，每边不超过 1800 个原像素。")
    profiles = valid_scanline_profiles(crop_rgb, visible, vertical)
    x = np.arange(profiles.shape[1])
    traces, spectrum_series, details = [], [], []
    reference = np.median(profiles, axis=0)
    for label_name, channel_name, index in (("R", "r", 0), ("G", "g", 1), ("B", "b", 2)):
        spec = spatial_spectrum(profiles, direction, channel_name)
        spectrum_series.append((label_name, spec.frequency, spec.amplitude))
        candidates = peak_frequency(spec, minimum, maximum, 4)
        for row in candidates:
            support = lag_support(reference[:, index], row["period"])
            row["channel"] = label_name
            row["lag_correlation"] = rounding(support, 4)
            details.append(row)
        traces.append((label_name, x, reference[:, index]))
    ranked = sorted(details, key=lambda value: (
        -max(value["lag_correlation"], 0.0) * value["amplitude_over_floor"], value["period"]
    ))
    best = ranked[0] if ranked and ranked[0]["lag_correlation"] >= .28 else None
    bands = {f"{r['channel']}_{i}": r["period"] for i, r in enumerate(ranked[:6], start=1)}
    metrics = {
        "source_pixel_window": [x0, y0, x1, y1],
        "source_samples_along_scan": profiles.shape[1],
        "qualified_scanlines": profiles.shape[0],
        "candidate_pitch_source_pixels": best["period"] if best else None,
        "dominant_channel": best["channel"] if best else None,
        "review_periods": bands,
    }
    report = Report("次像素近摄排布", "保留相机原像素的 RGB 周期证据和频谱／自相关候选。", metrics, ranked)
    report.add("micro-profile.png", chart_png(traces, x_label="Source pixel index", y_label="Recorded RGB code"),
               "image/png")
    report.add("micro-spectrum.png", chart_png(
        spectrum_series, x_label="Cycles per source pixel", y_label="Relative amplitude",
        x_bounds=(0, min(.5, 1.5 / minimum)),
    ), "image/png")
    report.image("micro-selection.png", masked_rgb(crop_rgb, visible))
    report.table("micro-periods.csv", ranked)
    report.warnings.append("坐标为相机原像素，未按面板标定插值，不表示面板的实际像素尺寸。")
    if not best:
        report.warnings.append("目前没有频谱和局部相关都明显的周期。")
    return report


@dataclass
class ReflectiveRegion:
    number: int
    count: int
    bounds: tuple[float, float, float, float]
    signal: float
    center: tuple[float, float]


def reflection_regions(values: np.ndarray, valid: np.ndarray, threshold: float, min_pixels: int):
    candidate = (values >= threshold) & valid
    isolated = binary_opening(candidate, np.ones((3, 3)))
    components, count = label(isolated, structure=np.ones((3, 3)))
    width = values.shape[1]
    height = values.shape[0]
    sizes = np.bincount(components.ravel())
    ranking = sorted(range(1, count + 1), key=lambda i: (-sizes[i], i))
    for number, index in enumerate(ranking[:85], start=1):
        if sizes[index] < min_pixels:
            break
        ys, xs = np.where(components == index)
        yield ReflectiveRegion(
            number, int(sizes[index]),
            (xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height),
            float(np.median(values[ys, xs])), (float(xs.mean() / width), float(ys.mean() / height)),
        )


def cover_reflection(scene: Scene, second: Scene, region: Region, options: dict) -> Report:
    floor = numeric(options.get("delta"), "最小变化码值", .004, .30, default=.045)
    support_frac = numeric(options.get("area"), "最小覆盖区域 %", .015, 3.0, default=.10) / 100
    ref, active, overlap = paired_plane(scene, second, region, max_side=850)
    first, current = ref.gray, active.gray
    first_observed = first[overlap]
    current_observed = current[overlap]
    correction = float(np.median(current_observed - first_observed))
    differences = current - first - correction
    residuals = differences[overlap]
    noise = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    support = gaussian_filter(overlap.astype(np.float32), 1.0)
    weight = gaussian_filter(np.where(overlap, differences, 0.0), 1.0)
    signal = np.divide(weight, support, out=np.zeros_like(weight), where=support > .8)
    threshold = max(floor, noise * 4.5)
    min_pixels = max(6, round(overlap.sum() * support_frac))
    regions = list(reflection_regions(signal, overlap & (support > .95), threshold, min_pixels))
    rows = []
    boxes = []
    for entry in regions:
        rows.append({
            "region": entry.number,
            "area_pixels": entry.count,
            "median_brightening": rounding(entry.signal),
            "left": rounding(entry.bounds[0]), "top": rounding(entry.bounds[1]),
            "right": rounding(entry.bounds[2]), "bottom": rounding(entry.bounds[3]),
            "center_x": rounding(entry.center[0]), "center_y": rounding(entry.center[1]),
        })
        boxes.append((entry.bounds, str(entry.number), (255, 219, 131)))
    limit = max(threshold, float(np.percentile(np.abs(residuals), 98)))
    stats = {
        **ref.info(),
        "shared_coverage_pct": rounding(100 * overlap.mean(), 2),
        "panel_wide_offset": rounding(correction),
        "robust_noise_code": rounding(noise),
        "threshold_used": rounding(threshold),
        "region_count": len(rows),
        "difference_scale": [-rounding(limit), rounding(limit)],
    }
    report = Report("盖板反射复核", "对照光照条件，查看剔除全局漂移后持续变亮的局部区域。", stats, rows)
    report.image("reflection-difference.png", overlay_regions(
        heatmap(signal, overlap, -limit, limit, diverging=True), boxes
    ))
    report.table("reflection-regions.csv", rows)
    report.warnings.append("这些区域是相对光照变化的复拍线索，反光、移动或遮挡也可能产生相近表现。")
    return report


def physical_box(first: Point, second: Point, panel: dict):
    dx = abs(second.x - first.x)
    dy = abs(second.y - first.y)
    dims = {
        "panel_box_px": [rounding(dx * panel["native_width"]), rounding(dy * panel["native_height"])],
        "panel_box_mm": None,
    }
    if panel["width_mm"] is not None:
        dims["panel_box_mm"] = [rounding(dx * panel["width_mm"]), rounding(dy * panel["height_mm"])]
    return dims


def compare_measurement(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    endpoints = options.get("points")
    if endpoints:
        first, last = [Point.parse(raw) for raw in sequence(endpoints, "量测端点", minimum=2, maximum=2)]
    else:
        first = Point(region.left, region.top)
        last = Point(region.right, region.bottom)
    if hypot(first.x - last.x, first.y - last.y) < .0001:
        raise Problem("两个量测端点应分开放置。")
    source_w, source_h = scene.image.size
    primary = segment_statistics(first, last, scene.panel, scene.calibration, source_w - 1, source_h - 1)
    capture_image = masked_rgb(scene.image.rgb, scene.image.valid)
    dims = physical_box(first, last, scene.panel)
    metrics = {
        "geometry_revision": scene.revision_id,
        "panel_points": [first.values(), last.values()],
        **primary,
        **dims,
    }
    from PIL import ImageDraw
    draw = ImageDraw.Draw(capture_image)
    points = [(u * (source_w - 1), v * (source_h - 1))
              for u, v in (primary["start_camera"], primary["end_camera"])]
    color = (116, 241, 191)
    draw.line(points, fill=color, width=max(2, source_w // 380))
    radius = max(4, source_w // 170)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    report = Report("面板端点量测", "保留端点在面板及原相机图中的复核定位。", metrics)
    report.image("measurement.png", capture_image)
    if metrics["panel_mm"] is None:
        report.warnings.append("面板未登记有效物理宽高，当前只提供像素单位。")
    if any(not 0 <= value <= 1 for point in (primary["start_camera"], primary["end_camera"]) for value in point):
        report.warnings.append("有端点投影在采集视野之外，物理坐标仅表示输入选区。")
    return report


def configs():
    return {
        "band": {
            "title": "空间条纹频谱",
            "group": "采样与细节",
            "description": "分析一张采集的空间周期及可能的谐波。",
            "run": spatial_bands,
            "options": [axis_option("vertical"), number("min_period", "最小周期", 5, 2.1, 90, .1),
                        number("max_period", "最大周期", 96, 3, 480, 1)],
        },
        "subpixel": {
            "title": "次像素近摄",
            "group": "采样与细节",
            "description": "预览原相机采样区域的 RGB 排布和周期候选。",
            "run": subpixel_layout,
            "coordinate_space": "camera",
            "options": [axis_option(), number("min_period", "最短重复宽度", 4, 3, 25),
                        number("max_period", "最长重复宽度", 32, 4, 80)],
        },
        "reflection": {
            "title": "盖板反射复核",
            "group": "对照证据",
            "description": "从两种光照的同面板采集定位相对变亮的区域。",
            "comparison": "required",
            "run": cover_reflection,
            "options": [number("delta", "最小变化码值", .045, .004, .30, .001),
                        number("area", "最小覆盖区域 %", .10, .015, 3, .005)],
        },
        "measure": {
            "title": "端点量测",
            "group": "采集坐标",
            "description": "使用选区的左上和右下端点量测面板尺寸并在原图标记。",
            "run": compare_measurement,
            "options": [],
        },
    }
