"""Camera-domain repeatability and dark/flat diagnostics with explicit source support."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from .assays import number, partition, tick
from .inspection import Report, Scene
from .model import Problem, Region, boolean, integral, numeric
from .raster import chart_png, code_statistics, heatmap, luma, masked_rgb, overlay_regions, rounding
from .sensor import NativeWindow


@dataclass
class Pair:
    first: np.ndarray
    later: np.ndarray
    visible: np.ndarray
    window: NativeWindow
    known_camera: bool
    known_exposure: bool
    valid_ratio: float


def prepare_pair(first: Scene, later: Scene, region: Region, avoid_clipping: bool) -> Pair:
    first_meta = first.capture["exposure"]
    later_meta = later.capture["exposure"]
    if first.capture["role"] != later.capture["role"]:
        raise Problem("重复测量需要相同采集角色。")
    if first.capture["role"] not in {"dark", "flat", "uniform", "white", "black"}:
        raise Problem("应选择两张同设置且内容稳定的暗场、平场或均匀图。")
    first_camera, later_camera = first_meta.get("camera"), later_meta.get("camera")
    if first_camera and later_camera and first_camera != later_camera:
        raise Problem("两张重复采集登记了不同相机。")
    if first.image.size != later.image.size:
        raise Problem("重复采集需要相同的相机栅格尺寸。")
    confirmed_exposure = True
    for key in ("shutter_ms", "iso"):
        a, b = first_meta.get(key), later_meta.get(key)
        if a is None or b is None:
            confirmed_exposure = False
        elif abs(a - b) / max(a, b) > .012:
            raise Problem("重复采集的快门或 ISO 不一致，会改变噪声基准。")
    window = NativeWindow.select(first, region, min_width=30, limit=1_750_000)
    primary = window.extract(first.image.rgb)
    secondary = window.extract(later.image.rgb)
    keep = window.extract(first.image.valid) & window.extract(later.image.valid)
    common = keep.mean()
    if avoid_clipping:
        if first.capture["role"] == "dark":
            keep &= (primary.max(axis=-1) < .994) & (secondary.max(axis=-1) < .994)
        else:
            for rgb in (primary, secondary):
                keep &= (rgb.min(axis=-1) > .006) & (rgb.max(axis=-1) < .994)
    if keep.sum() < 180 or keep.mean() < .18:
        raise Problem("重复采集的共同有效像素不足。")
    return Pair(primary, secondary, keep, window, bool(first_camera and later_camera), confirmed_exposure, float(common))


def weighted_smooth(data: np.ndarray, mask: np.ndarray, sigma: float):
    support = gaussian_filter(mask.astype(np.float32), sigma)
    summed = gaussian_filter(np.where(mask, data, 0), sigma)
    filtered = np.divide(summed, support, where=support > .2, out=np.zeros_like(summed))
    return filtered, support


def camera_repeatability(scene: Scene, second: Scene, region: Region, options: dict) -> Report:
    use_rgb = boolean(options.get("rgb"), "分别汇总 RGB", default=True)
    clips = boolean(options.get("avoid_clip"), "排除明显饱和", default=True)
    smooth = numeric(options.get("background_sigma"), "空间平滑半径", 1.5, 35, default=7)
    grid = integral(options.get("grid"), "网格边数", 2, 12, default=5)
    paired = prepare_pair(scene, second, region, clips)
    first_gray = luma(paired.first)
    later_gray = luma(paired.later)
    pooled = (first_gray + later_gray) * .5
    differential = (later_gray - first_gray) / np.sqrt(2)
    smooth_pool, smooth_support = weighted_smooth(pooled, paired.visible, smooth)
    static_residual = pooled - smooth_pool
    main_valid = paired.visible & (smooth_support >= .72)
    if main_valid.sum() < 120:
        raise Problem("当前平滑半径下无法得到可靠的局部背景。")
    global_bias = float(np.median(differential[main_valid]))
    spread = differential - global_bias
    levels = np.column_stack((pooled[main_valid], spread[main_valid]))
    texture = np.column_stack((static_residual[main_valid], spread[main_valid]))
    if len(levels) > 140_000:
        step = int(np.ceil(len(levels) / 140_000))
        levels = levels[::step]
        texture = texture[::step]
    if np.ptp(levels[:, 0]) > 1e-6 and np.ptp(np.abs(levels[:, 1])) > 1e-6:
        correlation = spearmanr(levels[:, 0], np.abs(levels[:, 1])).statistic
    else:
        correlation = None
    rows = []
    channels = [("Y", first_gray, later_gray)]
    if use_rgb:
        channels.extend((name, paired.first[..., i], paired.later[..., i]) for i, name in enumerate("RGB"))
    for title, a, b in channels:
        delta = (b - a)[main_valid] / np.sqrt(2)
        double = b[main_valid] - a[main_valid]
        offset = np.median(double)
        dr = delta - np.median(delta)
        rows.append({
            "channel": title,
            "samples": int(main_valid.sum()),
            "median_shift_code": rounding(offset),
            "temporal_noise_stddev": rounding(np.std(dr)),
            "temporal_noise_mad": rounding(1.4826 * np.median(np.abs(dr))),
            "p95_absolute_difference": rounding(np.percentile(np.abs(double - offset), 95)),
        })
    zones = []
    frames = []
    grid_data = np.full(spread.shape, np.nan)
    coverage_boxes = []
    for cell in partition(spread.shape[1], spread.shape[0], grid, grid):
        coverage = cell.values(main_valid)
        fraction = float(coverage.mean())
        if fraction < .50:
            zones.append({
                "row": cell.row + 1, "column": cell.column + 1,
                "coverage_pct": rounding(100 * fraction, 2), "temporal_mad": None,
                "local_texture_rms": None, "relative_median_level": None,
            })
            coverage_boxes.append((cell.box, str(len(zones)), (240, 163, 94)))
            continue
        absolute = cell.values(spread)[coverage]
        background = cell.values(static_residual)[coverage]
        brightness = cell.values(pooled)[coverage]
        med = np.median(absolute)
        variability = 1.4826 * np.median(np.abs(absolute - med))
        grid_data[cell.y, cell.x] = variability
        zones.append({
            "row": cell.row + 1, "column": cell.column + 1, "coverage_pct": rounding(100 * fraction, 2),
            "temporal_mad": rounding(variability), "local_texture_rms": rounding(np.std(background)),
            "relative_median_level": rounding(np.median(brightness)),
        })
        coverage_boxes.append((cell.box, str(len(zones)), (194, 245, 214)))
    amplitude = max(.002, float(np.percentile(np.abs(spread[main_valid]), 97)))
    max_zone = max(.003, float(np.nanmax(grid_data)) if np.any(np.isfinite(grid_data)) else .003)
    texture_amplitude = max(.005, float(np.percentile(np.abs(static_residual[main_valid]), 96)))
    metrics = {
        "source_window_px": paired.window.coords(),
        "joint_camera_coverage_pct": rounding(paired.valid_ratio * 100, 2),
        "inferred_qualified_pct": rounding(main_valid.mean() * 100, 2),
        "same_camera_documented": paired.known_camera,
        "same_exposure_documented": paired.known_exposure,
        "relative_median_brightness": rounding(np.median(pooled[main_valid])),
        "localized_fixed_texture_rms": rounding(np.std(static_residual[main_valid])),
        "temporal_brightness_correlation": rounding(correlation) if correlation is not None else None,
        "temporal_display_scale": [-rounding(amplitude), rounding(amplitude)],
    }
    hist_data = [spread[main_valid], static_residual[main_valid]]
    max_clip = max(amplitude, texture_amplitude) * 1.8
    for name, values in zip(("Temporal", "Fixed"), hist_data, strict=True):
        allocation, edges = np.histogram(values, bins=96, range=(-max_clip, max_clip))
        distribution = allocation * (100 / max(1, allocation.sum()))
        frames.append((name, (edges[:-1] + edges[1:]) * .5, distribution))
    report = Report("重复采集稳定度", "利用同一相机设置下的两帧估计短期噪声和公共纹理。", metrics, rows)
    report.image("temporal-residual.png", heatmap(spread, main_valid, -amplitude, amplitude, diverging=True))
    report.image("local-noise-grid.png", overlay_regions(
        heatmap(grid_data, main_valid & np.isfinite(grid_data), 0, max_zone), coverage_boxes,
    ))
    report.image("shared-texture.png", heatmap(
        static_residual, main_valid, -texture_amplitude, texture_amplitude, diverging=True,
    ))
    report.add("noise-distributions.png", chart_png(frames, x_label="Difference in code units",
                                                      y_label="Percent in plotted interior"), "image/png")
    report.table("noise-channels.csv", rows)
    report.table("noise-zones.csv", zones)
    if not paired.known_camera or not paired.known_exposure:
        report.warnings.append("至少一张采集缺少相机或曝光记录。请复核两张源图是否真为同设置。")
    report.warnings.append("共同纹理同时包含被摄物和固定像素图样；两帧估计不等于传感器噪声标定。")
    return report


def configs():
    return {
        "repeat_noise": {
            "title": "重复采集稳定度",
            "group": "采集质量",
            "description": "对同一设置下的重复采集比较相机原像素短期噪声和公共细节。",
            "run": camera_repeatability,
            "coordinate_space": "camera",
            "comparison": "required",
            "options": [
                number("background_sigma", "空间背景半径", 7, 1.5, 35, .5),
                number("grid", "网格边数", 5, 2, 12),
                tick("rgb", "分别汇总 RGB", True),
                tick("avoid_clip", "排除明显饱和", True),
            ],
        },
    }
