"""Estimate residual translation between two already calibrated capture regions."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift

from .assays import number
from .inspection import Report, Scene, paired_plane
from .model import Problem, Region, numeric
from .raster import chart_png, heatmap, masked_rgb, rounding, side_by_side


@dataclass
class Translation:
    dx: float
    dy: float
    peak: float
    peak_to_sidelobes: float
    next_peak_relative: float
    on_limit: bool
    score_map: np.ndarray
    coord_x: np.ndarray
    coord_y: np.ndarray


def tapered_signal(gray: np.ndarray, valid: np.ndarray, sigma: float):
    support = gaussian_filter(valid.astype(np.float32), sigma)
    baseline = gaussian_filter(np.where(valid, gray, 0.0), sigma)
    smooth = np.divide(baseline, support, out=np.zeros_like(baseline), where=support > .25)
    defined = (support > .85) & valid
    response = np.where(defined, gray - smooth, 0.0)
    rms = float(np.sqrt(np.mean(response[defined] ** 2))) if defined.any() else 0.0
    if rms < .0022:
        raise Problem("有效选区缺少可用于判断相对位移的纹理。")
    normalized = response / rms
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    return normalized * window


def axis_refine(part: np.ndarray, index: int):
    if not 0 < index < len(part) - 1:
        return 0.0
    left, center, right = part[index - 1:index + 2]
    curvature = left - 2.0 * center + right
    if curvature > -1e-10:
        return 0.0
    return float(np.clip(.5 * (left - right) / curvature, -.5, .5))


def match_translation(first: np.ndarray, compared: np.ndarray, radius: int, power: float):
    h, w = first.shape
    fft_height = 1 << (2 * h - 1).bit_length()
    fft_width = 1 << (2 * w - 1).bit_length()
    grid_shape = (fft_height, fft_width)
    a = np.fft.rfft2(first, s=grid_shape, axes=(0, 1))
    b = np.fft.rfft2(compared, s=grid_shape, axes=(0, 1))
    product = a * np.conj(b)
    magnitude = np.abs(product)
    keep = magnitude > max(1e-9, magnitude.max() * 1e-5)
    normalized = np.divide(product, np.power(magnitude, power, where=keep, out=np.ones_like(magnitude)),
                           out=np.zeros_like(product), where=keep)
    raw_correlation = np.fft.irfft2(normalized, s=grid_shape, axes=(0, 1))
    raw_y = np.arange(-min(radius, h // 4), min(radius, h // 4) + 1)
    raw_x = np.arange(-min(radius, w // 4), min(radius, w // 4) + 1)
    results = raw_correlation[np.ix_(raw_y % fft_height, raw_x % fft_width)]
    max_index = int(np.argmax(results))
    iy, ix = np.unravel_index(max_index, results.shape)
    peak = float(results[iy, ix])
    sidelobes = np.ones_like(results, dtype=bool)
    sidelobes[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = False
    sample = results[sidelobes]
    center = float(np.median(sample))
    mad = max(1e-10, float(1.4826 * np.median(np.abs(sample - center))))
    ratio = (float(sample.max()) - center) / max(1e-10, peak - center)
    boundary = ix in {0, results.shape[1] - 1} or iy in {0, results.shape[0] - 1}
    return Translation(
        float(raw_x[ix]) + axis_refine(results[iy], ix),
        float(raw_y[iy]) + axis_refine(results[:, ix], iy),
        peak, (peak - center) / mad, ratio, boundary, results, raw_x, raw_y,
    )


def photometric_residual(primary: np.ndarray, compared: np.ndarray, seen: np.ndarray):
    if seen.sum() < 140:
        return None, None
    gain = float(np.median(primary[seen] - compared[seen]))
    mismatch = primary - compared - gain
    return float(np.median(np.abs(mismatch[seen]))), mismatch


def displacement(scene: Scene, second: Scene, region: Region, options: dict) -> Report:
    radius_panel = numeric(options.get("radius"), "搜索半径（面板像素）", 4, 180, default=32)
    weighting = numeric(options.get("flatten"), "频谱归一强度", .15, 1, default=.80)
    sigma = numeric(options.get("sigma"), "去背景尺度", 1.5, 24, default=6)
    first, compared, overlap = paired_plane(scene, second, region, max_side=740)
    height, width = first.shape
    scale_x = scene.scale[0] * (region.right - region.left) / width
    scale_y = scene.scale[1] * (region.bottom - region.top) / height
    radius = int(np.clip(round(radius_panel / np.median([scale_x, scale_y])), 4, 180))
    if min(height, width) < 38 or overlap.mean() < .47:
        raise Problem("残余位移核对需要相互重叠且可采样的参考纹理。")
    first_gray = first.gray
    compared_gray = compared.gray
    first_signal = tapered_signal(first_gray, overlap, sigma)
    compared_signal = tapered_signal(compared_gray, overlap, sigma)
    model = match_translation(first_signal, compared_signal, radius, weighting)
    vector = (model.dy, model.dx)
    compensated = shift(compared_gray, vector, order=1, mode="constant", cval=0)
    seen_after = shift(compared.observed.astype(np.float32), vector, order=1, mode="constant", cval=0) > .999
    check = overlap & seen_after
    before, before_residual = photometric_residual(first_gray, compared_gray, check)
    after, after_residual = photometric_residual(first_gray, compensated, check)
    if before is None or after is None:
        raise Problem("位移预览留下的共同像素不足以做差，请复核源图和选区。")
    reliable = model.peak_to_sidelobes >= 7 and model.next_peak_relative < .86 and not model.on_limit
    metrics = {
        **first.info(),
        "shifted_common_coverage_pct": rounding(check.mean() * 100, 3),
        "working_image_displacement": [rounding(model.dx, 4), rounding(model.dy, 4)],
        "requested_panel_search_radius_px": radius_panel,
        "measured_panel_displacement_px": [rounding(model.dx * scale_x, 3), rounding(model.dy * scale_y, 3)],
        "correlation_peak": rounding(model.peak),
        "robust_peak_ratio": rounding(model.peak_to_sidelobes, 3),
        "second_peak_fraction": rounding(model.next_peak_relative, 3),
        "median_difference_before": rounding(before),
        "median_difference_after": rounding(after),
        "unique_shift_supported": reliable,
    }
    peak_row = int(np.argmin(np.abs(model.coord_y - model.dy)))
    peak_column = int(np.argmin(np.abs(model.coord_x - model.dx)))
    radius_error = max(.01, float(np.percentile(np.abs(before_residual[check]), 94)))
    report = Report("采集残余位移", "对已展平的同一选区做有界位移预览，核对两张源采集的相对位置。", metrics)
    report.add("offset-correlation.png", chart_png(
        [("Horizontal", model.coord_x, model.score_map[peak_row]),
         ("Vertical", model.coord_y, model.score_map[:, peak_column])],
        x_label="Shift in working samples", y_label="Windowed correlation",
    ), "image/png")
    report.image("registration-before-after.png", side_by_side(
        heatmap(before_residual, check, -radius_error, radius_error, diverging=True),
        heatmap(after_residual, check, -radius_error, radius_error, diverging=True),
    ))
    report.table("offset-correlation-sections.csv", [
        {"axis": axis, "shift_samples": int(shift_by), "score": rounding(score)}
        for axis, coords, values in (
            ("x", model.coord_x, model.score_map[peak_row]),
            ("y", model.coord_y, model.score_map[:, peak_column]),
        )
        for shift_by, score in zip(coords, values, strict=True)
    ])
    if not reliable:
        report.warnings.append("响应峰不唯一、峰支撑弱或落在搜索范围边缘，此位移仅用于复核。")
    if after > before * 1.10:
        report.warnings.append("按当前位移试算后绝对差反而增大；应核对重复纹理和相机条件。")
    report.warnings.append("左图为原差值，右图为应用位移后的试算。当前调用不调整已登记标定。")
    return report


def configs():
    return {
        "drift": {
            "title": "采集残余位移",
            "group": "采集坐标",
            "description": "在已有标定的共同选区查找残留的平移并提供可信度线索。",
            "comparison": "required",
            "run": displacement,
            "options": [
                number("radius", "搜索半径（面板像素）", 32, 4, 180),
                number("flatten", "频谱归一强度", .80, .15, 1, .05),
                number("sigma", "去背景尺度", 6, 1.5, 24, .5),
            ],
        },
    }
