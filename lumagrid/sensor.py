"""Camera-native sampling diagnostics that deliberately avoid panel interpolation."""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.ndimage import gaussian_filter, label, median_filter

from .assays import number, partition, tick
from .inspection import Report, Scene
from .model import Problem, Region, boolean, numeric
from .raster import chart_png, code_statistics, heatmap, luma, masked_rgb, overlay_regions, rounding


@dataclass(frozen=True)
class NativeWindow:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def select(cls, scene: Scene, region: Region, *, min_width=12, limit=2_500_000):
        source_w, source_h = scene.image.size
        left, top, right, bottom = region.pixels(source_w, source_h)
        if min(right - left, bottom - top) < min_width:
            raise Problem("相机原像素选区过小，请重新框选。")
        if (right - left) * (bottom - top) > limit:
            raise Problem("选区超过当前工具的原像素窗口上限，请缩小画面区域。")
        return cls(left, top, right, bottom)

    def extract(self, values: np.ndarray):
        return values[self.top:self.bottom, self.left:self.right]

    def coords(self):
        return [self.left, self.top, self.right, self.bottom]

    def x_source(self, value: float):
        return self.left + value

    def y_source(self, value: float):
        return self.top + value


def histogram_quantiles(values: np.ndarray, low: float, high: float, bins: int):
    histogram, edges = np.histogram(values, bins=bins, range=(0, 1))
    percent = histogram * (100 / max(1, len(values)))
    dark = np.count_nonzero(values <= low)
    blown = np.count_nonzero(values >= high)
    normalized_x = (edges[1:] + edges[:-1]) / 2
    return normalized_x, percent, dark, blown


def exposure_headroom(scene: Scene, second, region: Region, options: dict) -> Report:
    image = scene.image
    high = numeric(options.get("high"), "高码值提醒线", .70, 1, default=.988)
    low = numeric(options.get("low"), "低码值提醒线", 0, .20, default=.012)
    bins = int(numeric(options.get("bins"), "频数区间数", 32, 256, default=128))
    window = NativeWindow.select(scene, region)
    rgb = window.extract(image.rgb)
    valid = window.extract(image.valid)
    if valid.sum() < 24:
        raise Problem("只有少量像素有效，不能估算曝光余量。")
    clipped_dark = np.any(rgb <= low, axis=-1)
    clipped_bright = np.any(rgb >= high, axis=-1)
    notes = []
    series = []
    stats = []
    valid_count = int(valid.sum())
    grid_rows = []
    zones = np.array([low, high], dtype=np.float32)
    for label_name, channel in (
        ("R", rgb[..., 0]), ("G", rgb[..., 1]), ("B", rgb[..., 2]), ("Y", luma(rgb)),
    ):
        values = channel[valid]
        xs, percent, dark, bright = histogram_quantiles(values, low, high, bins)
        stats.append({
            "channel": label_name,
            **code_statistics(values),
            "near_dark_pct": rounding(100 * dark / valid_count, 4),
            "near_high_pct": rounding(100 * bright / valid_count, 4),
            "p99_headroom": rounding(1.0 - np.percentile(values, 99)),
        })
        series.append((label_name, xs, percent))
        if bright / valid_count > .15 and label_name != "Y":
            notes.append(f"{label_name} 通道高端堆积，需确认曝光是否保留亮区细节。")
    for cell in partition(rgb.shape[1], rgb.shape[0], 4, 4):
        keep = cell.values(valid)
        hits = np.count_nonzero(keep)
        grid_rows.append({
            "column": cell.column + 1,
            "row": cell.row + 1,
            "valid_count": int(hits),
            "any_channel_high_pct": rounding(100 * (cell.values(clipped_bright) & keep).sum() / hits)
            if hits else None,
            "any_channel_low_pct": rounding(100 * (cell.values(clipped_dark) & keep).sum() / hits)
            if hits else None,
        })
    display = rgb.copy()
    display[clipped_dark] = [.22, .43, .76]
    display[clipped_bright] = [.94, .45, .18]
    display[clipped_dark & clipped_bright] = [.77, .43, .74]
    report = Report(
        "相机曝光余量", "按原码值检查通道饱和、黑端压缩和亮区保留。",
        {
            "source_crop_pixels": window.coords(),
            "recorded_bit_depth": image.bits,
            "valid_source_pixels": valid_count,
            "warning_levels": zones.tolist(),
            "any_channel_high_pct": rounding(100 * (clipped_bright & valid).sum() / valid_count, 3),
            "any_channel_low_pct": rounding(100 * (clipped_dark & valid).sum() / valid_count, 3),
        },
        stats,
        notes,
    )
    report.add("native-code-histogram.png", chart_png(
        series, x_label="Recorded camera code", y_label="Percent of selected pixels",
        x_bounds=(0, 1), markers=[(low, "Low"), (high, "High")],
    ), "image/png")
    report.image("camera-clipping.png", masked_rgb(display, valid))
    report.table("exposure-channel.csv", stats)
    report.table("exposure-zones.csv", grid_rows)
    report.warnings.append("橙色为任意通道高码，蓝色为低码，淡紫色为同时含高低码。")
    return report


def robust_local_detector(gray: np.ndarray, visible: np.ndarray, strength: float, floor: float):
    padded = np.where(visible, gray, float(np.median(gray[visible])))
    middle = median_filter(padded, size=5, mode="mirror")
    residual = padded - middle
    accepted_neighbors = gaussian_filter(visible.astype(np.float32), 1.2) > .95
    if accepted_neighbors.sum() < 64:
        raise Problem("该选区没有足够的连续可见像素用于相机缺陷筛查。")
    reference = residual[accepted_neighbors & visible]
    med = float(np.median(reference))
    noise = max(1.0 / 1024, 1.4826 * float(np.median(np.abs(reference - med))))
    admissible = visible & accepted_neighbors
    admissible[:2] = False
    admissible[-2:] = False
    admissible[:, :2] = False
    admissible[:, -2:] = False
    threshold = max(noise * strength, floor)
    bright = (residual - med) > threshold
    return admissible & bright, residual, admissible, noise, threshold


def point_clusters(mask: np.ndarray, largest: int):
    labels, count = label(mask, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    allowed = [i for i in range(1, count + 1) if counts[i] <= largest]
    if len(allowed) > 2500:
        raise Problem("超过 2500 个候选，建议缩小选区或提高判定阈值。")
    result = []
    from scipy.ndimage import find_objects
    bounding = find_objects(labels, max_label=count)
    for index in allowed:
        sl = bounding[index - 1]
        cropped = labels[sl] == index
        ys, xs = np.where(cropped)
        result.append((ys + sl[0].start, xs + sl[1].start))
    return result


def camera_hot_pixels(scene: Scene, second: Scene | None, region: Region, options: dict):
    scene.require_role({"dark"}, "请从镜头暗场检查相机固定亮点。")
    scale = numeric(options.get("sigma"), "噪声阈值倍数", 2, 20, default=8)
    floor = numeric(options.get("floor"), "最低异常码差", .002, .18, default=.027)
    limit_size = int(numeric(options.get("limit_size"), "单候选像素上限", 1, 14, default=5))
    persistence = boolean(options.get("same_place"), "若有参考只留重复亮点", default=True)
    a = NativeWindow.select(scene, region, min_width=24, limit=2_000_000)
    main = a.extract(scene.image.rgb)
    alpha = a.extract(scene.image.valid)
    if alpha.sum() < 96:
        raise Problem("有效暗场像素不足。")
    candidates, residual, known, noise, threshold = robust_local_detector(luma(main), alpha, scale, floor)
    second_eligible = None
    second_mask = None
    if second is not None:
        second.require_role({"dark"}, "参考图也应选择相同相机的镜头暗场。")
        if scene.image.size != second.image.size:
            raise Problem("暗场参照必须保持相同的相机像素尺寸。")
        label_a = scene.capture["exposure"].get("camera")
        label_b = second.capture["exposure"].get("camera")
        if label_a and label_b and label_a != label_b:
            raise Problem("暗场主图和对照图使用了不同相机。")
        second_rgb = a.extract(second.image.rgb)
        second_alpha = a.extract(second.image.valid)
        if second_alpha.sum() < 96:
            raise Problem("有效的对照暗场像素不足。")
        second_mask, _, second_eligible, _, _ = robust_local_detector(luma(second_rgb), second_alpha, scale, floor)
    rows = []
    locations = []
    height, width = known.shape
    for ys, xs in point_clusters(candidates, limit_size):
        peak = int(np.argmax(residual[ys, xs]))
        peak_x, peak_y = int(xs[peak]), int(ys[peak])
        reference_known = bool(second_eligible[peak_y, peak_x]) if second_eligible is not None else None
        repeated = bool(second_mask[peak_y, peak_x]) if reference_known else None
        if second is not None and persistence and repeated is not True:
            continue
        rows.append({
            "camera_px_x": a.x_source(peak_x), "camera_px_y": a.y_source(peak_y),
            "candidate_area_pixels": len(xs),
            "excess_code": rounding(residual[peak_y, peak_x]),
            "reference_visible": reference_known,
            "reference_repeat": repeated,
        })
        pad = 2
        outer = ((xs.min() - pad) / width, (ys.min() - pad) / height,
                 (xs.max() + 1 + pad) / width, (ys.max() + 1 + pad) / height)
        locations.append((outer, "", (254, 207, 94) if repeated is not True else (129, 240, 206)))
    report = Report(
        "相机固定亮点检查", "在暗场原像素上定位局部孤立亮点与相机固定位置证据。",
        {
            "camera_pixels": a.coords(),
            "robust_noise": rounding(noise),
            "excess_threshold": rounding(threshold),
            "candidates": len(rows),
            "repeat_at_same_pixel": sum(row["reference_repeat"] is True for row in rows),
            "valid_source_pct": rounding(100 * known.mean(), 3),
        },
        rows[:100],
    )
    report.image("native-hot-pixels.png", overlay_regions(masked_rgb(main, alpha), locations))
    report.table("camera-hot-pixels.csv", rows)
    report.warnings.append("该检查使用相机坐标。单张暗场的亮点可能来自瞬间噪声，需在同设置下复拍确认。")
    return report


def configs():
    return {
        "headroom": {
            "title": "相机曝光余量",
            "group": "采集质量",
            "description": "检查原像素码值饱和、各通道曝光余量和出现位置。",
            "run": exposure_headroom,
            "coordinate_space": "camera",
            "options": [number("low", "低码值提醒线", .012, 0, .20, .001),
                        number("high", "高码值提醒线", .988, .70, 1, .001),
                        number("bins", "频数区间数", 128, 32, 256)],
        },
        "hot_pixels": {
            "title": "相机固定亮点",
            "group": "采集质量",
            "description": "筛查镜头暗场的原像素孤立异常；同设置复拍可检查重复性。",
            "run": camera_hot_pixels,
            "comparison": "optional",
            "coordinate_space": "camera",
            "options": [number("sigma", "噪声阈值倍数", 8, 2, 20, .5),
                        number("floor", "最低异常码差", .027, .002, .18, .001),
                        number("limit_size", "候选像素上限", 5, 1, 14),
                        tick("same_place", "若有对照只留重复亮点", True)],
        },
    }
