"""Gray reference card sampling and per-channel illumination correction.

The reference card sits next to the panel in camera space (not panel space), so
boxes are normalized camera coordinates drawn on the original captures. Patches
are rejected when they are clipped by the frame, saturated, non-uniform (for
example a specular reflection on the card) or simply too small to trust. When
only one gray level survives on both captures the correction degrades to a
single-anchor gain and panel cells outside that anchor's brightness are marked
as extrapolated rather than presented as settled measurements.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .inspection import Scene
from .model import Problem, choice, mapping, numeric, sequence
from .raster import (
    linear_code, luma, masked_rgb, overlay_regions, resize_report, rounding, side_by_side,
)

LEVELS = {"light": "亮灰档", "dark": "暗灰档"}
SCENE_NAMES = {"primary": "主采集", "comparison": "对照采集"}
MIN_BOX_SPAN = .008
MIN_COVERAGE = .55
MIN_INLIERS = 24
MAX_UNIFORM_CV = .12
SAT_LOW, SAT_HIGH = .006, .994
MIN_LEVEL_GAP = .03
MIN_CHANNEL_SEPARATION = .02
ANCHOR_BAND = .12
GAIN_SOFT_MIN, GAIN_SOFT_MAX = .5, 2.0
STATUS_BRACKETED = "夹在亮暗档之间"
STATUS_NEAR = "靠近参考档"
STATUS_EXTRAPOLATED = "档外外推（暂不能定）"
STATUS_UNREFERENCED = "无灰片参照"


@dataclass(frozen=True)
class GrayBox:
    scene: str
    level: str
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def parse(cls, raw, scene: str) -> "GrayBox":
        data = mapping(raw, "灰片框")
        level = choice(data.get("level"), tuple(LEVELS), "灰片档位", default="light")
        coords = [
            numeric(data.get(name), f"{LEVELS[level]}框 {name}", 0, 1)
            for name in ("left", "top", "right", "bottom")
        ]
        left, top, right, bottom = coords
        if right - left < MIN_BOX_SPAN or bottom - top < MIN_BOX_SPAN:
            raise Problem("灰片框太小，请在相机原图上框出更大的灰片区域。")
        return cls(scene, level, left, top, right, bottom)

    def normalized(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def as_value(self) -> dict:
        return {
            "level": self.level,
            "left": rounding(self.left, 6),
            "top": rounding(self.top, 6),
            "right": rounding(self.right, 6),
            "bottom": rounding(self.bottom, 6),
        }


def parse_gray_boxes(raw) -> list[GrayBox]:
    data = mapping(raw, "灰片参考框") if raw is not None else {}
    boxes: list[GrayBox] = []
    for scene in ("primary", "comparison"):
        items = sequence(data.get(scene, []), f"{SCENE_NAMES[scene]}灰片框", minimum=0, maximum=2)
        seen = set()
        for item in items:
            box = GrayBox.parse(item, scene)
            if box.level in seen:
                raise Problem(f"{SCENE_NAMES[scene]}的同一灰片档位只能框一个区域。")
            seen.add(box.level)
            boxes.append(box)
    return boxes


@dataclass
class PatchSample:
    scene: str
    level: str
    box: GrayBox
    usable: bool
    reason: str | None
    coverage_pct: float
    clipped_fraction: float
    uniform_cv: float | None
    median_linear: np.ndarray | None
    median_luma: float | None
    inlier_count: int

    def row(self) -> dict:
        median = self.median_linear
        return {
            "scene": SCENE_NAMES[self.scene],
            "gray_level": LEVELS[self.level],
            "box_left": rounding(self.box.left, 6),
            "box_top": rounding(self.box.top, 6),
            "box_right": rounding(self.box.right, 6),
            "box_bottom": rounding(self.box.bottom, 6),
            "usable": "是" if self.usable else "否",
            "reason": self.reason or "",
            "coverage_pct": rounding(self.coverage_pct * 100, 2),
            "clipped_pct": rounding(self.clipped_fraction * 100, 2),
            "uniform_cv": rounding(self.uniform_cv, 4) if self.uniform_cv is not None else None,
            "median_linear_r": rounding(median[0]) if median is not None else None,
            "median_linear_g": rounding(median[1]) if median is not None else None,
            "median_linear_b": rounding(median[2]) if median is not None else None,
            "median_linear_luma": rounding(self.median_luma, 6) if self.median_luma is not None else None,
            "inlier_samples": self.inlier_count,
        }


def _pixel_window(shape, box: GrayBox):
    height, width = shape[:2]
    nominal_x0 = int(round(box.left * width))
    nominal_y0 = int(round(box.top * height))
    nominal_x1 = int(round(box.right * width))
    nominal_y1 = int(round(box.bottom * height))
    x0 = max(0, nominal_x0)
    y0 = max(0, nominal_y0)
    x1 = min(width, nominal_x1)
    y1 = min(height, nominal_y1)
    nominal_area = max(1, (nominal_x1 - nominal_x0) * (nominal_y1 - nominal_y0))
    return x0, y0, x1, y1, nominal_area


def read_patch(scene: Scene, box: GrayBox) -> PatchSample:
    height, width = scene.image.rgb.shape[:2]
    x0, y0, x1, y1, nominal_area = _pixel_window((height, width), box)
    if x1 <= x0 or y1 <= y0:
        return PatchSample(box.scene, box.level, box, False, "灰片框完全落在画面之外。",
                           0.0, 1.0, None, None, None, 0)
    valid_window = scene.image.valid[y0:y1, x0:x1]
    observed = int(valid_window.sum())
    coverage = observed / nominal_area
    if observed < MIN_INLIERS:
        return PatchSample(box.scene, box.level, box, False,
                           "框内有效像素过少，可能出画或被裁开。", coverage, 1.0, None, None, None, 0)
    rgb_window = scene.image.rgb[y0:y1, x0:x1][valid_window]
    linear = linear_code(rgb_window)
    unclipped = np.all((rgb_window > SAT_LOW) & (rgb_window < SAT_HIGH), axis=-1)
    clipped_fraction = float(1 - unclipped.mean())
    if unclipped.sum() < MIN_INLIERS:
        return PatchSample(box.scene, box.level, box, False,
                           "灰片大面积顶到码值极限（饱和或死黑），无法定标。",
                           coverage, clipped_fraction, None, None, None, int(unclipped.sum()))
    candidates = linear[unclipped]
    median = np.median(candidates, axis=0)
    mad = np.median(np.abs(candidates - median), axis=0) * 1.4826
    tolerance = np.maximum(4 * mad, .02)
    inlier_mask = np.all(np.abs(candidates - median) <= tolerance, axis=-1)
    inliers = candidates[inlier_mask]
    if len(inliers) < MIN_INLIERS:
        return PatchSample(box.scene, box.level, box, False,
                           "框内反光或明暗差异过大，稳健样本不足。",
                           coverage, clipped_fraction, None, None, None, len(inliers))
    # A specular spot often stays inside the MAD tolerance on a large card; flag
    # it explicitly through the robust outlier tail before judging uniformity.
    tail_fraction = float(1 - inlier_mask.mean())
    if tail_fraction > .08:
        return PatchSample(box.scene, box.level, box, False,
                           f"框内有 {tail_fraction * 100:.0f}% 像素偏离灰片主体，疑似反光或污渍。",
                           coverage, clipped_fraction, None, None, None, len(inliers))
    mean = inliers.mean(axis=0)
    cv = float(np.max(inliers.std(axis=0) / np.maximum(mean, 1e-6)))
    if coverage < MIN_COVERAGE:
        reason = f"仅 {coverage * 100:.0f}% 的框位于有效画面内，出画或裁开部分不参与定标。"
        return PatchSample(box.scene, box.level, box, False, reason,
                           coverage, clipped_fraction, cv, None, None, len(inliers))
    if cv > MAX_UNIFORM_CV:
        return PatchSample(box.scene, box.level, box, False,
                           f"灰片均匀度不足（变异系数 {cv:.3f}），疑似反光或污渍。",
                           coverage, clipped_fraction, cv, None, None, len(inliers))
    return PatchSample(
        box.scene, box.level, box, True, None, coverage, clipped_fraction, cv,
        median, float(luma(median[None, :])[0]), len(inliers),
    )


@dataclass
class GrayCorrection:
    available: bool
    method: str
    reliable_levels: list[str]
    gains: np.ndarray | None
    offsets: np.ndarray | None
    anchor_level: str | None
    target_luma: dict[str, float]
    patches: list[PatchSample]
    boxes: list[GrayBox]
    warnings: list[str]

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        if not self.available:
            return linear_rgb
        return linear_rgb * self.gains + self.offsets

    def cell_status(self, primary_linear_luma: float | None) -> str:
        if not self.available or primary_linear_luma is None:
            return STATUS_UNREFERENCED
        if self.method == "two_level_affine":
            low = self.target_luma["dark"]
            high = self.target_luma["light"]
            if low <= primary_linear_luma <= high:
                return STATUS_BRACKETED
            return STATUS_EXTRAPOLATED
        anchor = self.target_luma[self.anchor_level]
        if abs(primary_linear_luma - anchor) <= ANCHOR_BAND * anchor:
            return STATUS_NEAR
        return STATUS_EXTRAPOLATED


def _find(patches: list[PatchSample], scene: str, level: str) -> PatchSample | None:
    return next((item for item in patches if item.scene == scene and item.level == level), None)


def evaluate(boxes: list[GrayBox], first: Scene, second: Scene) -> GrayCorrection:
    scene_for = {"primary": first, "comparison": second}
    patches = [read_patch(scene_for[box.scene], box) for box in boxes]
    warnings = [
        f"{SCENE_NAMES[patch.scene]}{LEVELS[patch.level]}未采用：{patch.reason}"
        for patch in patches if not patch.usable
    ]
    for scene in ("primary", "comparison"):
        light = _find(patches, scene, "light")
        dark = _find(patches, scene, "dark")
        if light and dark and light.usable and dark.usable:
            if light.median_luma <= dark.median_luma + MIN_LEVEL_GAP:
                raise Problem(
                    f"{SCENE_NAMES[scene]}亮灰档没有明显亮于暗灰档，请核对两个灰片框的档位标注。"
                )
    shared = [
        level for level in ("light", "dark")
        if (primary := _find(patches, "primary", level)) and primary.usable
        and (secondary := _find(patches, "comparison", level)) and secondary.usable
    ]
    if not shared:
        for scene in ("primary", "comparison"):
            present = {box.level for box in boxes if box.scene == scene}
            usable = {p.level for p in patches if p.scene == scene and p.usable}
            missing = present - usable
            if present and not usable:
                warnings.append(f"{SCENE_NAMES[scene]}没有可读取的灰片档位。")
            elif missing:
                warnings.append(f"{SCENE_NAMES[scene]}缺少{ '、'.join(LEVELS[x] for x in sorted(missing)) }框。")
        warnings.append("两张采集中没有共同可靠的灰片档位，本次仅保留未校正结果，网格参照状态暂不能定。")
        return GrayCorrection(False, "unavailable", [], None, None, None, {}, patches, boxes, warnings)

    primary_patch = lambda level: _find(patches, "primary", level)
    secondary_patch = lambda level: _find(patches, "comparison", level)
    target_luma = {level: primary_patch(level).median_luma for level in shared}

    if len(shared) == 2:
        source_light = secondary_patch("light").median_linear
        source_dark = secondary_patch("dark").median_linear
        target_light = primary_patch("light").median_linear
        target_dark = primary_patch("dark").median_linear
        spread = source_light - source_dark
        if np.min(np.abs(spread)) < MIN_CHANNEL_SEPARATION:
            warnings.append("对照图两档灰片在部分通道差距过小，双档仿射退化为主亮灰档的单档增益。")
            level = "light"
            gains = primary_patch(level).median_linear / secondary_patch(level).median_linear
            offsets = np.zeros(3)
            method, anchor = "single_level_gain", level
        else:
            gains = (target_light - target_dark) / spread
            offsets = target_light - gains * source_light
            method, anchor = "two_level_affine", None
    else:
        level = shared[0]
        gains = primary_patch(level).median_linear / secondary_patch(level).median_linear
        offsets = np.zeros(3)
        method, anchor = "single_level_gain", level
        warnings.append(
            f"仅{LEVELS[level]}在两图中可靠读取，按单档增益校正；亮度偏离该档 ±{ANCHOR_BAND * 100:.0f}% "
            "的网格标记为档外外推，暂不能定。"
        )
    if method == "two_level_affine":
        warnings.append("亮暗双档均可靠，按每通道线性仿射扣除补光颜色，档外网格标记为外推，结论仅供参照。")
    if np.any(gains < GAIN_SOFT_MIN) or np.any(gains > GAIN_SOFT_MAX):
        warnings.append(
            "灰片估计的通道增益超出 0.5～2.0 的常见范围，通常是框到了非灰片区域，"
            "本次放弃灰片校正并保留未校正结果。"
        )
        return GrayCorrection(False, "unavailable", [], None, None, None, {}, patches, boxes, warnings)
    return GrayCorrection(True, method, shared, gains, offsets, anchor, target_luma, patches, boxes, warnings)


def patch_rows(correction: GrayCorrection) -> list[dict]:
    return [patch.row() for patch in correction.patches]


def _evidence_thumb(scene: Scene, correction: GrayCorrection, scene_key: str):
    photo = masked_rgb(scene.image.rgb, scene.image.valid)
    regions = []
    for patch in correction.patches:
        if patch.scene != scene_key:
            continue
        tag = "L" if patch.level == "light" else "D"
        if not patch.usable:
            tag += "!"
        color = (132, 244, 215) if patch.usable else (232, 143, 103)
        regions.append((patch.box.normalized(), tag, color))
    decorated = overlay_regions(photo, regions)
    width, height = decorated.size
    target_width = 520
    resized = decorated.resize((target_width, round(height * target_width / width)))
    bar = Image.new("RGB", (resized.width, 26), (21, 35, 46))
    ImageDraw.Draw(bar).text(
        (8, 6), "Primary capture" if scene_key == "primary" else "Comparison capture",
        fill=(220, 229, 237),
    )
    combined = Image.new("RGB", (resized.width, resized.height + 26), (21, 35, 46))
    combined.paste(bar, (0, 0))
    combined.paste(resized, (0, 26))
    return combined


def evidence_image(first: Scene, second: Scene, correction: GrayCorrection):
    combined = side_by_side(
        _evidence_thumb(first, correction, "primary"),
        _evidence_thumb(second, correction, "comparison"),
    )
    return resize_report(combined, max_side=1000)


def correction_metrics(correction: GrayCorrection) -> dict:
    method_labels = {
        "two_level_affine": "亮暗双档仿射",
        "single_level_gain": "单档增益",
        "unavailable": "无可用灰片参照",
    }
    metrics = {
        "graycard_mode": method_labels[correction.method],
        "graycard_reliable_levels": [LEVELS[level] for level in correction.reliable_levels],
        "graycard_reference_luma": {
            LEVELS[level]: rounding(value, 6) for level, value in correction.target_luma.items()
        },
    }
    if correction.available:
        metrics["graycard_gains_rgb"] = [rounding(value, 6) for value in correction.gains]
        metrics["graycard_offsets_rgb"] = [rounding(value, 6) for value in correction.offsets]
    return metrics
