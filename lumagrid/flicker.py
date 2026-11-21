"""Resolve panel flicker frequency bands from rolling-shutter stripes.

A rolling-shutter camera exposes consecutive rows at different instants.  A
periodically modulated light therefore records as a horizontal stripe pattern
whose period is measured in camera rows.  The row sampling also makes every
stripe ambiguous: a light at ``f`` records ``f * row_time`` cycles per row, and
whole cycles disappear into the fold, so the observed stripe fraction ``q``
only implies ``f = (k ± q) / row_time`` for an integer ``k``.

One capture therefore supports a comb of frequency candidates, not a single
frequency.  This module measures the stripe period(s) in two or three captures
taken with different row times, intersects those candidate families inside a
user-given range, and keeps every band the measured captures jointly support.
Captures without a measurable stripe are reported as insufficient evidence and
never used to rule frequencies out.
"""
from dataclasses import dataclass, field
from math import ceil, floor

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import find_peaks

from .inspection import Report, Scene
from .model import Problem, Region, mapping, numeric, plain, required_text, sequence
from .raster import chart_png, luma, masked_rgb, overlay_regions, png_bytes, rounding

MIN_IMAGES = 2
MAX_IMAGES = 3
MIN_ROWS = 24
MIN_COLUMNS = 8
DEFAULT_TOLERANCE_PCT = 2.0
DEFAULT_MIN_STRENGTH = 3.6
DEFAULT_MIN_AMPLITUDE = .0009
DEFAULT_MIN_LAG = .24
DEFAULT_CYCLES_REQUIRED = 3
DEFAULT_MIN_EXCESS_CODE = .002
MAX_STRIPES_PER_IMAGE = 4
MAX_BANDS = 120
STATUS_MEASURED = "measured"
STATUS_INSUFFICIENT = "insufficient"
STATUS_TEXT = {STATUS_MEASURED: "已测到条纹", STATUS_INSUFFICIENT: "证据不足"}


@dataclass(frozen=True)
class FlickerCapture:
    index: int
    scene: Scene
    row_time_us: float
    region: Region

    @property
    def row_time_seconds(self) -> float:
        return self.row_time_us * 1e-6

    @property
    def capture_id(self) -> str:
        return self.scene.capture["id"]

    @property
    def title(self) -> str:
        return self.scene.capture["title"]

    def region_pixels(self) -> list[int]:
        width, height = self.scene.image.size
        return list(self.region.pixels(width, height))


@dataclass(frozen=True)
class Stripe:
    cycles_per_row: float
    period_rows: float
    amplitude_code: float
    peak_over_floor: float
    prominence_over_floor: float
    lag_correlation: float
    cycles_tolerance: float
    repeat_correlation: float = 0.0
    repeat_cycles: int = 0
    excess_code: float = 0.0

    @property
    def is_stripe(self) -> bool:
        """A stripe must repeat over several cycles with a visible amplitude."""
        return (self.lag_correlation >= DEFAULT_MIN_LAG
                and self.repeat_cycles >= 2
                and self.repeat_correlation >= DEFAULT_MIN_LAG
                and self.excess_code >= DEFAULT_MIN_EXCESS_CODE)


@dataclass
class Scan:
    capture: FlickerCapture
    status: str
    note: str
    row_count: int
    column_count: int
    row_coverage: float
    spectral_floor: float
    frequency: np.ndarray
    power: np.ndarray
    profile: np.ndarray
    profile_column: np.ndarray
    stripes: list[tuple[str, Stripe]] = field(default_factory=list)


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def read_settings(data: dict) -> tuple[float, float, float]:
    tolerance = numeric(
        data.get("tolerance_pct"), "条纹周期容差 %", .2, 25, default=DEFAULT_TOLERANCE_PCT
    ) / 100
    strength = numeric(
        data.get("min_strength"), "最低谱峰突出度", 1.5, 60, default=DEFAULT_MIN_STRENGTH
    )
    amplitude = numeric(
        data.get("min_amplitude"), "最低条纹幅度（码值）", 0, .25, default=DEFAULT_MIN_AMPLITUDE
    )
    return tolerance, strength, amplitude


def read_frequency_range(data: dict) -> tuple[float, float]:
    low = numeric(data.get("freq_min"), "频率范围下限（Hz）", .1, 60_000, default=100)
    high = numeric(data.get("freq_max"), "频率范围上限（Hz）", .1, 60_000, default=3_000)
    if high <= low:
        raise Problem("频率范围上限应大于下限。")
    return low, high


def load_selections(store, panel_id: str, raw) -> list[FlickerCapture]:
    entries = sequence(raw, "参与核对的采集", minimum=MIN_IMAGES, maximum=MAX_IMAGES)
    seen_ids, seen_titles = set(), set()
    selected = []
    for index, entry in enumerate(entries, start=1):
        data = mapping(entry, f"第 {index} 张采集参数")
        capture_id = required_text(data.get("capture_id"), f"第 {index} 张采集", 64)
        row_time = numeric(
            data.get("row_time_us"), f"第 {index} 张的逐行读出时间（微秒）", .05, 100_000
        )
        region = Region.parse(data.get("region"))
        if plain(data.get("calibration_id"), f"第 {index} 张的标定版本", 64) not in ("", "identity"):
            raise Problem("条纹核对使用相机原像素行，不接受面板角点标定版本。")
        scene = Scene.load(store, capture_id, "identity")
        if scene.panel["id"] != panel_id:
            raise Problem(f"第 {index} 张采集不属于当前面板。")
        if capture_id in seen_ids:
            raise Problem("同一张采集不能重复参与条纹核对。")
        if scene.capture["title"] in seen_titles:
            raise Problem("参与核对的采集名称彼此相同，无法在结果中区分。")
        seen_ids.add(capture_id)
        seen_titles.add(scene.capture["title"])
        selected.append(FlickerCapture(index, scene, row_time, region))
    return selected


def row_profile(rgb: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Per-row stripe signal with the across-row image gradient removed.

    Rolling-shutter stripes run along the image rows, so every row has the same
    phase but the scene still adds a fixed brightness profile from column to
    column.  Taking the median of each row keeps genuine row-to-row
    modulation.  A slowly varying panel gradient also differs between rows,
    and the cubic trend below cannot remove it over a short selection, so the
    column-common shape is removed first.  The shape comes from the median
    over all rows rather than one row, because a single row also carries its
    own stripe phase.
    """
    values = luma(rgb)
    shared = np.median(values, axis=0)
    signal = values - shared[None, :]
    return np.median(signal[keep], axis=1)


def detrend(profile: np.ndarray) -> np.ndarray:
    """Remove slow shading without eating the stripe periods of interest.

    The shortest searched period is about an eighth of the selection, so a
    low-order polynomial trend is safely below every candidate stripe.
    """
    position = np.arange(len(profile), dtype=np.float64)
    degree = min(3, max(0, len(profile) // 16 - 1))
    if degree <= 0:
        return profile - profile.mean()
    trend = np.polyval(np.polyfit(position, profile, degree), position)
    return profile - trend


def refine_stripe(signal: np.ndarray, cycles: float) -> tuple[float, float]:
    """Least-squares sinusoid search around the FFT peak for a precise period.

    The FFT bin spacing limits the plain peak estimate to ``1 / rows``; the
    stripe frequency enters the frequency intersection divided by the row
    time, so bin-level error would dominate the reported band width.  A small
    grid search of a sinusoid fit gives a much finer period while remaining
    robust because it stays inside the already-detected peak.
    """
    length = len(signal)
    step = 1 / length
    low = max(.5 / length, cycles - 2.5 * step)
    high = min(.999, cycles + 2.5 * step)
    grid = np.linspace(low, high, 41)
    position = np.arange(length, dtype=np.float64)
    base = np.ones_like(position)
    amplitudes = np.empty_like(grid)
    for index, candidate in enumerate(grid):
        angle = 2 * np.pi * candidate * position
        design = np.column_stack((base, np.cos(angle), np.sin(angle)))
        coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
        amplitudes[index] = float(np.hypot(coefficients[1], coefficients[2]))
    best = int(np.argmax(amplitudes))
    refined = float(grid[best])
    amplitude = float(amplitudes[best])
    if 0 < best < len(grid) - 1:
        left, center, right = np.log(np.clip(amplitudes[best - 1:best + 2], 1e-12, None))
        curvature = left - 2 * center + right
        if curvature < -1e-12:
            offset = float(np.clip(.5 * (left - right) / curvature, -.5, .5))
            refined = float(grid[best] + offset * (grid[1] - grid[0]))
            amplitude = float(np.exp(center - .25 * (left - right) * offset))
    return refined, amplitude


def _lag_support(signal: np.ndarray, period: float) -> float:
    position = int(round(period))
    if position < 2 or position >= len(signal) * .7:
        return 0.0
    first = signal[:-position] - signal[:-position].mean()
    second = signal[position:] - signal[position:].mean()
    scale = float(np.sqrt(np.dot(first, first) * np.dot(second, second)))
    return float(np.dot(first, second) / scale) if scale > 1e-9 else 0.0


def _lag_span(signal: np.ndarray, period: float, cycles: int) -> float:
    """Correlation between the profile and itself ``cycles`` stripes away.

    A genuine stripe stays correlated over many periods; a short accidental
    ripple or a single blemish does not.
    """
    position = int(round(period * cycles))
    if position < 2 or position >= len(signal) - 8:
        return 0.0
    first = signal[:-position] - signal[:-position].mean()
    second = signal[position:] - signal[position:].mean()
    scale = float(np.sqrt(np.dot(first, first) * np.dot(second, second)))
    return float(np.dot(first, second) / scale) if scale > 1e-9 else 0.0


def _blank_scan(capture: FlickerCapture, note: str, rows: int, columns: int,
                coverage: float) -> Scan:
    return Scan(
        capture, STATUS_INSUFFICIENT, note, rows, columns, coverage, 0.0,
        np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
    )


def scan_capture(capture: FlickerCapture, tolerance: float, strength: float,
                 amplitude_floor_code: float) -> Scan:
    """Measure horizontal stripe periods along the camera rows of one capture."""
    image = capture.scene.image
    source_width, source_height = image.size
    left, top, right, bottom = capture.region.pixels(source_width, source_height)
    rows = bottom - top
    columns = right - left
    if rows < MIN_ROWS or columns < MIN_COLUMNS:
        raise Problem(
            f"第 {capture.index} 张（{capture.title}）的选区太小："
            f"条纹核对至少需要 {MIN_ROWS} 行和 {MIN_COLUMNS} 列原相机像素，"
            "请在条纹清楚的位置重新框选。"
        )
    rgb = image.rgb[top:bottom, left:right]
    valid = image.valid[top:bottom, left:right]
    keep = valid.mean(axis=1) >= .85
    row_coverage = float(keep.mean())
    keep &= np.all((rgb > .004) & (rgb < .996), axis=2).mean(axis=1) >= .90
    if keep.sum() < MIN_ROWS or row_coverage < .35:
        return _blank_scan(
            capture, "选区内完整、未接近码值极限的相机行不足，无法测量条纹。",
            rows, columns, row_coverage,
        )
    profile = row_profile(rgb, keep)
    profile_column = np.median(luma(rgb[keep]), axis=0)
    length = len(profile)
    fluctuation = detrend(profile)
    window = np.hanning(length)
    gain = 2 / max(float(window.sum()), 1.0)
    power = np.abs(np.fft.rfft(fluctuation * window)) * gain
    frequency = np.fft.rfftfreq(length)
    # Bins below 5 correspond to periods longer than a quarter of the selection;
    # the cubic trend removal cannot separate those from slow panel shading.
    # Beyond one cycle per row the row sampling cannot tell stripes apart, so
    # the search stops just below it.
    start = 5
    stop = min(len(power) - 1, int(floor(length / 1.6)))
    if stop - start < 3:
        return _blank_scan(capture, "选区行数不足以分辨条纹周期。", rows, columns, row_coverage)
    baseline = float(np.median(power[start:stop]))
    if not np.isfinite(baseline) or baseline <= 0:
        return _blank_scan(capture, "频谱没有可用的噪声底。", rows, columns, row_coverage)
    peaks, details = find_peaks(power[start:stop], prominence=baseline * strength)
    positions = [int(item) + start for item in peaks]
    rank = sorted(range(len(positions)), key=lambda i: (-power[positions[i]], positions[i]))
    stripe_floor = max(amplitude_floor_code, 6 * baseline)
    stripes = []
    for rank_index in rank[:MAX_STRIPES_PER_IMAGE]:
        index = positions[rank_index]
        if power[index] < stripe_floor:
            continue
        cycles, amplitude = refine_stripe(fluctuation, index / length)
        period = 1 / cycles if cycles > 0 else float("inf")
        span_cycles = int((length - 9) / max(period, 1e-9))
        repeat_cycles = 0
        repeat_correlation = 0.0
        if span_cycles >= 2:
            repeat_cycles = min(span_cycles, DEFAULT_CYCLES_REQUIRED)
            repeat_correlation = _lag_span(fluctuation, period, repeat_cycles)
        stripes.append((f"P{len(stripes) + 1}", Stripe(
            cycles_per_row=cycles,
            period_rows=period,
            amplitude_code=amplitude,
            peak_over_floor=power[index] / baseline,
            prominence_over_floor=float(details["prominences"][rank_index]) / baseline,
            lag_correlation=_lag_support(fluctuation, period),
            cycles_tolerance=max(cycles * tolerance, .15 / length),
            repeat_correlation=repeat_correlation,
            repeat_cycles=repeat_cycles,
            excess_code=amplitude,
        )))
    if not stripes:
        return _blank_scan(
            capture, "频谱里没有越过噪声底的条纹周期。", rows, columns, row_coverage,
        )
    regular = [entry for entry in stripes if entry[1].is_stripe]
    if not regular:
        best = stripes[0][1]
        return _blank_scan(
            capture,
            f"频谱有抬升，但选区里没有沿行方向稳定重复的条纹（最强者为 "
            f"{best.period_rows:.1f} 行，{best.repeat_cycles} 个周期后的重复性 "
            f"{best.repeat_correlation:.2f}，幅度 {best.excess_code:.4g} 码值）。"
            "请改用条纹更清楚、没有大面积图案的区域。",
            rows, columns, row_coverage,
        )
    stripes = regular
    note = ""
    if max(stripe.period_rows for _, stripe in stripes) >= length / 4.2:
        note = "最强候选的周期接近本工具可测的低频边缘，频率估计很粗。"
    elif stripes[0][1].lag_correlation < DEFAULT_MIN_LAG:
        note = "最强候选沿行的重复性偏弱。"
    return Scan(
        capture, STATUS_MEASURED, note, rows, columns, row_coverage, baseline,
        frequency, power, fluctuation, profile_column, stripes,
    )


def family_intervals(scan: Scan, stripe: Stripe, low_hz: float, high_hz: float) -> list[Interval]:
    """Every frequency interval the measured stripe fraction can stand for."""
    row_time = scan.capture.row_time_seconds
    cycles = stripe.cycles_per_row
    spread = stripe.cycles_tolerance
    cycles_low = max(1e-7, cycles - spread)
    cycles_high = min(1.0, cycles + spread)
    if cycles_high <= cycles_low:
        cycles_high = min(.5, cycles_low + 1e-4)
    first = max(0, floor(low_hz * row_time) - 1)
    last = max(first, ceil(high_hz * row_time) + 1)
    intervals = []
    for fold in range(first, last + 1):
        for sign in (1, -1):
            start = (fold + sign * cycles_low) / row_time
            stop = (fold + sign * cycles_high) / row_time
            start, stop = min(start, stop), max(start, stop)
            start = max(start, low_hz)
            stop = min(stop, high_hz)
            if stop > start:
                intervals.append(Interval(start, stop))
    return merge_intervals(intervals)


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item.low, item.high))
    merged = [ordered[0]]
    for item in ordered[1:]:
        current = merged[-1]
        if item.low <= current.high * (1 + 1e-12):
            if item.high > current.high:
                merged[-1] = Interval(current.low, item.high)
        else:
            merged.append(item)
    return merged


def _scan_union(scan: Scan, low_hz: float, high_hz: float):
    """Disjoint intervals for one capture, tagged by the stripe that covers them."""
    pieces = []
    for order, (name, stripe) in enumerate(scan.stripes):
        for interval in family_intervals(scan, stripe, low_hz, high_hz):
            pieces.append((interval.low, interval.high, order))
    pieces.sort()
    union = []
    for start, stop, order in pieces:
        if union and start <= union[-1][1]:
            previous = union[-1]
            if stop > previous[1]:
                union[-1] = (previous[0], stop, previous[2])
        else:
            union.append((start, stop, order))
    return union


def combine_bands(scans: list[Scan], low_hz: float, high_hz: float,
                  per_group: int = 14):
    """Intersect the measured stripe families and keep every supported band.

    A single fold assignment is rarely unique, so all stripe candidates take
    part.  Instead of enumerating every candidate combination (which explodes
    for a wide range and a large row time), a fine frequency grid is labelled
    once per capture; a run of grid points with the same combination of stripe
    labels is exactly the intersection interval for that combination.

    A band can be explained by several fold assignments; those are kept side by
    side instead of picking one.
    """
    measured = [scan for scan in scans if scan.status == STATUS_MEASURED and scan.stripes]
    if not measured:
        return [], 0
    unions = [_scan_union(scan, low_hz, high_hz) for scan in measured]
    if any(not part for part in unions):
        return [], 0
    narrowest = min(stop - start for part in unions for start, stop, _ in part)
    step = max(1e-4, narrowest / 8)
    grid = int(min(600_000, max(2048, ceil((high_hz - low_hz) / step))))
    centers = low_hz + (np.arange(grid) + .5) * ((high_hz - low_hz) / grid)
    labels = []
    for scan, union in zip(measured, unions):
        starts = np.asarray([start for start, _, _ in union])
        stops = np.asarray([stop for _, stop, _ in union])
        orders = np.asarray([order for _, _, order in union], dtype=np.int64)
        index = np.searchsorted(starts, centers, side="right") - 1
        covered = index >= 0
        safe = np.where(covered, index, 0)
        covered &= centers <= stops[safe]
        labels.append(np.where(covered, orders[safe], -1))
    keys = np.stack(labels, axis=1)
    supported = np.all(keys >= 0, axis=1)
    keys = np.where(supported[:, None], keys, -1)
    change = np.ones(grid, dtype=bool)
    change[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    starts_at = np.flatnonzero(change)
    ends_at = np.append(starts_at[1:], grid)
    explanations = []
    for begin, finish in zip(starts_at, ends_at):
        if not supported[begin]:
            continue
        low = low_hz + begin * (high_hz - low_hz) / grid
        high = low_hz + finish * (high_hz - low_hz) / grid
        matches = []
        for scan, label in zip(measured, keys[begin]):
            name, stripe = scan.stripes[int(label)]
            matches.append((scan, name, stripe))
        explanations.append({"low": low, "high": high, "matches": matches})
    groups: list[dict] = []
    for item in explanations:
        if groups and item["low"] <= groups[-1]["high"] * (1 + 1e-9):
            group = groups[-1]
            group["high"] = max(group["high"], item["high"])
            group["raw"].append(item)
        else:
            groups.append({"low": item["low"], "high": item["high"], "raw": [item]})
    kept = []
    omitted = 0
    for group in groups:
        unique = []
        seen = set()
        for item in group["raw"]:
            signature = (tuple((scan.capture.index, name) for scan, name, _ in item["matches"]))
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(item)
        unique.sort(key=lambda item: (item["low"], item["high"]))
        if len(unique) > per_group:
            omitted += len(unique) - per_group
            unique = unique[:per_group]
        kept.append({"low": group["low"], "high": group["high"], "explanations": unique})
    kept = kept[:MAX_BANDS]
    omitted += max(0, len(groups) - MAX_BANDS)
    return kept, omitted


def fold_relation(frequency: float, scan: Scan, stripe: Stripe):
    """Name the fold index and side that maps this frequency onto the stripe."""
    row_time = scan.capture.row_time_seconds
    cycles = frequency * row_time
    below = floor(cycles + 1e-12)
    above = ceil(cycles - 1e-12)
    spread = stripe.cycles_tolerance * 1.5 + 1e-9
    options = ((below, 1), (above, -1))
    scored = sorted(options, key=lambda item: abs(cycles - (item[0] + item[1] * stripe.cycles_per_row)))
    fold, sign = scored[0]
    if abs(cycles - (fold + sign * stripe.cycles_per_row)) > spread:
        return None
    return fold, sign


def band_rows(scans: list[Scan], groups: list[dict]) -> list[dict]:
    rows = []
    for number, group in enumerate(groups, start=1):
        for position, item in enumerate(group["explanations"], start=1):
            matches = {scan.capture.index: (name, stripe)
                       for scan, name, stripe in item["matches"]}
            row = {
                "band": f"F{number}",
                "band_low_hz": rounding(group["low"], 4),
                "band_high_hz": rounding(group["high"], 4),
                "explanation": f"{position}/{len(group['explanations'])}",
                "explanation_low_hz": rounding(item["low"], 4),
                "explanation_high_hz": rounding(item["high"], 4),
            }
            for scan in scans:
                index = scan.capture.index
                row[f"image_{index}"] = scan.capture.title
                if index not in matches:
                    row[f"stripe_{index}"] = STATUS_TEXT[STATUS_INSUFFICIENT]
                    row[f"period_rows_{index}"] = None
                    row[f"period_ms_{index}"] = None
                    row[f"fold_{index}"] = None
                    continue
                name, stripe = matches[index]
                row[f"stripe_{index}"] = name
                row[f"period_rows_{index}"] = rounding(stripe.period_rows, 3)
                row[f"period_ms_{index}"] = rounding(
                    stripe.period_rows * scan.capture.row_time_us / 1000, 6
                )
                relation = fold_relation((item["low"] + item["high"]) / 2, scan, stripe)
                row[f"fold_{index}"] = (
                    f"k={relation[0]} {'+q' if relation[1] > 0 else '−q'}"
                    if relation else None
                )
            rows.append(row)
    return rows


def evidence_rows(scans: list[Scan]) -> list[dict]:
    rows = []
    for scan in scans:
        strongest = scan.stripes[0][1] if scan.stripes else None
        rows.append({
            "image": scan.capture.index,
            "title": scan.capture.title,
            "row_time_us": scan.capture.row_time_us,
            "region_pixels": scan.capture.region_pixels(),
            "region_normalized": scan.capture.region.values(),
            "region_rows": scan.row_count,
            "region_columns": scan.column_count,
            "row_coverage_pct": rounding(100 * scan.row_coverage, 2),
            "state": STATUS_TEXT[scan.status],
            "note": scan.note,
            "stripe_candidates": len(scan.stripes),
            "strongest_period_rows": rounding(strongest.period_rows, 3) if strongest else None,
            "strongest_amplitude_code": rounding(strongest.amplitude_code) if strongest else None,
            "strongest_lag_correlation": (
                rounding(strongest.lag_correlation, 4) if strongest else None
            ),
        })
    return rows


def band_chart(groups: list[dict], low_hz: float, high_hz: float) -> bytes:
    width, height = 820, 190
    left, right, top, bottom = 62, width - 18, 40, height - 44
    image = Image.new("RGB", (width, height), (19, 29, 40))
    draw = ImageDraw.Draw(image)
    ink, grid = (220, 229, 237), (56, 77, 90)
    draw.text((left, 12), "Commonly supported frequency bands", fill=ink)
    draw.line((left, bottom, right, bottom), fill=grid)
    span = high_hz - low_hz
    for fraction in np.linspace(0, 1, 7):
        value = low_hz + fraction * span
        x = round(left + (right - left) * fraction)
        draw.line((x, top, x, bottom), fill=grid)
        draw.text((x, bottom + 6), f"{value:.4g}", fill=ink, anchor="mt")
    for number, group in enumerate(groups, start=1):
        x0 = round(left + (right - left) * (group["low"] - low_hz) / span)
        x1 = round(left + (right - left) * (group["high"] - low_hz) / span)
        draw.rectangle((x0, top + 5, max(x0 + 2, x1), bottom - 5), fill=(96, 207, 173))
        if x1 - x0 >= 12:
            draw.text(((x0 + x1) / 2, top - 2), f"F{number}", fill=ink, anchor="mb")
    draw.text((right, height - 16), "Hz", fill=ink, anchor="rt")
    return png_bytes(image)


def selection_image(capture: FlickerCapture) -> Image.Image:
    scene = capture.scene
    width, height = scene.image.size
    left, top, right, bottom = capture.region.pixels(width, height)
    box = (left / width, top / height, right / width, bottom / height)
    return overlay_regions(
        masked_rgb(scene.image.rgb, scene.image.valid),
        [(box, f"#{capture.index}", (129, 240, 213))],
    )


def profile_chart(scan: Scan) -> bytes:
    position = np.arange(len(scan.profile))
    return chart_png(
        [("Row median", position, scan.profile), ("Detrended", position, detrend(scan.profile))],
        x_label="Camera row within selection", y_label="Recorded code",
    )


def spectrum_chart(scan: Scan) -> bytes:
    markers = [(stripe.cycles_per_row, name) for name, stripe in scan.stripes]
    top = max((stripe.cycles_per_row for _, stripe in scan.stripes), default=.1)
    return chart_png(
        [("Row profile", scan.frequency, scan.power)],
        x_label="Cycles per camera row", y_label="Relative amplitude",
        x_bounds=(0, min(.5, max(.08, 1.5 * top))), markers=markers,
    )


def flicker_report(store, panel_id: str, data: dict):
    data = mapping(data, "条纹核对参数")
    selections = load_selections(store, panel_id, data.get("captures"))
    low_hz, high_hz = read_frequency_range(data)
    tolerance, strength, amplitude = read_settings(data)
    scans = [scan_capture(capture, tolerance, strength, amplitude) for capture in selections]
    measured = [scan for scan in scans if scan.status == STATUS_MEASURED]
    insufficient = [scan for scan in scans if scan.status == STATUS_INSUFFICIENT]
    groups, omitted = combine_bands(scans, low_hz, high_hz)
    rows = band_rows(scans, groups)
    evidence = evidence_rows(scans)

    metrics = {
        "frequency_range_hz": [low_hz, high_hz],
        "captures_in_check": len(scans),
        "captures_measured": len(measured),
        "captures_insufficient": len(insufficient),
        "common_band_count": len(groups),
        "period_tolerance_pct": rounding(100 * tolerance, 3),
        "row_times_us": [scan.capture.row_time_us for scan in scans],
        "images": [
            {
                "title": scan.capture.title,
                "row_time_us": scan.capture.row_time_us,
                "state": STATUS_TEXT[scan.status],
            }
            for scan in scans
        ],
    }
    summary = (
        "按各张采集的相机行序测量条纹周期，并在给定范围内求这些采集共同支持的频率段。"
        "每一段都列出该频段在各张原图对应的条纹周期与折叠级次。"
    )
    report = Report("多图条纹闪烁频率核对", summary, metrics, rows)
    report.add("flicker-bands.png", band_chart(groups, low_hz, high_hz), "image/png")
    for scan in scans:
        index = scan.capture.index
        report.image(f"selection-{index}.png", selection_image(scan.capture))
        if scan.status == STATUS_MEASURED:
            report.add(f"profile-{index}.png", profile_chart(scan), "image/png")
            report.add(f"spectrum-{index}.png", spectrum_chart(scan), "image/png")
    report.table("flicker-bands.csv", rows)
    report.table("image-evidence.csv", evidence)

    for scan in insufficient:
        report.warnings.append(
            f"第 {scan.capture.index} 张（{scan.capture.title}）证据不足：{scan.note}"
            "这张不参与共同频段求交，也不能用来排除其它图仍支持的频率。"
        )
    if not measured:
        report.warnings.append(
            "所选采集都没有可测量的条纹；本次没有频率结论，也不表示面板不闪烁。"
        )
    elif len(measured) == 1:
        supported = "、".join(f"第 {scan.capture.index} 张" for scan in measured)
        report.warnings.append(
            f"只有 {supported} 测到条纹，所列频段仍是单张的混叠候选，未获得多张共同支持。"
        )
    elif len(measured) < len(scans):
        supported = "、".join(f"第 {scan.capture.index} 张" for scan in measured)
        report.warnings.append(
            f"下列频段仅由 {supported} 共同支持，不是所有所选照片都已确认。"
        )
    if measured and not groups:
        report.warnings.append(
            "在给定范围内没有能同时对上所有已测采集的频段；这不表示面板不闪烁，"
            "请复核频率范围、逐行读出时间和选区。"
        )
    if omitted:
        report.warnings.append(f"共同支持的频段超过 {MAX_BANDS} 段，仅列出最低的 {MAX_BANDS} 段。")
    if groups:
        report.warnings.append(
            f"结果保留 {len(groups)} 个频率段；同一段内可能有多种折叠解释，均已列出，未强行取舍。"
        )
    row_times = [scan.capture.row_time_us for scan in scans]
    if len(row_times) > 1 and np.ptp(row_times) <= .002 * max(row_times):
        report.warnings.append(
            "所选采集的逐行读出时间几乎相同，折叠间隔一致，共同支持的范围可能很宽。"
        )
    report.warnings.append(
        "频率段来自条纹周期在逐行读出下的混叠求交，假设曝光按相机行顺序线性推进；"
        "如需唯一频率，请再用已知频率图案或外部测量确认。"
    )

    snapshot = {
        "tool": "flicker",
        "captures": [
            {
                "capture_id": scan.capture.capture_id,
                "capture_title": scan.capture.title,
                "calibration_id": "identity",
                "row_time_us": scan.capture.row_time_us,
                "region": scan.capture.region.values(),
                "region_pixels": scan.capture.region_pixels(),
                "source_size": list(scan.capture.scene.image.size),
                "source_sha256": scan.capture.scene.capture["original"],
                "state": STATUS_TEXT[scan.status],
            }
            for scan in scans
        ],
        "frequency_range_hz": [low_hz, high_hz],
        "options": {
            "tolerance_pct": 100 * tolerance,
            "min_strength": strength,
            "min_amplitude": amplitude,
        },
        "coordinate_space": "camera",
        "synthetic": scans[0].capture.scene.panel["synthetic"],
    }
    return scans, snapshot, report
