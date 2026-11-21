"""Narrow, reproducible steps along a calibrated gray ramp.

The assay scans parallel profiles in the direction of the brightness gradient.
Transitions that are narrow and repeat on neighbouring profiles become a step
candidate; wider transitions and local bright/dark features are kept aside so a
smooth ramp or an isolated spot is not reported as a banding edge.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .assays import number
from .inspection import Plane, Report, Scene, make_plane
from .model import Problem, Region, choice, integral, numeric, plain, sequence
from .raster import chart_png, heatmap, masked_rgb, overlay_regions, rounding


MAX_CANDIDATES = 60
MAX_PROFILE_CHARTS = 8
NEIGHBOUR_LINES = 2
BRIGHT_INK = (250, 198, 96)
DARK_INK = (126, 184, 252)
MUTED_INK = (150, 160, 170)


@dataclass
class Hit:
    line: int
    position: float
    step: float
    raw_width: float | None
    local: bool


@dataclass
class Scan:
    gray: np.ndarray
    observed: np.ndarray
    scan_coord: np.ndarray
    line_coord: np.ndarray
    along_native: int
    across_native: int
    along_span: float
    across_span: float
    axis: str
    region: Region

    @property
    def lines(self) -> int:
        return self.gray.shape[0]

    @property
    def length(self) -> int:
        return self.gray.shape[1]

    @property
    def scan_step(self) -> float:
        """Panel pixels per working sample along the scan direction."""
        return self.along_native * self.along_span / self.length

    @property
    def line_step(self) -> float:
        """Panel pixels per working profile across the scan direction."""
        return self.across_native * self.across_span / self.lines

    def panel_point(self, line_index: float, position: float) -> tuple[float, float]:
        along = self.scan_coord[0] + (position + .5) / self.length * self.along_span
        across = self.line_coord[0] + (line_index + .5) / self.lines * self.across_span
        return (across, along) if self.axis == "horizontal" else (along, across)


@dataclass
class Candidate:
    hits: list[Hit]
    runs: list[tuple[int, int]]
    position: float
    step: float
    step_mad: float
    width: float
    direction: str
    supported: int
    span_lines: int
    identifier: str = ""


@dataclass
class ScanResult:
    scan: Scan
    hits: list[Hit]
    candidates: list[Candidate] = field(default_factory=list)
    ignored: list[dict] = field(default_factory=list)
    skipped_lines: int = 0


def robust_step(values, observed, position, flank):
    """Piecewise-linear level fit around a transition, with iterative trimming."""
    length = len(values)
    lo = max(0, int(np.floor(position)) - flank)
    hi = min(length - 1, int(np.ceil(position)) + flank)
    if hi - lo < 6:
        return None
    xs = np.arange(lo, hi + 1, dtype=np.float64) - position
    ys = values[lo:hi + 1].astype(np.float64)
    keep = observed[lo:hi + 1].copy()
    if keep.sum() < 6:
        return None
    design = np.column_stack([np.ones_like(xs), xs, (xs >= 0).astype(np.float64)])
    coefficients = np.zeros(3)
    scale = 0.0
    for _ in range(4):
        if np.count_nonzero(keep) < 6:
            return None
        coefficients, *_ = np.linalg.lstsq(design[keep], ys[keep], rcond=None)
        residual = ys - design @ coefficients
        sample = residual[keep]
        scale = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
        cutoff = max(2.8 * scale, 1e-6)
        renewed = np.abs(residual) <= cutoff
        if np.count_nonzero(renewed) < 6:
            break
        keep = renewed
    return {"step": float(coefficients[2]), "scale": scale}


def refined_peak_position(gradient, peak):
    if peak < 1 or peak > len(gradient) - 2:
        return float(peak)
    magnitude = abs(float(gradient[peak]))
    low = peak
    while low - 1 >= 0 and abs(abs(float(gradient[low - 1])) - magnitude) < 1e-12:
        low -= 1
    high = peak
    while high + 1 < len(gradient) and abs(abs(float(gradient[high + 1])) - magnitude) < 1e-12:
        high += 1
    if low != high:
        return (low + high) / 2.0
    left, right = abs(float(gradient[peak - 1])), abs(float(gradient[peak + 1]))
    curvature = left - 2 * magnitude + right
    if curvature < -1e-12:
        return peak + float(np.clip(.5 * (left - right) / curvature, -.5, .5))
    return float(peak)


def extrapolated_level(values, observed, start, stop, at):
    """Linear level of a reference zone extrapolated to the transition position."""
    start = max(0, int(start))
    stop = min(len(values), int(stop))
    if stop - start < 4:
        return None
    keep = observed[start:stop]
    if keep.sum() < 4:
        return None
    xs = np.arange(start, stop, dtype=np.float64)
    slope, intercept = np.linalg.lstsq(
        np.column_stack([xs[keep], np.ones(int(keep.sum()))]),
        values[start:stop][keep].astype(np.float64), rcond=None,
    )[0]
    return float(intercept + slope * at)


def transition_slope(gradient, position):
    """Peak slope of |gradient| near a candidate transition."""
    peak = int(round(position))
    if peak < 1 or peak > len(gradient) - 2:
        return None
    slope = float(np.max(np.abs(gradient[peak - 1:peak + 2])))
    return slope if slope > 1e-12 else None


def profile_hits(values, observed, *, sigma, min_step, max_width, min_spacing):
    """List narrow level changes on one profile with their raw width and locality.

    A change counts as local when its far reference levels do not confirm it, or when an
    opposite change of comparable size sits within a few transition widths — a bright or
    dark feature returns to its original level instead of stepping to a new one.
    """
    length = len(values)
    x = np.arange(length)
    if observed.mean() < .82:
        return []
    filled = values.astype(np.float64) if observed.all() else np.interp(
        x, x[observed], values[observed])
    smoothed = gaussian_filter1d(filled, sigma)
    gradient = np.gradient(smoothed)
    central = gradient[2:-2]
    if central.size == 0:
        return []
    median = float(np.median(central))
    noise = 1.4826 * float(np.median(np.abs(central - median)))
    floor = max(noise * 4.5, min_step / max(2 * max_width, 1))
    peaks, _ = find_peaks(np.abs(gradient), height=floor, distance=max(1, min_spacing))
    if peaks.size > 40:
        peaks = peaks[np.argsort(np.abs(gradient[peaks]))[::-1][:40]]
        peaks.sort()
    inner = max(3, int(round(2 * max_width)))
    far = max(inner + 6, int(round(8 * max_width)))
    margin = far + 3
    flank = max(5, int(round(2.5 * max_width)) + 1)
    found = []
    for peak in peaks:
        peak = int(peak)
        if peak < margin or peak > length - 1 - margin:
            continue
        position = refined_peak_position(gradient, peak)
        fit = robust_step(smoothed, observed, position, flank)
        if fit is None or abs(fit["step"]) < min_step:
            continue
        left = extrapolated_level(smoothed, observed, np.floor(position) - far,
                                  np.floor(position) - inner + 1, position)
        right = extrapolated_level(smoothed, observed, np.ceil(position) + inner,
                                   np.ceil(position) + far + 1, position)
        slope = transition_slope(gradient, position)
        amplitude = None if left is None or right is None else right - left
        local = (
            amplitude is None
            or abs(amplitude) < max(.45 * abs(fit["step"]), 1e-6)
            or np.sign(amplitude) != np.sign(fit["step"])
            or slope is None
        )
        raw_width = None if amplitude is None or slope is None else abs(amplitude) / slope
        found.append(Hit(-1, position, float(fit["step"]), raw_width, local))
    for hit in found:
        if hit.local:
            continue
        for other in found:
            if (other is not hit and other.step * hit.step < 0
                    and abs(other.position - hit.position) <= max(4.0, 3.0 * max_width)
                    and abs(other.step) >= .5 * abs(hit.step)):
                hit.local = True
                break
    return found


def neighbour_counts(hits, line_window, position_tolerance):
    by_line = {}
    for hit in hits:
        by_line.setdefault(hit.line, []).append(hit.position)
    counts = []
    for hit in hits:
        seen = 0
        for line in range(hit.line - line_window, hit.line + line_window + 1):
            if line == hit.line:
                continue
            seen += sum(1 for position in by_line.get(line, ())
                        if abs(position - hit.position) <= position_tolerance)
        counts.append(seen)
    return counts


def position_clusters(hits, tolerance, line_window=NEIGHBOUR_LINES):
    """Link hits that continue a transition across nearby profiles.

    Neighbouring profiles may drift slightly in position (perspective, calibration),
    so a link is made to any hit within ``tolerance`` on one of the next
    ``line_window`` profiles.
    """
    ordered = sorted(hits, key=lambda hit: (hit.line, hit.position))
    parent = list(range(len(ordered)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    active = []
    for index, hit in enumerate(ordered):
        active = [other for other in active if hit.line - ordered[other].line <= line_window]
        for other in active:
            if abs(ordered[other].position - hit.position) <= tolerance:
                a, b = root(other), root(index)
                if a != b:
                    parent[b] = a
        active.append(index)
    groups = {}
    for index, hit in enumerate(ordered):
        groups.setdefault(root(index), []).append(hit)
    return list(groups.values())


def line_runs(lines, gap_limit, min_lines):
    ordered = sorted(set(lines))
    if not ordered:
        return []
    runs = []
    start = previous = ordered[0]
    for line in ordered[1:]:
        if line - previous > gap_limit + 1:
            runs.append((start, previous))
            start = line
        previous = line
    runs.append((start, previous))
    return [(first, last) for first, last in runs if last - first + 1 >= min_lines]


def classify_cluster(group, max_width, sigma):
    """Decide the cluster's kind from the median width of its member transitions."""
    widths = np.asarray([hit.raw_width for hit in group if hit.raw_width is not None])
    if widths.size == 0:
        return None
    own = 2.563 * sigma
    raw = float(np.median(widths))
    effective = float(np.sqrt(max(raw * raw - own * own, 0.0)))
    return {"kind": "step" if effective <= max_width else "smooth", "width": effective}


def build_candidate(group, gap_limit, min_lines):
    inside = [hit for hit in group if not hit.local]
    if not inside:
        return None, "local"
    runs = line_runs([hit.line for hit in inside], gap_limit, min_lines)
    if not runs:
        return None, "short"
    inside = [hit for hit in inside if any(a <= hit.line <= b for a, b in runs)]
    steps = np.asarray([hit.step for hit in inside])
    median_step = float(np.median(steps))
    if np.mean(np.sign(steps) == np.sign(median_step)) < .75:
        return None, "mixed"
    inside = [hit for hit in inside if np.sign(hit.step) == np.sign(median_step)]
    runs = line_runs([hit.line for hit in inside], gap_limit, min_lines)
    if not runs:
        return None, "short"
    inside = [hit for hit in inside if any(a <= hit.line <= b for a, b in runs)]
    return Candidate(
        hits=inside,
        runs=runs,
        position=float(np.median([hit.position for hit in inside])),
        step=median_step,
        step_mad=rounding(1.4826 * float(np.median(np.abs(steps - median_step)))) or 0.0,
        width=0.0,
        direction="bright" if median_step > 0 else "dark",
        supported=len(inside),
        span_lines=sum(b - a + 1 for a, b in runs),
    ), None


def ignored_row(group, kind, reason):
    lines = [hit.line for hit in group]
    widths = [hit.raw_width for hit in group if hit.raw_width is not None]
    return {
        "kind": kind,
        "reason": reason,
        "profiles": len(group),
        "panel_position": rounding(np.median([hit.position for hit in group])),
        "step_code": rounding(float(np.median([hit.step for hit in group]))),
        "transition_width_samples": rounding(float(np.median(widths))) if widths else None,
        "first_line": int(min(lines)) + 1,
        "last_line": int(max(lines)) + 1,
    }


def run_scan(plane: Plane, axis: str, *, sigma, min_step, max_width, min_spacing,
             neighbours, min_lines, merge, gap) -> ScanResult:
    gray, observed = plane.gray, plane.observed
    if axis == "vertical":
        gray, observed = gray.T, observed.T
    height, width = gray.shape
    scan = Scan(
        gray=gray, observed=observed,
        scan_coord=np.empty(width), line_coord=np.empty(height),
        along_native=0, across_native=0, along_span=0.0, across_span=0.0, axis=axis,
        region=plane.region,
    )
    hits = []
    skipped = 0
    for index in range(height):
        covered = observed[index]
        if covered.mean() < .82:
            skipped += 1
            continue
        for hit in profile_hits(gray[index], covered, sigma=sigma, min_step=min_step,
                                max_width=max_width, min_spacing=min_spacing):
            hit.line = index
            hits.append(hit)
    result = ScanResult(scan=scan, hits=hits, skipped_lines=skipped)
    counts = neighbour_counts(hits, NEIGHBOUR_LINES, merge)
    supported = [hit for hit, count in zip(hits, counts, strict=True) if count >= neighbours]
    clusters = position_clusters(supported, merge, line_window=gap + 1)
    ignored = []
    candidates = []
    for group in clusters:
        if len(group) < 2:
            continue
        measured = classify_cluster(group, max_width, sigma)
        if all(hit.local for hit in group):
            ignored.append(ignored_row(group, "local", "孤立亮点或回返"))
            continue
        if measured is None:
            ignored.append(ignored_row(group, "unclear", "两端参照不足"))
            continue
        if measured["kind"] == "smooth":
            ignored.append(ignored_row(group, "smooth", "平滑过渡较宽"))
            continue
        candidate, reason = build_candidate(group, gap, min_lines)
        if candidate is None:
            if reason == "local":
                ignored.append(ignored_row(group, "local", "孤立亮点或回返"))
            else:
                ignored.append(ignored_row(group, "step", "可见段过短或方向不一致"))
            continue
        candidate.width = measured["width"]
        candidates.append(candidate)
    candidates.sort(key=lambda item: item.position)
    if len(candidates) > MAX_CANDIDATES:
        candidates.sort(key=lambda item: (-item.supported * abs(item.step), item.position))
        candidates = candidates[:MAX_CANDIDATES]
        candidates.sort(key=lambda item: item.position)
    for index, candidate in enumerate(candidates, start=1):
        candidate.identifier = f"C{index:02d}"
    result.candidates = candidates
    ignored.sort(key=lambda row: (row["kind"], -row["profiles"], -abs(row["step_code"] or 0)))
    result.ignored = ignored[:60]
    return result


def decision_of(identifier, confirmed):
    if confirmed is None:
        return "candidate"
    return "confirmed" if identifier in confirmed else "excluded"


def normalized_box(scan: Scan, position, first, last):
    """Region-normalized box in half-open working samples across a run of profiles."""
    half = max(1.2 / scan.length, .0035)
    if scan.axis == "horizontal":
        x = (position + .5) / scan.length
        return (x - half, first / scan.lines, x + half, (last + 1) / scan.lines)
    y = (position + .5) / scan.length
    return (first / scan.lines, y - half, (last + 1) / scan.lines, y + half)


def region_pixels(scan: Scan, box):
    """Lift a region-normalized box into panel coordinates."""
    region = scan.region
    left, top, right, bottom = box
    return (
        region.left + left * (region.right - region.left),
        region.top + top * (region.bottom - region.top),
        region.left + right * (region.right - region.left),
        region.top + bottom * (region.bottom - region.top),
    )


def caption_for(candidate, confirmed):
    decision = decision_of(candidate.identifier, confirmed)
    if decision == "excluded":
        return MUTED_INK, f"{candidate.identifier}x"
    ink = BRIGHT_INK if candidate.direction == "bright" else DARK_INK
    return ink, f"{candidate.identifier}{'+' if candidate.direction == 'bright' else '-'}"


def row_for(candidate, scan: Scan, bits: int, confirmed):
    segments = " / ".join(
        f"{scan.line_coord[a]:.3f}–{scan.line_coord[b]:.3f}" for a, b in candidate.runs
    )
    first, last = candidate.runs[0]
    longest = max(candidate.runs, key=lambda run: run[1] - run[0])
    anchor = (longest[0] + longest[1]) / 2
    panel_x, panel_y = scan.panel_point(anchor, candidate.position)
    column = int(np.clip(round(candidate.position), 0, scan.length - 1))
    plates = 65535 if bits == 16 else 255
    box = normalized_box(scan, candidate.position, first, last)
    return {
        "candidate": candidate.identifier,
        "direction": "往亮" if candidate.direction == "bright" else "往暗",
        "step_code": rounding(candidate.step, 6),
        "step_recorded_counts": rounding(candidate.step * plates, 2),
        "step_percent_full_scale": rounding(abs(candidate.step) * 100, 3),
        "step_mad": rounding(candidate.step_mad, 6),
        "panel_position": rounding(scan.scan_coord[column], 5),
        "panel_offset_px": rounding(scan.scan_coord[column] * scan.along_native, 2),
        "transition_width_samples": rounding(candidate.width, 3),
        "width_panel_px": rounding(candidate.width * scan.scan_step, 3),
        "visible_segments": segments,
        "run_count": len(candidate.runs),
        "visible_from": rounding(scan.line_coord[first], 5),
        "visible_to": rounding(scan.line_coord[last], 5),
        "visible_panel_px": rounding(
            (scan.line_coord[last] - scan.line_coord[first]) * scan.across_native + scan.line_step, 2
        ),
        "supported_lines": candidate.supported,
        "span_lines": candidate.span_lines,
        "line_coverage_pct": rounding(100 * candidate.supported / max(1, candidate.span_lines), 2),
        "camera_x": rounding(panel_x, 6),
        "camera_y": rounding(panel_y, 6),
        "box": [rounding(value) for value in region_pixels(scan, box)],
        "decision": {"confirmed": "已确认", "excluded": "已排除",
                     "candidate": "未确认"}[decision_of(candidate.identifier, confirmed)],
    }


def boxes_for(candidate, scan: Scan, confirmed):
    ink, label = caption_for(candidate, confirmed)
    return [(normalized_box(scan, candidate.position, a, b), label, ink)
            for a, b in candidate.runs[:12]]


def step_map(result: ScanResult):
    """Mark each retained candidate's transition on every profile where it held."""
    scan = result.scan
    values = np.full((scan.lines, scan.length), np.nan, dtype=np.float32)
    for candidate in result.candidates:
        for hit in candidate.hits:
            column = int(np.clip(round(hit.position), 0, scan.length - 1))
            current = values[hit.line, column]
            if not np.isfinite(current) or abs(hit.step) > abs(current):
                values[hit.line, column] = hit.step
    return values.T if scan.axis == "vertical" else values


def original_overlay(scene: Scene, scan: Scan, candidates, confirmed):
    photo = masked_rgb(scene.image.rgb, scene.image.valid)
    draw = ImageDraw.Draw(photo)
    source_width, _ = scene.image.size
    scale = [source_width - 1, scene.image.size[1] - 1]
    thickness = max(2, source_width // 400)
    radius = max(4, source_width // 190)
    for candidate in candidates:
        ink, caption = caption_for(candidate, confirmed)
        for first, last in candidate.runs[:12]:
            samples = np.linspace(first, last, 9) if last > first else np.asarray([float(first)])
            panel = np.asarray([scan.panel_point(line, candidate.position) for line in samples])
            camera = scene.calibration.panel_to_camera(panel) * scale
            points = [(float(u), float(v)) for u, v in camera
                      if np.isfinite(u) and np.isfinite(v)]
            if len(points) >= 2:
                draw.line(points, fill=ink, width=thickness)
            if points:
                mx, my = points[len(points) // 2]
                draw.ellipse((mx - radius, my - radius, mx + radius, my + radius),
                             outline=ink, width=2)
                draw.text((mx + radius + 3, my - radius - 3), caption, fill=ink)
    return photo


def profile_chart(scan: Scan, candidate: Candidate, half: int):
    first, last = max(candidate.runs, key=lambda run: run[1] - run[0])
    center = int(round((first + last) / 2))
    indices = sorted({int(np.clip(center + offset, first, last)) for offset in (-2, -1, 0, 1, 2)})
    start = max(0, int(np.floor(candidate.position)) - half)
    stop = min(scan.length, int(np.ceil(candidate.position)) + half + 1)
    if stop - start < 6:
        return None
    offsets = (np.arange(start, stop) - candidate.position) * scan.scan_step
    xs = np.arange(start, stop, dtype=np.float64)
    inner = max(3.0, 2.0 * candidate.width)
    near_left = (xs >= candidate.position - 3 * inner) & (xs <= candidate.position - inner)
    near_right = (xs >= candidate.position + inner) & (xs <= candidate.position + 3 * inner)
    series = []
    for line in indices:
        values = scan.gray[line, start:stop].astype(np.float64)
        observed = scan.observed[line, start:stop]
        left = observed & near_left
        right = observed & near_right
        if left.sum() < 2 or right.sum() < 2:
            continue
        left_x, left_y = float(xs[left].mean()), float(np.median(values[left]))
        right_x, right_y = float(xs[right].mean()), float(np.median(values[right]))
        slope = (right_y - left_y) / max(right_x - left_x, 1e-9)
        residual = values - (left_y + slope * (xs - left_x))
        residual[~observed] = np.nan
        series.append((f"L{line + 1}", offsets, residual))
    if not series:
        return None
    return chart_png(series, x_label="Panel pixels from step",
                     y_label="Code relative to the two-sided flank trend",
                     markers=[(0.0, "step")])


def candidate_payload(candidate, scan: Scan, confirmed):
    first, last = candidate.runs[0]
    longest = max(candidate.runs, key=lambda run: run[1] - run[0])
    return {
        "id": candidate.identifier,
        "direction": candidate.direction,
        "step": rounding(candidate.step, 6),
        "width": rounding(candidate.width, 3),
        "position": rounding(scan.scan_coord[
            int(np.clip(round(candidate.position), 0, scan.length - 1))], 5),
        "segments": [[rounding(scan.line_coord[a]), rounding(scan.line_coord[b])]
                     for a, b in candidate.runs],
        "supported": candidate.supported,
        "span": candidate.span_lines,
        "decision": decision_of(candidate.identifier, confirmed),
        "box": [rounding(value) for value in region_pixels(
            scan, normalized_box(scan, candidate.position, first, last))],
    }


def banding_assay(scene: Scene, second, region: Region, options: dict) -> Report:
    scene.require_role({"ramp"}, "请选择灰阶渐变（灰阶角色）采集。")
    axis = choice(options.get("axis"), {"horizontal", "vertical"}, "明暗递变方向", default="horizontal")
    polarity = choice(options.get("polarity"), {"any", "bright", "dark"}, "保留跳变方向", default="any")
    sigma = numeric(options.get("sigma"), "剖面去噪尺度", .2, 3, default=.6)
    min_step = numeric(options.get("min_step"), "最小跳变码值", .0005, .2, default=.004)
    max_width = numeric(options.get("max_width"), "突变宽度上限（工作采样点）", 1, 12, default=3)
    min_spacing = integral(options.get("spacing"), "相邻跳变最小间距", 1, 80, default=4)
    neighbours = integral(options.get("neighbours"), "相邻剖面佐证数", 1, 8, default=2)
    min_lines = integral(options.get("min_lines"), "最短可见段剖面数", 2, 400, default=8)
    merge = integral(options.get("merge"), "位置合并容差", 1, 30, default=3)
    gap = integral(options.get("gap"), "可见段允许断档剖面数", 0, 20, default=3)
    confirmed_raw = options.get("confirmed_candidates")
    confirmed = None
    if confirmed_raw is not None:
        confirmed = {
            plain(item, "确认候选编号", 8)
            for item in sequence(confirmed_raw, "确认候选编号", minimum=0, maximum=MAX_CANDIDATES)
        }
    plane = make_plane(scene, region, max_side=1000)
    plane.require_coverage(.55)
    result = run_scan(plane, axis, sigma=sigma, min_step=min_step, max_width=max_width,
                      min_spacing=min_spacing, neighbours=neighbours, min_lines=min_lines,
                      merge=merge, gap=gap)
    scan = result.scan
    working_height, working_width = plane.shape
    scan.along_native = scene.scale[1] if axis == "vertical" else scene.scale[0]
    scan.across_native = scene.scale[0] if axis == "vertical" else scene.scale[1]
    scan.along_span = (region.bottom - region.top) if axis == "vertical" else (
        region.right - region.left)
    scan.across_span = (region.right - region.left) if axis == "vertical" else (
        region.bottom - region.top)
    if axis == "vertical":
        scan.scan_coord = (region.top + (np.arange(working_height) + .5)
                           / working_height * (region.bottom - region.top))
        scan.line_coord = (region.left + (np.arange(working_width) + .5)
                           / working_width * (region.right - region.left))
    else:
        scan.scan_coord = (region.left + (np.arange(working_width) + .5)
                           / working_width * (region.right - region.left))
        scan.line_coord = (region.top + (np.arange(working_height) + .5)
                           / working_height * (region.bottom - region.top))
    if scan.length < 64:
        raise Problem("选区沿明暗方向的采样点太少，请扩大选区。")
    if scan.lines < 12:
        raise Problem("选区横向的采样剖面太少，请扩大选区。")
    candidates = result.candidates
    if polarity != "any":
        candidates = [item for item in candidates if item.direction == polarity]
        for index, candidate in enumerate(candidates, start=1):
            candidate.identifier = f"C{index:02d}"
        result.candidates = candidates
    if confirmed is not None:
        unknown = sorted(confirmed - {item.identifier for item in candidates})
        if unknown:
            raise Problem(f"候选编号 {unknown[0]} 已不在本次结果中，请重新预览后再保存。", 409)
    rows = [row_for(candidate, scan, scene.image.bits, confirmed) for candidate in candidates]
    confirmed_rows = [row for row in rows if row["decision"] == "已确认"]
    visible = [item for item in candidates if decision_of(item.identifier, confirmed) != "excluded"]
    step_values = [abs(hit.step) for hit in result.hits if not hit.local]
    limit = max(min_step, float(np.percentile(step_values, 97))) if step_values else min_step
    metrics = {
        **plane.info(),
        "along_axis": "垂直（自上而下）" if axis == "vertical" else "水平（自左向右）",
        "polarity": {"any": "两个方向都保留", "bright": "只保留往亮",
                     "dark": "只保留往暗"}[polarity],
        "candidate_count": len(candidates),
        "confirmed_count": len(confirmed_rows) if confirmed is not None else None,
        "excluded_count": len(rows) - len(confirmed_rows) if confirmed is not None else None,
        "step_threshold_code": rounding(min_step),
        "max_transition_width_samples": rounding(max_width),
        "sample_size_panel_px": [rounding(scan.scan_step, 3), rounding(scan.line_step, 3)],
        "neighbour_profiles_required": neighbours,
        "minimum_visible_lines": min_lines,
        "merge_tolerance_samples": merge,
        "profiles_used": scan.lines - result.skipped_lines,
        "profiles_skipped": result.skipped_lines,
        "smooth_transitions_ignored": sum(
            1 for row in result.ignored if row["kind"] == "smooth"),
        "local_features_ignored": sum(
            1 for row in result.ignored if row["kind"] == "local"),
        "step_map_scale": [-rounding(limit), rounding(limit)],
    }
    report = Report(
        "灰阶跳变条带",
        "沿选定明暗方向检查在邻近剖面上重复出现的窄跳变；较宽的平滑过渡和孤立亮点单列，不计入条带。",
        metrics, rows,
    )
    report.image("banding-overlay.png", overlay_regions(
        masked_rgb(plane.rgb, plane.observed),
        [box for candidate in candidates for box in boxes_for(candidate, scan, confirmed)],
    ) if candidates else masked_rgb(plane.rgb, plane.observed))
    if candidates:
        report.add("banding-median-profile.png", chart_png(
            [("Median profile", scan.scan_coord, np.nanmedian(
                np.where(scan.observed, scan.gray, np.nan), axis=0))],
            x_label="Panel position along scan", y_label="Recorded code",
            markers=[(row["panel_position"], row["candidate"])
                     for row in rows if row["decision"] != "已排除"],
        ), "image/png")
        computed_map = step_map(result)
        report.image("banding-step-map.png", heatmap(
            computed_map, np.isfinite(computed_map), -limit, limit, diverging=True,
        ))
        report.image("banding-original.png", original_overlay(scene, scan, candidates, confirmed))
        half = int(np.clip(round(.025 * scan.length), 12, 48))
        drawn = 0
        for candidate in visible:
            if drawn >= MAX_PROFILE_CHARTS:
                break
            chart = profile_chart(scan, candidate, half)
            if chart:
                report.add(f"banding-candidate-{candidate.identifier[1:]}-profile.png", chart,
                           "image/png")
                drawn += 1
    report.table("banding-candidates.csv", rows)
    report.table("banding-ignored.csv", result.ignored)
    if confirmed is None:
        report.warnings.append("当前为候选预览；保存前可勾选确认或排除，保存后记录保留每位候选的确认状态。")
    if result.ignored:
        report.warnings.append("平滑过渡、孤立亮点和佐证不足的过渡只列在排除表中，不进入条带标记。")
    if result.skipped_lines > .4 * scan.lines:
        report.warnings.append("较多剖面因有效采样不足被跳过，结果只反映可用部分。")
    if not candidates:
        report.warnings.append("当前门槛下没有同时在邻近剖面上成立的窄跳变。")
    report.warnings.append("覆图暖色 + 为沿扫描方向往亮，冷色 - 为往暗；位置以面板归一化坐标为准。")
    report.warnings.append("可测宽度受工作分辨率、相机采样和镜头平滑限制，窄于一个工作采样的跳变只能以宽度上限表述。")
    report.controls = {
        "kind": "candidate-review",
        "option": "confirmed_candidates",
        "decided": confirmed is not None,
        "decisions": ["confirmed", "excluded"],
        "items": [candidate_payload(candidate, scan, confirmed) for candidate in candidates],
        "note": "勾选要保留的跳变并排除其余，保存时记录每项状态。",
    }
    return report


def configs():
    return {
        "banding": {
            "title": "灰阶跳变条带",
            "group": "采样与细节",
            "description": "在灰阶渐变中定位邻近剖面共同出现的窄跳变，可预览后逐项确认。",
            "run": banding_assay,
            "options": [
                {"key": "axis", "label": "明暗递变方向", "type": "select",
                 "values": {"horizontal": "水平（自左向右）", "vertical": "垂直（自上而下）"},
                 "default": "horizontal"},
                {"key": "polarity", "label": "保留跳变方向", "type": "select",
                 "values": {"any": "两个方向都保留", "bright": "只保留往亮", "dark": "只保留往暗"},
                 "default": "any"},
                number("min_step", "最小跳变码值", .004, .0005, .2, step=.0005),
                number("max_width", "突变宽度上限（工作采样点）", 3, 1, 12, step=.5),
                number("sigma", "剖面去噪尺度", .6, .2, 3, step=.1),
                number("spacing", "相邻跳变最小间距", 4, 1, 80),
                number("neighbours", "相邻剖面佐证数", 2, 1, 8),
                number("min_lines", "最短可见段剖面数", 8, 2, 400),
                number("merge", "位置合并容差", 3, 1, 30),
                number("gap", "可见段允许断档剖面数", 3, 0, 20),
            ],
        },
    }
