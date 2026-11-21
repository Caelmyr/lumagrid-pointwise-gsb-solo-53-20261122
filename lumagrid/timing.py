"""Session timing audit joins camera observations to a reference recorder log."""
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment

from .inspection import Report, Scene
from .model import Problem, integral, numeric, plain, recorded_time
from .raster import chart_png, rounding


MAX_TIME_GUIDE_ROWS = 1200
MAX_LOG_BYTES = 160 * 1024
MAX_GROUP_FRAMES = 90


@dataclass(frozen=True)
class Trigger:
    timestamp: float
    original_time: str
    event: str
    level: float | None
    seq: int


@dataclass(frozen=True)
class CameraEvent:
    timestamp: float
    original_time: str
    capture_id: str
    title: str
    level: float | None
    shutter_seconds: float
    role: str


def parse_log(raw: str) -> list[Trigger]:
    if not isinstance(raw, str) or not raw.strip():
        raise Problem("请粘贴包含 timestamp 和 event 表头的参考 CSV。")
    if len(raw.encode("utf-8")) > MAX_LOG_BYTES:
        raise Problem("参考 CSV 超过 160 KiB。", 413)
    stream = StringIO(raw.lstrip("\ufeff"), newline="")
    try:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not {"timestamp", "event"}.issubset(fields):
            raise Problem("参考 CSV 必须含 timestamp 和 event 两列，允许 level，不允许重复表头。")
        rows = []
        for index, item in enumerate(reader, start=1):
            if index > MAX_TIME_GUIDE_ROWS:
                raise Problem(f"参考记录最多 {MAX_TIME_GUIDE_ROWS} 行。")
            if None in item or any(item.get(key) is None for key in ("timestamp", "event")):
                raise Problem(f"参考 CSV 第 {index} 条数据的字段数不正确。")
            timestamp = plain(item.get("timestamp"), "录制时间", 50)
            stamp = recorded_time(timestamp)
            if stamp is None:
                raise Problem(f"参考 CSV 第 {index} 条缺少带时区的录制时间。")
            tag = plain(item.get("event"), "外部事件标签", 100)
            if not tag:
                raise Problem(f"参考 CSV 第 {index} 条缺少事件标签。")
            level_str = (item.get("level") or "").strip()
            level = numeric(level_str, "图案亮度", 0, 1) if level_str else None
            rows.append(Trigger(stamp.timestamp(), timestamp, tag, level, index))
    except csv.Error as exc:
        raise Problem(f"参考 CSV 无法读取：{exc}。") from exc
    if not rows:
        raise Problem("参考 CSV 没有外部记录。")
    return sorted(rows, key=lambda item: (item.timestamp, item.seq))


def camera_events(captures: list[dict], camera: str | None):
    events, no_clock, filtered = [], [], []
    for item in captures:
        exposure = item["exposure"]
        if camera and exposure.get("camera") != camera:
            filtered.append(item["title"])
            continue
        time = exposure.get("time")
        if not time:
            no_clock.append(item["title"])
            continue
        shutter = (exposure.get("shutter_ms") or 0.0) / 1000
        events.append(CameraEvent(
            recorded_time(time).timestamp(),
            time,
            item["id"],
            item["title"],
            exposure.get("level"),
            shutter,
            item["role"],
        ))
    return sorted(events, key=lambda item: (item.timestamp, item.capture_id)), no_clock, filtered


def candidate_matrix(
    frames: list[CameraEvent],
    triggers: list[Trigger],
    offset: float,
    max_delta: float,
    level_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trigger_times = np.array([item.timestamp for item in triggers])
    actual_times = np.array([item.timestamp + .5 * item.shutter_seconds + offset for item in frames])
    differences = actual_times[:, None] - trigger_times
    allowed = np.abs(differences) <= max_delta
    unknown = np.zeros_like(allowed)
    if level_margin <= 1:
        frame_level = np.asarray([item.level if item.level is not None else np.nan for item in frames])
        trigger_level = np.asarray([item.level if item.level is not None else np.nan for item in triggers])
        level_delta = np.abs(frame_level[:, None] - trigger_level)
        unknown = np.isnan(level_delta)
        allowed &= unknown | (level_delta <= level_margin)
    return differences, allowed, unknown


def choose_pairs(
    milliseconds: np.ndarray,
    allowed: np.ndarray,
    unknown: np.ndarray,
    extra_video: float,
) -> dict[int, int]:
    # Unmatched placeholders are abundant; their penalty prioritizes match count
    # before timing residual. Recorded events cannot be paired to two frames.
    frame_count, log_count = allowed.shape
    cell_cost = np.abs(milliseconds) + .015 * extra_video * unknown
    max_real_cost = max(1.0, extra_video * 1.2)
    missing_weight = (frame_count + 1) * max_real_cost
    costs = np.full((frame_count, log_count + frame_count), missing_weight * 10)
    costs[:, :log_count][allowed] = cell_cost[allowed]
    for frame in range(frame_count):
        costs[frame, log_count + frame] = missing_weight
    a, b = linear_sum_assignment(costs)
    return {int(i): int(j) for i, j in zip(a, b, strict=True) if j < log_count and allowed[i, j]}


def plausible_offsets(frames: list[CameraEvent], triggers: list[Trigger], bound: float):
    event_times = np.array([item.timestamp for item in triggers])
    captured = np.array([item.timestamp + .5 * item.shutter_seconds for item in frames])
    diffs = event_times - captured[:, None]
    accepted = diffs[(np.abs(diffs) <= bound) & np.isfinite(diffs)]
    if not accepted.size:
        return []
    quantized = np.rint(accepted * 100).astype(int)
    bins, counts = np.unique(quantized, return_counts=True)
    ranking = np.lexsort((np.abs(bins), -counts))
    return [float(bins[index]) * .01 for index in ranking[:45]]


def estimate_offset(
    frames: list[CameraEvent],
    triggers: list[Trigger],
    bound: float,
    max_delta: float,
    level_margin: float,
):
    offsets = plausible_offsets(frames, triggers, bound)
    candidates = []
    for seconds in offsets:
        residual, keep, unknown = candidate_matrix(frames, triggers, seconds, max_delta, level_margin)
        selected = choose_pairs(residual * 1000, keep, unknown, 1000 * max_delta)
        selected_deltas = [residual[i, j] for i, j in selected.items()]
        spread = float(np.sum(np.abs(selected_deltas))) if selected_deltas else float("inf")
        candidates.append((len(selected), spread, seconds, selected_deltas))
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1], abs(item[2]), item[2]))
    if not ranked or ranked[0][0] < 3:
        raise Problem("自动对时至少需要三对可匹配的已登记采集和外部事件。")
    count, _, offset, residuals = ranked[0]
    refined = offset - float(np.median(residuals))
    if abs(refined) > bound:
        refined = offset
    near = [candidate for candidate in ranked[1:] if candidate[0] == count
            and abs(candidate[2] - refined) > max(2 * max_delta, .03)
            and candidate[1] - ranked[0][1] < max_delta * .30]
    return refined, len(near)


def timing_report(captures: list[dict], raw: str, data: dict) -> Report:
    triggers = parse_log(raw)
    camera = plain(data.get("camera"), "相机筛选", 80) or None
    frames, no_clock, filtered = camera_events(captures, camera)
    if not frames:
        raise Problem("当前相机筛选下没有已登记采集时间的图片。")
    if len(frames) > MAX_GROUP_FRAMES:
        raise Problem(f"参与配对的采集最多 {MAX_GROUP_FRAMES} 张。")
    mode = data.get("mode") or "manual"
    margin = numeric(data.get("margin_ms"), "时间容差", 1, 5000, default=38) / 1000
    level_margin = numeric(data.get("level_margin"), "亮度差容差", 0, 1, default=.08)
    if mode == "estimate":
        bound = numeric(data.get("offset_bound_ms"), "偏移搜索范围", 5, 60_000, default=1000) / 1000
        offset, alternatives = estimate_offset(frames, triggers, bound, margin, level_margin)
    elif mode == "manual":
        offset = numeric(data.get("offset_ms"), "相机时间补偿", -60_000, 60_000, default=0) / 1000
        alternatives = 0
    else:
        raise Problem("未知的对时方式。")
    residual, allowed, unknown = candidate_matrix(frames, triggers, offset, margin, level_margin)
    matches = choose_pairs(residual * 1000, allowed, unknown, 1000 * margin)
    report_rows = []
    used = set(matches.values())
    for index, frame in enumerate(frames):
        event_number = matches.get(index)
        triggered = triggers[event_number] if event_number is not None else None
        report_rows.append({
            "state": "paired" if triggered else "unmatched_frame",
            "capture_id": frame.capture_id,
            "capture": frame.title,
            "camera_timestamp": frame.original_time,
            "reference_timestamp": triggered.original_time if triggered else None,
            "event": triggered.event if triggered else None,
            "reference_row": triggered.seq if triggered else None,
            "capture_level": frame.level,
            "reference_level": triggered.level if triggered else None,
            "residual_ms": rounding(1000 * residual[index, event_number], 3) if triggered else None,
            "eligible_events": int(np.count_nonzero(allowed[index])),
        })
    for i, event in enumerate(triggers):
        if i not in used:
            report_rows.append({
                "state": "unmatched_reference",
                "reference_timestamp": event.original_time,
                "event": event.event,
                "reference_row": event.seq,
                "reference_level": event.level,
            })
    residuals = np.asarray([row["residual_ms"] for row in report_rows if row.get("residual_ms") is not None])
    metrics = {
        "offset_mode": mode,
        "clock_offset_ms": rounding(offset * 1000, 3),
        "matching_tolerance_ms": rounding(margin * 1000, 3),
        "frames_with_clock": len(frames),
        "unmatched_frames": len(frames) - len(matches),
        "frames_without_clock": len(no_clock),
        "excluded_camera_frames": len(filtered),
        "reference_events": len(triggers),
        "matched_pairs": len(matches),
        "unmatched_reference": len(triggers) - len(matches),
        "max_abs_residual_ms": rounding(np.abs(residuals).max(), 3) if residuals.size else None,
        "rms_residual_ms": rounding(np.sqrt(np.mean(residuals ** 2)), 3) if residuals.size else None,
    }
    report = Report("采集时钟配对", "依据曝光中心和外部触发日志建立一对一记录串联。", metrics, report_rows[:150])
    report.table("frame-timeline.csv", report_rows)
    if residuals.size:
        report.add("clock-residuals.png", chart_png(
            [("Matched residual", np.arange(1, len(residuals) + 1), residuals)],
            x_label="Matched camera frames", y_label="Camera - reference (ms)",
        ), "image/png")
    if alternatives:
        report.warnings.append(f"{alternatives} 组明显不同的偏移量获得接近支撑，请用设备时间标记复核。")
    if no_clock:
        report.warnings.append("这些采集缺少时间戳：" + "、".join(no_clock[:12]))
    report.warnings.append("配对检验时间和已登记亮度，不能单凭日志证明画面内容同步。")
    return report
