"""Consolidate manually marked bright spots from several calibrated re-shoots.

The operator marks suspicious spots on each original photo and gives each photo its own
positional tolerance (panel physical pixels).  Spots are then merged by their calibrated
panel position.  Two clicks on the same photo always stay two separate spots; a click on
another (usually blurrier) photo that fits more than one spot is held for review and is
counted for neither.  For every consolidated spot and photo the result says whether the
position was outside that photo, inside it but without any clicked evidence, or backed by
a click.
"""
from dataclasses import dataclass, field
from math import hypot
import json

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Calibration, scaled_dimensions
from .inspection import Report
from .model import Point, Problem, mapping, numeric, required_text, sequence
from .raster import decode, masked_rgb, resize_report, rounding
from .storage import Store


MIN_CAPTURES = 2
MAX_CAPTURES = 4
MAX_POINTS_PER_CAPTURE = 48
MAX_TOTAL_POINTS = 96
PANEL_EDGE_MARGIN = .01

STATE_TEXT = {
    "matched": "有点位",
    "review_overlap": "待复核重叠",
    "no_evidence": "画面内无点位",
    "not_captured": "未拍到该位置",
}
STATUS_TEXT = {
    "multi_capture": "多图互证",
    "single_capture": "仅单图有点位",
    "unconfirmed": "待复核",
}
STATE_COLORS = {
    "matched": (74, 202, 143),
    "review_overlap": (247, 189, 70),
    "no_evidence": (239, 143, 69),
    "not_captured": (150, 164, 171),
}
CAPTURE_COLORS = ((52, 130, 196), (213, 96, 58), (126, 92, 181), (52, 145, 99))


@dataclass
class CaptureEvidence:
    index: int
    panel: dict
    capture: dict
    revision: dict
    calibration: Calibration
    size: tuple[int, int]
    valid: np.ndarray
    display: Image.Image
    tolerance: float
    clicks: list[dict] = field(default_factory=list)


def _distance(first, second, size) -> float:
    return hypot((first[0] - second[0]) * size[0], (first[1] - second[1]) * size[1])


def _weighted_center(clicks, size):
    weights = np.asarray([1.0 / max(click["tolerance"], 1e-6) ** 2 for click in clicks])
    points = np.asarray([click["panel"] for click in clicks], dtype=np.float64)
    pixels = points * np.asarray(size, dtype=np.float64)
    center = (pixels * weights[:, None]).sum(axis=0) / weights.sum()
    return (float(center[0] / size[0]), float(center[1] / size[1]))


def _visibility(entry: CaptureEvidence, panel_point):
    """Return (visible, camera point).  Visible means inside the photo and opaque there."""
    camera = entry.calibration.panel_to_camera(np.asarray([panel_point], dtype=np.float64))[0]
    if not np.all(np.isfinite(camera)):
        return False, None
    x, y = float(camera[0]), float(camera[1])
    if not (-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6):
        return False, (x, y)
    width, height = entry.size
    column = int(np.clip(round(x * (width - 1)), 0, width - 1))
    row = int(np.clip(round(y * (height - 1)), 0, height - 1))
    return bool(entry.valid[row, column]), (x, y)


def _load(store: Store, panel_id: str, position: int, item: dict) -> CaptureEvidence:
    item = mapping(item, f"第 {position} 张照片")
    capture_id = required_text(item.get("capture_id"), "采集标识", 64)
    calibration_id = required_text(item.get("calibration_id"), "四角标定版本", 64)
    if calibration_id == "identity":
        raise Problem("亮点复检需要每张照片已保存的四角标定。")
    tolerance = numeric(item.get("tolerance_px"), "点位误差范围", .5, 200, default=6)
    raw_points = item.get("points")
    points = sequence([] if raw_points is None else raw_points, "照片点位",
                      minimum=0, maximum=MAX_POINTS_PER_CAPTURE)
    panel, capture, original, revision = store.read_for_analysis(capture_id, calibration_id)
    if panel["id"] != panel_id:
        raise Problem("所选照片不属于当前面板。")
    if revision is None:
        raise Problem("亮点复检需要每张照片已保存的四角标定。")
    calibration = Calibration.from_raw(revision["corners"])
    image = decode(original)
    size = image.size
    evidence = CaptureEvidence(
        position, panel, capture, revision, calibration, size,
        image.valid, resize_report(masked_rgb(image.rgb, image.valid), 1100), tolerance,
    )
    del image
    width, height = size
    for number, raw in enumerate(points, start=1):
        point = Point.parse(raw)
        camera = np.asarray(point.values(), dtype=np.float64)
        column = int(np.clip(round(camera[0] * (width - 1)), 0, width - 1))
        row = int(np.clip(round(camera[1] * (height - 1)), 0, height - 1))
        if not bool(evidence.valid[row, column]):
            raise Problem(f"第 {position} 张照片的第 {number} 个点落在透明或画面外区域。")
        mapped = calibration.camera_to_panel(camera[None, :])[0]
        if not np.all(np.isfinite(mapped)):
            raise Problem(f"第 {position} 张照片的第 {number} 个点无法映射到面板。")
        panel_x, panel_y = float(mapped[0]), float(mapped[1])
        if not (-.05 <= panel_x <= 1.05 and -.05 <= panel_y <= 1.05):
            raise Problem(f"第 {position} 张照片的第 {number} 个点不在四角标定的面板范围内。")
        evidence.clicks.append({
            "capture_index": position,
            "label": f"C{position}P{number}",
            "camera": (float(camera[0]), float(camera[1])),
            "camera_px": (round(float(camera[0]) * (width - 1), 3),
                          round(float(camera[1]) * (height - 1), 3)),
            "panel": (panel_x, panel_y),
            "tolerance": tolerance,
            "edge": min(panel_x, panel_y, 1 - panel_x, 1 - panel_y) <= PANEL_EDGE_MARGIN,
            "ambiguous": False,
            "candidates": [],
            "spot": None,
        })
    return evidence


def _belongs(click, cluster):
    """A click may join a spot only if that spot holds no other click from its photo."""
    return bool(cluster["clicks"]) and all(
        member is click or member["capture_index"] != click["capture_index"]
        for member in cluster["clicks"]
    )


def _candidates(click, clusters, size):
    """Spots whose settled position lies within this click's tolerance."""
    return [
        cluster["id"] for cluster in clusters
        if _belongs(click, cluster)
        and _distance(click["panel"], cluster["panel"], size) <= click["tolerance"]
    ]


def _settle(clusters, size):
    for cluster in clusters:
        cluster["panel"] = _weighted_center(cluster["clicks"], size)


def _assign(entries, clusters, size, order):
    """Attach every click to exactly one spot, a spot new to it, or leave it for review.

    A click is left unassigned when it fits more than one settled spot.  A spot that already
    holds a click from the same photo is never joined, so two points marked separately on
    one photo always stay two spots.  Clicks are visited in the canonical sharpest-first
    order so the result does not depend on the order the operator selected the photos.
    """
    for entry in order:
        for click in entry.clicks:
            found = _candidates(click, clusters, size)
            if len(found) == 1:
                next(item for item in clusters if item["id"] == found[0])["clicks"].append(click)
                click["spot"] = found[0]
            elif not found:
                cluster = {"id": max((item["id"] for item in clusters), default=0) + 1,
                           "clicks": [click], "panel": click["panel"]}
                click["spot"] = cluster["id"]
                clusters.append(cluster)
            else:
                click["spot"] = None


def _rebuild(entries, size):
    """Consolidate clicks into shared panel positions.

    Seeding visits the sharpest photo (smallest tolerance) first so clicks are compared on
    the tighter evidence.  Each pass re-settles the spot positions, then re-attaches every
    click.  The loop stops once no click changes owner, so a click that fits two spots is
    never left counted for one of them.
    """
    order = sorted(entries, key=lambda entry: (entry.tolerance, entry.index))
    for entry in entries:
        for click in entry.clicks:
            click["spot"] = None
    clusters = []
    for _ in range(8):
        for cluster in clusters:
            cluster["clicks"] = []
        _assign(entries, clusters, size, order)
        clusters = [cluster for cluster in clusters if cluster["clicks"]]
        _settle(clusters, size)
        settled = {}
        for entry in entries:
            for click in entry.clicks:
                found = _candidates(click, clusters, size)
                settled[click["label"]] = (
                    None if len(found) > 1 else (found[0] if len(found) == 1 else click["spot"])
                )
        if all(settled[click["label"]] == click["spot"]
               for entry in entries for click in entry.clicks):
            break
    clusters.sort(key=lambda cluster: (
        cluster["panel"][1], cluster["panel"][0], min(c["capture_index"] for c in cluster["clicks"])
    ))
    for number, cluster in enumerate(clusters, start=1):
        cluster["id"] = number
    for cluster in clusters:
        for click in cluster["clicks"]:
            click["spot"] = cluster["id"]
    for entry in entries:
        for click in entry.clicks:
            click["candidates"] = _candidates(click, clusters, size)
            click["ambiguous"] = len(click["candidates"]) > 1
            if click["ambiguous"]:
                click["spot"] = None
    return clusters


def _consolidate(entries, size):
    return _rebuild(entries, size)


def _observe(entry: CaptureEvidence, cluster, size):
    """How one photo supports one consolidated spot.

    not_captured: the spot's panel position is outside this photo (or over transparent
    pixels), so the photo simply never recorded it.  no_evidence: the position is inside
    the photo but the operator did not click anything near it.  review_overlap: a clicked
    point fits this spot and at least one other, so it is held for review and counted for
    neither.  matched: at least one clicked point belongs to this spot alone.
    """
    visible, camera = _visibility(entry, cluster["panel"])
    if not visible:
        return {"state": "not_captured", "camera": camera, "distance": None,
                "matched": [], "ambiguous": [], "spots": []}
    ambiguous = [click for click in entry.clicks if click["ambiguous"]
                 and cluster["id"] in click["candidates"]]
    unique = [click for click in entry.clicks
              if not click["ambiguous"] and click["spot"] == cluster["id"]]
    if unique:
        state = "matched"
    elif ambiguous:
        state = "review_overlap"
    else:
        state = "no_evidence"
    chosen = unique or ambiguous
    distance = (min(_distance(click["panel"], cluster["panel"], size) for click in chosen)
                if chosen else None)
    return {
        "state": state,
        "camera": camera,
        "distance": distance,
        "matched": [click["label"] for click in unique],
        "ambiguous": [click["label"] for click in ambiguous],
        "spots": sorted({value for click in ambiguous for value in click["candidates"]}),
    }


def _ascii(text: str, fallback: str = "") -> str:
    cleaned = "".join(char if 32 <= ord(char) < 127 else " " for char in text)
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback


def _label(draw, position, text, color):
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=13)
    except (TypeError, ValueError):
        font = None
    draw.text(position, text, fill=color, font=font)


def _capture_overlay(entry: CaptureEvidence, clusters, observations, size):
    image = entry.display.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for cluster in clusters:
        observation = observations[cluster["id"]][entry.index]
        camera = observation["camera"]
        color = STATE_COLORS[observation["state"]]
        if camera is None or not (-0.15 <= camera[0] <= 1.15 and -0.15 <= camera[1] <= 1.15):
            continue
        x, y = camera[0] * (width - 1), camera[1] * (height - 1)
        radius = max(5.0, entry.tolerance / size[0] * (width - 1))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.line((x - 6, y, x + 6, y), fill=color, width=2)
        draw.line((x, y - 6, x, y + 6), fill=color, width=2)
        _label(draw, (x + radius + 2, y - 6), f"S{cluster['id']}", color)
    for click in entry.clicks:
        x, y = click["camera"][0] * (width - 1), click["camera"][1] * (height - 1)
        color = (255, 214, 102) if click["ambiguous"] else (128, 236, 208)
        if click["spot"] is None and not click["ambiguous"]:
            color = (205, 213, 218)
        draw.line((x - 7, y - 7, x + 7, y + 7,), fill=color, width=2)
        draw.line((x - 7, y + 7, x + 7, y - 7), fill=color, width=2)
        _label(draw, (x + 9, y + 2), click["label"], color)
    _draw_legend(draw, width, height,
                 [("S# circle: merged spot", (215, 225, 230)),
                  ("O has click", STATE_COLORS["matched"]),
                  ("O review (fits 2)", STATE_COLORS["review_overlap"]),
                  ("O inside, no click", STATE_COLORS["no_evidence"]),
                  ("X clicked point", (128, 236, 208)),
                  ("X review point", (255, 214, 102))])
    return image


def _draw_legend(draw, width, height, items):
    entries = [(text, color) for text, color in items]
    pad = 5
    line_height = 15
    longest = max(len(text) for text, _ in entries) * 6
    box_width = min(width - 2 * pad, longest + 2 * pad)
    box_height = pad * 2 + line_height * len(entries)
    draw.rectangle((pad, height - box_height - pad, pad + box_width,
                    height - pad), fill=(21, 35, 46))
    for row, (text, color) in enumerate(entries):
        y = height - box_height - pad + pad + row * line_height
        _label(draw, (pad + 4, y), text, color)


def _panel_overview(entry: CaptureEvidence, entries, clusters):
    native_width, native_height = entry.panel["native_width"], entry.panel["native_height"]
    width, height = scaled_dimensions(native_width, native_height, maximum=900, limit_area=560_000)
    image = Image.new("RGB", (width, height), (243, 247, 248))
    draw = ImageDraw.Draw(image)
    for step in range(1, 10):
        x = width * step / 10
        y = height * step / 10
        draw.line((x, 0, x, height), fill=(224, 232, 235))
        draw.line((0, y, width, y), fill=(224, 232, 235))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(83, 104, 115), width=2)
    for order, evidence in enumerate(entries):
        color = CAPTURE_COLORS[order % len(CAPTURE_COLORS)]
        for click in evidence.clicks:
            x = click["panel"][0] * (width - 1)
            y = click["panel"][1] * (height - 1)
            radius = max(2, round(width / 320))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            _label(draw, (x + radius + 2, y - 6), click["label"],
                   (45, 62, 72) if click["spot"] is not None or click["ambiguous"]
                   else (140, 150, 156))
        _label(draw, (10, 8 + order * 15),
               f"C{evidence.index} {_ascii(evidence.capture['title'], evidence.capture['filename'])}",
               color)
    for cluster in clusters:
        x = cluster["panel"][0] * (width - 1)
        y = cluster["panel"][1] * (height - 1)
        for order, evidence in enumerate(entries):
            radius = max(3.0, evidence.tolerance / native_width * (width - 1))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=CAPTURE_COLORS[order % len(CAPTURE_COLORS)], width=1)
        draw.line((x - 9, y, x + 9, y), fill=(25, 40, 48), width=2)
        draw.line((x, y - 9, x, y + 9), fill=(25, 40, 48), width=2)
        _label(draw, (x + 10, y + 2 + (cluster["id"] - 1) % 3 * 12), f"S{cluster['id']}", (25, 40, 48))
    _label(draw, (10, height - 20),
           "Dots: clicked points  |  rings: per-photo tolerance  |  cross: merged spot",
           (72, 88, 96))
    return image


def _capture_payload(entry: CaptureEvidence):
    return {
        "capture_id": entry.capture["id"],
        "capture_title": entry.capture["title"],
        "source_sha256": entry.capture["original"],
        "source_size": list(entry.size),
        "calibration_id": entry.revision["id"],
        "calibration_revision": entry.revision["revision"],
        "calibration_corners": entry.revision["corners"],
        "tolerance_panel_px": entry.tolerance,
        "points": [
            {
                "label": click["label"],
                "camera_normalized": [rounding(value, 8) for value in click["camera"]],
                "camera_px": list(click["camera_px"]),
                "panel": [rounding(value, 8) for value in click["panel"]],
                "assigned_spot": click["spot"],
                "candidate_spots": click["candidates"],
                "ambiguous": click["ambiguous"],
                "on_panel_edge": click["edge"],
            }
            for click in entry.clicks
        ],
    }


def run_recheck(store: Store, panel_id: str, data: dict):
    data = mapping(data, "亮点复检参数")
    raw_captures = sequence(data.get("captures"), "复检照片",
                            minimum=MIN_CAPTURES, maximum=MAX_CAPTURES)
    seen = set()
    entries = []
    for position, item in enumerate(raw_captures, start=1):
        capture_id = mapping(item, f"第 {position} 张照片").get("capture_id")
        if capture_id in seen:
            raise Problem("同一张照片不能在复检中选择两次。")
        seen.add(capture_id)
        entries.append(_load(store, panel_id, position, item))
    total_clicks = sum(len(entry.clicks) for entry in entries)
    if total_clicks > MAX_TOTAL_POINTS:
        raise Problem(f"一次复检最多保留 {MAX_TOTAL_POINTS} 个点位。")
    if total_clicks == 0:
        raise Problem("请至少在一张照片上点出可疑亮点。")
    size = (entries[0].panel["native_width"], entries[0].panel["native_height"])
    clusters = _consolidate(entries, size)
    observations = {
        cluster["id"]: {entry.index: _observe(entry, cluster, size) for entry in entries}
        for cluster in clusters
    }
    spot_rows = []
    multi = single = unconfirmed = 0
    review_observations = missing_observations = absent_observations = 0
    for cluster in clusters:
        states = observations[cluster["id"]]
        matched = sum(value["state"] == "matched" for value in states.values())
        review = sum(value["state"] == "review_overlap" for value in states.values())
        missing = sum(value["state"] == "no_evidence" for value in states.values())
        absent = sum(value["state"] == "not_captured" for value in states.values())
        review_observations += review
        missing_observations += missing
        absent_observations += absent
        if matched >= 2:
            status = "multi_capture"
            multi += 1
        elif matched == 1:
            status = "single_capture"
            single += 1
        else:
            status = "unconfirmed"
            unconfirmed += 1
        panel_x, panel_y = cluster["panel"]
        row = {
            "spot": cluster["id"],
            "panel_x": rounding(panel_x),
            "panel_y": rounding(panel_y),
            "panel_px": [rounding(panel_x * size[0], 2), rounding(panel_y * size[1], 2)],
            "panel_mm": None,
            "status": STATUS_TEXT[status],
            "confirmed_captures": matched,
            "review_observations": review,
            "no_evidence_observations": missing,
            "not_captured_observations": absent,
        }
        if entries[0].panel["width_mm"] is not None:
            row["panel_mm"] = [
                rounding(panel_x * entries[0].panel["width_mm"], 3),
                rounding(panel_y * entries[0].panel["height_mm"], 3),
            ]
        for entry in entries:
            value = states[entry.index]
            detail = STATE_TEXT[value["state"]]
            if value["distance"] is not None:
                detail += f" · {value['distance']:.2f} px"
            row[f"c{entry.index}"] = detail
        spot_rows.append(row)

    ambiguous_clicks = [click for entry in entries for click in entry.clicks if click["ambiguous"]]
    unattributed = [click for entry in entries for click in entry.clicks
                    if click["spot"] is None and not click["ambiguous"]]
    edge_clicks = [click for entry in entries for click in entry.clicks if click["edge"]]
    metrics = {
        "captures": len(entries),
        "clicks": total_clicks,
        "consolidated_spots": len(clusters),
        "multi_capture_spots": multi,
        "single_capture_spots": single,
        "unconfirmed_spots": unconfirmed,
        "ambiguous_clicks": len(ambiguous_clicks),
        "review_observations": review_observations,
        "no_evidence_observations": missing_observations,
        "not_captured_observations": absent_observations,
        "tolerances_panel_px": [rounding(entry.tolerance, 2) for entry in entries],
        "edge_clicks": len(edge_clicks),
    }
    summary = (
        f"{len(entries)} 张照片的 {total_clicks} 个点位归并为 {len(clusters)} 处面板位置；"
        f"{multi} 处由多张照片互证。"
    )
    report = Report("多图亮点位置归并", summary, metrics, spot_rows)
    click_rows = []
    for entry in entries:
        for click in entry.clicks:
            if click["ambiguous"]:
                state = "待复核重叠"
            elif click["spot"] is not None:
                state = "已归并"
            else:
                state = "未归并"
            click_rows.append({
                "capture": f"C{entry.index}",
                "point": click["label"],
                "camera_x": rounding(click["camera"][0], 6),
                "camera_y": rounding(click["camera"][1], 6),
                "camera_px_x": click["camera_px"][0],
                "camera_px_y": click["camera_px"][1],
                "panel_x": rounding(click["panel"][0], 6),
                "panel_y": rounding(click["panel"][1], 6),
                "tolerance_panel_px": rounding(click["tolerance"], 2),
                "assigned_spot": click["spot"],
                "candidate_spots": " ".join(str(value) for value in click["candidates"]),
                "state": state,
                "on_panel_edge": click["edge"],
            })
    calibration_payload = {
        "tool": "multi_capture_bright_spots",
        "panel": {
            "id": entries[0].panel["id"],
            "name": entries[0].panel["name"],
            "native_width": size[0],
            "native_height": size[1],
            "width_mm": entries[0].panel["width_mm"],
            "height_mm": entries[0].panel["height_mm"],
        },
        "captures": [_capture_payload(entry) for entry in entries],
        "spots": spot_rows,
        "ambiguous_clicks": [click["label"] for click in ambiguous_clicks],
        "unattributed_clicks": [click["label"] for click in unattributed],
    }
    report.add("recheck-calibrations.json",
               (json.dumps(calibration_payload, ensure_ascii=False, indent=2,
                           allow_nan=False) + "\n").encode("utf-8"),
               "application/json")
    report.table("recheck-spots.csv", spot_rows)
    report.table("recheck-clicks.csv", click_rows)
    for entry in entries:
        report.image(f"recheck-capture-{entry.index}.png",
                     _capture_overlay(entry, clusters, observations, size))
    report.image("recheck-panel.png", _panel_overview(entries[0], entries, clusters))
    if ambiguous_clicks:
        report.warnings.append(
            f"{len(ambiguous_clicks)} 个点位同时能对上两处归并位置，已单独留待复核，"
            "未计入任何一处。"
        )
    if unattributed:
        report.warnings.append(f"{len(unattributed)} 个点位没有落入任何归并位置，请复核。")
    if single:
        report.warnings.append(f"{single} 处亮点只在单张照片上有点位，缺少同面板互证。")
    if unconfirmed:
        report.warnings.append(f"{unconfirmed} 处归并位置没有唯一确认的点位，需人工复核。")
    if edge_clicks:
        report.warnings.append(f"{len(edge_clicks)} 个点位位于面板边缘 1% 内，请复核四角标定。")
    report.warnings.append(
        "误差范围按每张照片的面板物理像素填写；归并位置只作候选提示，不修改任何标定。"
    )
    snapshot_spots = []
    for row in spot_rows:
        states = observations[row["spot"]]
        snapshot_spots.append({
            "spot": row["spot"],
            "panel": [row["panel_x"], row["panel_y"]],
            "panel_px": row["panel_px"],
            "status": row["status"],
            "confirmed_captures": row["confirmed_captures"],
            "states": {f"c{entry.index}": states[entry.index]["state"] for entry in entries},
            "matched_points": {f"c{entry.index}": states[entry.index]["matched"]
                               for entry in entries},
        })
    snapshot = {
        "tool": "recheck",
        "panel_id": panel_id,
        "calibration_id": entries[0].revision["id"],
        "recheck_captures": [_capture_payload(entry) for entry in entries],
        "spots": snapshot_spots,
        "ambiguous_clicks": [click["label"] for click in ambiguous_clicks],
        "tolerances_panel_px": metrics["tolerances_panel_px"],
        "source": entries[0].capture["title"],
        "synthetic": entries[0].panel["synthetic"],
    }
    return entries, report, snapshot
