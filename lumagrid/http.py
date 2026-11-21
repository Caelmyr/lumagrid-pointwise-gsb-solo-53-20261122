"""Local inspection service and version-preserving upload, analysis and review routes."""
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import os
import sqlite3

from flask import Flask, Response, request, send_file
from werkzeug.exceptions import HTTPException

from .calibration_tools import flatfield_report
from .demo import demo_trigger_log, install as install_demo
from .geometry import Calibration, segment_statistics
from .handoff import inspect_bundle_content, prepare_bundle
from .inspection import Scene, execute, public_tools
from .intake import MAX_CAPTURE_PACKET_BYTES, package_captures, packet_attempt_summary
from .model import (
    CAPTURE_ROLES, CaptureSpec, PanelSpec, Point, Problem, boolean, choice, mapping, numeric,
    plain, required_text, sequence,
)
from .raster import MAX_SOURCE_BYTES, decode_asset
from .storage import STATES, Store
from .timing import timing_report


def args_json():
    data = request.get_json(silent=True)
    return mapping(data, "请求体")


def preview_file(content, mime):
    return f"data:{mime};base64,{b64encode(content).decode('ascii')}"


def read_upload(field: str, limit=MAX_SOURCE_BYTES):
    file = request.files.get(field)
    if file is None or not file.filename:
        raise Problem("请选择要读取的本地文件。")
    content = file.stream.read(limit + 1)
    if len(content) > limit:
        raise Problem("文件超过该入口的上传限制。", 413)
    return file.filename, content


def stream_asset(content, mime, name, *, attachment=False):
    response = send_file(
        BytesIO(content), mimetype=mime, download_name=name, as_attachment=attachment,
        conditional=False, max_age=0,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "private, no-store"
    return response


def report_preview(report, metadata=None):
    available = []
    for name, (content, mime) in report.files.items():
        item = {"name": name, "mime": mime, "bytes": len(content)}
        if mime.startswith("image/"):
            item["preview"] = preview_file(content, mime)
        available.append(item)
    result = {**report.as_json(), "files": available}
    if metadata is not None:
        result["request"] = metadata
    return result


def same_scene(store, panel_id, capture_id, revision=None):
    scene = Scene.load(store, capture_id, revision)
    if scene.panel["id"] != panel_id:
        raise Problem("所选采集不属于当前面板。")
    return scene


def mark_point(scene: Scene, points: list, kind: str):
    source_width, source_height = scene.image.size
    if kind == "spot":
        point = points[0]
        from numpy import array
        px, py = scene.calibration.panel_to_camera(array([point.values()]))[0]
        return {
            "camera_normalized": [float(px), float(py)],
            "camera_px": [round(px * (source_width - 1), 3), round(py * (source_height - 1), 3)],
            "in_view": bool(0 <= px <= 1 and 0 <= py <= 1),
        }
    return segment_statistics(
        points[0], points[1], scene.panel, scene.calibration, source_width - 1, source_height - 1,
    )


def create_app(data_root: Path | None = None):
    static = Path(__file__).resolve().parent.parent / "static"
    app = Flask(__name__, static_folder=str(static), static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"] = 29 * 1024 * 1024
    root = data_root or Path(os.environ.get("LUMAGRID_DATA", "runtime-data"))
    store = Store(root)

    @app.after_request
    def secure_reply(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.errorhandler(Problem)
    def invalid_input(error):
        return {"error": str(error)}, error.status

    @app.errorhandler(HTTPException)
    def invalid_http(error):
        if error.code == 413:
            return {"error": "上传体积超过当前入口上限。"}, 413
        return {"error": "未找到这个页面或操作。" if error.code == 404 else error.description}, error.code

    @app.errorhandler(sqlite3.OperationalError)
    def database_busy(error):
        if "locked" not in str(error) and "busy" not in str(error):
            app.logger.exception("SQLite operation failed")
            return {"error": "数据存储发生错误。"}, 500
        return {"error": "当前资料库忙，请稍后重试。"}, 503

    @app.get("/")
    def index():
        response = app.send_static_file("index.html")
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/api/health")
    def health():
        return store.health()

    @app.get("/api/catalog")
    def catalog():
        return {"roles": CAPTURE_ROLES, "tools": public_tools(), "review_states": STATES}

    @app.get("/api/panels")
    def panel_list():
        return {"panels": store.panels(plain(request.args.get("q"), "筛选条件", 100))}

    @app.post("/api/panels")
    def panel_create():
        return store.create_panel(PanelSpec.parse(args_json())), 201

    @app.get("/api/panels/<panel_id>")
    def panel_detail(panel_id):
        return store.panel(panel_id)

    @app.post("/api/demo")
    def demo():
        return install_demo(store)

    @app.get("/api/demo/reference-log")
    def demo_log():
        return stream_asset(
            demo_trigger_log().encode("utf-8-sig"), "text/csv",
            "synthetic-trigger-reference.csv", attachment=False,
        )

    @app.get("/api/panels/<panel_id>/captures")
    def capture_list(panel_id):
        return {"captures": store.captures(panel_id)}

    @app.post("/api/panels/<panel_id>/captures")
    def capture_create(panel_id):
        filename, content = read_upload("image")
        spec = CaptureSpec.parse(request.form.to_dict(), filename)
        asset = decode_asset(content)
        parent = plain(request.form.get("parent_capture"), "上游采集", 64) or None
        return store.add_capture(panel_id, spec, asset, parent), 201

    @app.post("/api/panels/<panel_id>/intake")
    def intake(panel_id):
        _, payload = read_upload("bundle", limit=MAX_CAPTURE_PACKET_BYTES)
        save = request.form.get("save") == "1"
        expected = request.form.get("expected_sha256") if save else None
        report = packet_attempt_summary(store, panel_id, payload, save, expected)
        return report, 201 if save else 200

    @app.post("/api/panels/<panel_id>/capture-packet")
    def capture_packet(panel_id):
        ids = request.form.getlist("selected")
        filename, payload = package_captures(store, panel_id, ids)
        return stream_asset(payload, "application/zip", filename, attachment=True)

    @app.get("/api/captures/<capture_id>")
    def capture_detail(capture_id):
        return store.capture(capture_id)

    @app.get("/api/captures/<capture_id>/image/<variant>")
    def capture_image(capture_id, variant):
        capture, content, mime = store.capture_asset(capture_id, variant)
        is_original = variant == "original"
        name = capture["filename"] if is_original else "camera-preview.png"
        return stream_asset(content, mime, name, attachment=is_original)

    @app.get("/api/captures/<capture_id>/calibrations")
    def calibration_list(capture_id):
        return {"calibrations": store.calibration_history(capture_id)}

    @app.post("/api/captures/<capture_id>/calibrations")
    def calibration_create(capture_id):
        data = args_json()
        capture = store.capture(capture_id)
        cal = Calibration.from_raw(data.get("corners"))
        corners = cal.corners.tolist()
        note = plain(data.get("note"), "标定备注", 400)
        overview = cal.overview((capture["height"], capture["width"]))
        return store.add_calibration(capture_id, corners, overview, note), 201

    @app.get("/api/captures/<capture_id>/marks")
    def measurement_marks(capture_id):
        return {"marks": store.markers(capture_id)}

    @app.post("/api/captures/<capture_id>/marks")
    def mark_create(capture_id):
        data = args_json()
        kind = choice(data.get("kind"), {"spot", "segment"}, "标记类型", default="spot")
        rev = plain(data.get("calibration_id"), "标定版本", 64) or None
        scene = Scene.load(store, capture_id, rev)
        needed = 1 if kind == "spot" else 2
        raw_points = sequence(data.get("points"), "标记点", minimum=needed, maximum=needed)
        points = [Point.parse(value) for value in raw_points]
        if kind == "segment" and points[0] == points[1]:
            raise Problem("量测标记的两个端点不能重合。")
        label_text = required_text(data.get("label"), "标记名称", 70)
        info = mark_point(scene, points, kind)
        stored_rev = scene.revision["id"] if scene.revision else None
        return store.add_marker(
            capture_id, stored_rev, kind, [point.values() for point in points], label_text, info,
        ), 201

    @app.post("/api/captures/<capture_id>/analysis")
    def analysis(capture_id):
        data = args_json()
        primary, secondary, snapshot, report = execute(store, capture_id, data)
        if not boolean(data.get("save"), "保存检查", default=False):
            return report_preview(report, snapshot)
        title = plain(data.get("title"), "检查名称", 110, default=report.title)
        saved = store.add_inspection(
            primary.capture, secondary.capture if secondary else None,
            snapshot["tool"], title, snapshot, report.as_json(), report.files,
        )
        return {"record": saved, "result": report_preview(report, snapshot)}, 201

    @app.post("/api/panels/<panel_id>/timing")
    def timing(panel_id):
        data = args_json()
        captures = store.captures(panel_id)
        report = timing_report(captures, data.get("reference_csv"), data)
        if not boolean(data.get("save"), "保存检查", default=False):
            return report_preview(report)
        identifiers = {row.get("capture_id") for row in report.rows if row.get("state") == "paired"}
        primary = next((capture for capture in captures if capture["id"] in identifiers), None)
        if primary is None:
            raise Problem("没有配对结果可存档，先确认两份时间数据及容差。")
        snapshot = {
            "reference_csv": data["reference_csv"],
            "camera": data.get("camera"),
            "calibration_id": "identity",
            "mode": data.get("mode"),
            "offset_ms": report.metrics["clock_offset_ms"],
            "margin_ms": report.metrics["matching_tolerance_ms"],
            "level_margin": data.get("level_margin"),
        }
        report.add("reference-log.csv", data["reference_csv"].encode("utf-8"), "text/csv")
        record = store.add_inspection(
            primary, None, "timing", "采集时钟配对", snapshot, report.as_json(), report.files,
        )
        return {"record": record, "result": report_preview(report, snapshot)}, 201

    @app.post("/api/captures/<capture_id>/flatfield")
    def flatfield(capture_id):
        data = args_json()
        scene = Scene.load(store, capture_id, plain(data.get("calibration_id"), "标定版本", 64) or None)
        panel_id = scene.panel["id"]
        minimum = numeric(data.get("minimum"), "最低平场与暗场差值", .005, .3, default=.035)
        dark_id = required_text(data.get("dark"), "暗场采集", 64)
        flat_id = required_text(data.get("flat"), "平场采集", 64)
        if dark_id == flat_id:
            raise Problem("请分别选择暗场和平场采集。")
        dark = same_scene(store, panel_id, dark_id, "identity")
        flat = same_scene(store, panel_id, flat_id, "identity")
        computed, report = flatfield_report(scene, dark, flat, minimum)
        if not boolean(data.get("save"), "保存派生采集", default=False):
            return report_preview(report)
        parent_exposure = scene.capture["exposure"]
        title = plain(data.get("title"), "派生采集标题", 80, default="背景已校正")
        spec = CaptureSpec.parse({
            **parent_exposure,
            "role": scene.capture["role"],
            "title": title,
            "note": f"相机坐标暗平场校正；暗场 {dark_id}，平场 {flat_id}；无效区域透明。",
        }, "flatfield-" + capture_id[:8] + ".png")
        png = computed.camera_rgba()
        asset = decode_asset(png)
        output_capture = store.add_derived_capture(
            panel_id, spec, asset, capture_id, scene.revision,
        )
        return {"capture": output_capture, "result": report_preview(report)}, 201

    @app.get("/api/panels/<panel_id>/records")
    def record_list(panel_id):
        status = plain(request.args.get("state"), "复核状态", 40) or None
        if status is not None and status not in STATES:
            raise Problem("没有这个复核状态。")
        capture = plain(request.args.get("capture"), "采集筛选", 64) or None
        return {"records": store.inspections(panel_id, status, capture)}

    @app.get("/api/records/<inspection_id>")
    def record_detail(inspection_id):
        return {"record": store.inspection(inspection_id), "decisions": store.decisions(inspection_id)}

    @app.get("/api/records/<inspection_id>/files/<name>")
    def record_file(inspection_id, name):
        content, mime = store.inspection_file(inspection_id, name)
        attachment = mime != "image/png" or request.args.get("download") == "1"
        return stream_asset(content, mime, name, attachment=attachment)

    @app.post("/api/records/<inspection_id>/decisions")
    def decision_write(inspection_id):
        data = args_json()
        state = choice(data.get("state"), STATES, "复核结果", default="verified")
        operator = required_text(data.get("operator"), "复核人", 80)
        note = plain(data.get("note"), "复核说明", 600)
        old = required_text(data.get("expected"), "原复核结果", 40)
        return store.decide(inspection_id, state, operator, note, old)

    @app.post("/api/panels/<panel_id>/handoff")
    def handoff(panel_id):
        if request.is_json:
            data = args_json()
            ids = sequence(data.get("records"), "检查清单", minimum=1, maximum=18)
            include = boolean(data.get("includes_sources"), "包含原图", default=True)
        else:
            ids = request.form.getlist("record")
            include = request.form.get("includes_sources") == "1"
        filename, payload = prepare_bundle(store, panel_id, ids, include)
        return stream_asset(payload, "application/zip", filename, attachment=True)

    @app.post("/api/handoff/inspect")
    def handoff_inspect():
        _, content = read_upload("archive", limit=28 * 1024 * 1024)
        return inspect_bundle_content(content)

    return app
