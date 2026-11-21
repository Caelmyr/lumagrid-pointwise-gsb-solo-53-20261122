from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
from uuid import uuid4

from .model import CaptureSpec, PanelSpec, Problem, now


SCHEMA = """
CREATE TABLE IF NOT EXISTS panels (
 id TEXT PRIMARY KEY,
 name TEXT NOT NULL,
 station TEXT NOT NULL,
 native_width INTEGER NOT NULL,
 native_height INTEGER NOT NULL,
 width_mm REAL,
 height_mm REAL,
 note TEXT NOT NULL,
 synthetic INTEGER NOT NULL,
 seed_key TEXT UNIQUE,
 created TEXT NOT NULL,
 updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (
 checksum TEXT PRIMARY KEY,
 content BLOB NOT NULL,
 mime TEXT NOT NULL,
 size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS captures (
 id TEXT PRIMARY KEY,
 panel_id TEXT NOT NULL REFERENCES panels(id),
 title TEXT NOT NULL,
 role TEXT NOT NULL,
 exposure TEXT NOT NULL,
 note TEXT NOT NULL,
 filename TEXT NOT NULL,
 original TEXT NOT NULL REFERENCES blobs(checksum),
 preview TEXT NOT NULL REFERENCES blobs(checksum),
 width INTEGER NOT NULL,
 height INTEGER NOT NULL,
 mode TEXT NOT NULL,
 description TEXT NOT NULL,
 parent_capture TEXT REFERENCES captures(id),
 created TEXT NOT NULL,
 UNIQUE(panel_id, original, role)
);
CREATE TABLE IF NOT EXISTS calibrations (
 id TEXT PRIMARY KEY,
 capture_id TEXT NOT NULL REFERENCES captures(id),
 revision INTEGER NOT NULL,
 corners TEXT NOT NULL,
 info TEXT NOT NULL,
 note TEXT NOT NULL,
 created TEXT NOT NULL,
 UNIQUE(capture_id, revision)
);
CREATE TABLE IF NOT EXISTS inspections (
 id TEXT PRIMARY KEY,
 panel_id TEXT NOT NULL REFERENCES panels(id),
 capture_id TEXT NOT NULL REFERENCES captures(id),
 compare_id TEXT REFERENCES captures(id),
 tool TEXT NOT NULL,
 title TEXT NOT NULL,
 request TEXT NOT NULL,
 report TEXT NOT NULL,
 state TEXT NOT NULL,
 created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inspection_files (
 inspection_id TEXT NOT NULL REFERENCES inspections(id),
 name TEXT NOT NULL,
 checksum TEXT NOT NULL REFERENCES blobs(checksum),
 PRIMARY KEY(inspection_id, name)
);
CREATE TABLE IF NOT EXISTS decisions (
 id TEXT PRIMARY KEY,
 inspection_id TEXT NOT NULL REFERENCES inspections(id),
 old_state TEXT NOT NULL,
 new_state TEXT NOT NULL,
 operator TEXT NOT NULL,
 note TEXT NOT NULL,
 created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS markers (
 id TEXT PRIMARY KEY,
 panel_id TEXT NOT NULL REFERENCES panels(id),
 capture_id TEXT NOT NULL REFERENCES captures(id),
 calibration_id TEXT REFERENCES calibrations(id),
 kind TEXT NOT NULL,
 coordinates TEXT NOT NULL,
 label TEXT NOT NULL,
 info TEXT NOT NULL,
 created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS capture_panel ON captures(panel_id, created DESC);
CREATE INDEX IF NOT EXISTS inspection_panel ON inspections(panel_id, created DESC);
CREATE INDEX IF NOT EXISTS markers_capture ON markers(capture_id, created DESC);
CREATE INDEX IF NOT EXISTS decisions_inspection ON decisions(inspection_id, created);
PRAGMA user_version=1;
"""

MAX_PANELS = 150
MAX_CAPTURES = 90
MAX_INSPECTIONS = 180
MAX_MARKERS = 400
MAX_LIBRARY_BYTES = 2 * 1024 * 1024 * 1024
STATES = {"pending": "待复核", "verified": "已复核", "returned": "需补拍"}


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def immutable_id() -> str:
    return str(uuid4())


def row_json(row: sqlite3.Row, *, decode: tuple[str, ...] = ()) -> dict:
    value = dict(row)
    for field in decode:
        if value.get(field) is not None:
            value[field] = json.loads(value[field])
    return value


def require_row(row, label: str):
    if row is None:
        raise Problem(f"找不到{label}。", 404)
    return row


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


class Store:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "lumagrid.sqlite3"
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("不兼容的工作库版本")
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            yield conn
        finally:
            conn.close()

    def health(self) -> dict:
        with self.connect() as conn:
            conn.execute("SELECT id FROM panels LIMIT 1").fetchone()
        return {"status": "ok", "schema": 1}

    def panels(self, search: str = "") -> list[dict]:
        with self.connect() as conn:
            if search:
                escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                token = f"%{escaped}%"
                matches = conn.execute(
                    "SELECT * FROM panels WHERE name LIKE ? ESCAPE '\\' OR station LIKE ? ESCAPE '\\' "
                    "ORDER BY updated DESC, id", (token, token)
                ).fetchall()
            else:
                matches = conn.execute("SELECT * FROM panels ORDER BY updated DESC, id").fetchall()
            return [self.panel_view(conn, row) for row in matches]

    def panel(self, panel_id: str) -> dict:
        with self.connect() as conn:
            return self.panel_view(conn, self.require_panel(conn, panel_id))

    @staticmethod
    def require_panel(conn, panel_id: str):
        return require_row(conn.execute("SELECT * FROM panels WHERE id = ?", (panel_id,)).fetchone(), "面板")

    @staticmethod
    def panel_view(conn, row) -> dict:
        result = dict(row)
        result["synthetic"] = bool(result["synthetic"])
        result["capture_count"] = conn.execute(
            "SELECT count(*) FROM captures WHERE panel_id = ?", (row["id"],)
        ).fetchone()[0]
        result["record_count"] = conn.execute(
            "SELECT count(*) FROM inspections WHERE panel_id = ?", (row["id"],)
        ).fetchone()[0]
        return result

    def create_panel(self, spec: PanelSpec) -> dict:
        with self.connect() as conn, transaction(conn):
            panel_id = self.insert_panel(conn, spec)
            return self.panel_view(conn, self.require_panel(conn, panel_id))

    @staticmethod
    def insert_panel(conn, spec: PanelSpec, seed_key: str | None = None) -> str:
        if conn.execute("SELECT count(*) FROM panels").fetchone()[0] >= MAX_PANELS:
            raise Problem(f"本工作库最多保存 {MAX_PANELS} 个面板。", 409)
        panel_id = immutable_id()
        clock = now()
        conn.execute(
            "INSERT INTO panels VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (panel_id, spec.name, spec.station, spec.native_width, spec.native_height,
             spec.width_mm, spec.height_mm, spec.note, int(spec.synthetic), seed_key, clock, clock),
        )
        return panel_id

    @staticmethod
    def touch(conn, panel_id: str) -> None:
        conn.execute("UPDATE panels SET updated = ? WHERE id = ?", (now(), panel_id))

    @staticmethod
    def blob(conn, data: bytes, mime: str) -> str:
        checksum = sha256(data).hexdigest()
        if conn.execute("SELECT 1 FROM blobs WHERE checksum = ?", (checksum,)).fetchone():
            return checksum
        used = conn.execute("SELECT coalesce(sum(size), 0) FROM blobs").fetchone()[0]
        if used + len(data) > MAX_LIBRARY_BYTES:
            raise Problem("工作库可用空间不足，已达到 2 GiB 的材料限额。", 409)
        conn.execute("INSERT INTO blobs VALUES (?, ?, ?, ?)", (checksum, data, mime, len(data)))
        return checksum

    @staticmethod
    def read_blob(conn, checksum: str) -> tuple[bytes, str]:
        row = require_row(conn.execute(
            "SELECT content, mime FROM blobs WHERE checksum = ?", (checksum,)
        ).fetchone(), "图像内容")
        return row["content"], row["mime"]

    def add_capture(self, panel_id: str, spec: CaptureSpec, asset, parent_id: str | None) -> dict:
        with self.connect() as conn, transaction(conn):
            self.require_panel(conn, panel_id)
            capture_id = self.insert_capture(conn, panel_id, spec, asset, parent_id)
            return self.capture_view(conn, self.require_capture(conn, capture_id))

    def add_derived_capture(self, panel_id: str, spec: CaptureSpec, asset, parent_id: str,
                            calibration: dict | None) -> dict:
        with self.connect() as conn, transaction(conn):
            self.require_panel(conn, panel_id)
            new_id = self.insert_capture(conn, panel_id, spec, asset, parent_id)
            if calibration:
                self.insert_calibration(
                    conn, new_id, calibration["corners"], calibration["info"],
                    "继承同一相机栅格上的父采集校准",
                )
            return self.capture_view(conn, self.require_capture(conn, new_id))

    def insert_capture(self, conn, panel_id: str, spec: CaptureSpec, asset,
                       parent_id: str | None = None) -> str:
        if conn.execute("SELECT count(*) FROM captures WHERE panel_id = ?", (panel_id,)).fetchone()[0] >= MAX_CAPTURES:
            raise Problem(f"每个面板最多保存 {MAX_CAPTURES} 张采集。", 409)
        if parent_id:
            parent = self.require_capture(conn, parent_id)
            if parent["panel_id"] != panel_id:
                raise Problem("派生采集只能保留在同一面板中。")
        original = self.blob(conn, asset.original, asset.mime)
        existing = conn.execute(
            "SELECT id FROM captures WHERE panel_id = ? AND original = ? AND role = ?",
            (panel_id, original, spec.role),
        ).fetchone()
        if existing:
            raise Problem("该面板在相同角色中已有字节完全一致的采集。", 409)
        preview = self.blob(conn, asset.preview, "image/png")
        capture_id = immutable_id()
        conn.execute(
            "INSERT INTO captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (capture_id, panel_id, spec.title, spec.role, encode(asdict(spec.exposure)), spec.note,
             spec.filename, original, preview, asset.width, asset.height, asset.mode,
             encode(asset.description), parent_id, now()),
        )
        self.touch(conn, panel_id)
        return capture_id

    def captures(self, panel_id: str) -> list[dict]:
        with self.connect() as conn:
            self.require_panel(conn, panel_id)
            records = conn.execute(
                "SELECT * FROM captures WHERE panel_id = ? ORDER BY created DESC, rowid DESC", (panel_id,)
            ).fetchall()
            return [self.capture_view(conn, row) for row in records]

    @staticmethod
    def require_capture(conn, capture_id: str):
        return require_row(conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone(), "采集图像")

    @staticmethod
    def capture_view(conn, row) -> dict:
        capture = row_json(row, decode=("exposure", "description"))
        calibration = conn.execute(
            "SELECT id, revision FROM calibrations WHERE capture_id = ? ORDER BY revision DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        capture["calibration"] = dict(calibration) if calibration else None
        return capture

    def capture(self, capture_id: str) -> dict:
        with self.connect() as conn:
            return self.capture_view(conn, self.require_capture(conn, capture_id))

    def capture_asset(self, capture_id: str, variant: str) -> tuple[dict, bytes, str]:
        if variant not in {"original", "preview"}:
            raise Problem("未知的采集图版本。", 404)
        with self.connect() as conn:
            row = self.require_capture(conn, capture_id)
            content, mime = self.read_blob(conn, row[variant])
            return self.capture_view(conn, row), content, mime

    def add_calibration(self, capture_id: str, corners: list, info: dict, note: str) -> dict:
        with self.connect() as conn, transaction(conn):
            self.require_capture(conn, capture_id)
            calibration_id = self.insert_calibration(conn, capture_id, corners, info, note)
            return self.calibration(conn, capture_id, calibration_id)

    def insert_calibration(self, conn, capture_id: str, corners: list, info: dict, note: str) -> str:
        capture = self.require_capture(conn, capture_id)
        revision = conn.execute(
            "SELECT coalesce(max(revision), 0) + 1 FROM calibrations WHERE capture_id = ?", (capture_id,)
        ).fetchone()[0]
        if revision > 90:
            raise Problem("该采集已保存 90 个角点版本。", 409)
        uid = immutable_id()
        conn.execute(
            "INSERT INTO calibrations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, capture_id, revision, encode(corners), encode(info), note, now()),
        )
        self.touch(conn, capture["panel_id"])
        return uid

    def calibration_history(self, capture_id: str) -> list[dict]:
        with self.connect() as conn:
            self.require_capture(conn, capture_id)
            rows = conn.execute(
                "SELECT * FROM calibrations WHERE capture_id = ? ORDER BY revision DESC", (capture_id,)
            ).fetchall()
            return [row_json(row, decode=("corners", "info")) for row in rows]

    @staticmethod
    def calibration(conn, capture_id: str, calibration_id: str | None = None) -> dict | None:
        if calibration_id == "identity":
            return None
        if calibration_id:
            row = require_row(conn.execute(
                "SELECT * FROM calibrations WHERE id = ? AND capture_id = ?", (calibration_id, capture_id)
            ).fetchone(), "指定的角点校准")
        else:
            row = conn.execute(
                "SELECT * FROM calibrations WHERE capture_id = ? ORDER BY revision DESC LIMIT 1", (capture_id,)
            ).fetchone()
        return row_json(row, decode=("corners", "info")) if row else None

    def read_for_analysis(self, capture_id: str, calibration_id: str | None) -> tuple[dict, dict, bytes, dict | None]:
        with self.connect() as conn:
            conn.execute("BEGIN")
            capture = self.capture_view(conn, self.require_capture(conn, capture_id))
            panel = self.panel_view(conn, self.require_panel(conn, capture["panel_id"]))
            original, _ = self.read_blob(conn, capture["original"])
            calibration = self.calibration(conn, capture_id, calibration_id)
            return panel, capture, original, calibration

    def find_seed(self, key: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM panels WHERE seed_key = ?", (key,)).fetchone()
            return self.panel_view(conn, row) if row else None

    def install_demo(self, key: str, panel: PanelSpec, assets: list) -> dict:
        with self.connect() as conn, transaction(conn):
            stored = conn.execute("SELECT * FROM panels WHERE seed_key = ?", (key,)).fetchone()
            if stored:
                return self.panel_view(conn, stored)
            panel_id = self.insert_panel(conn, panel, key)
            for spec, asset, calibration in assets:
                capture_id = self.insert_capture(conn, panel_id, spec, asset)
                if calibration:
                    self.insert_calibration(conn, capture_id, calibration["corners"], calibration, "合成几何参照")
            return self.panel_view(conn, self.require_panel(conn, panel_id))

    def add_inspection(self, capture: dict, comparison: dict | None, tool: str, title: str,
                       request: dict, report: dict, files: dict[str, tuple[bytes, str]]) -> dict:
        with self.connect() as conn, transaction(conn):
            panel_id = capture["panel_id"]
            if comparison and comparison["panel_id"] != panel_id:
                raise Problem("对照图不在当前面板中。")
            if conn.execute("SELECT count(*) FROM inspections WHERE panel_id = ?", (panel_id,)).fetchone()[0] >= MAX_INSPECTIONS:
                raise Problem(f"每个面板最多保存 {MAX_INSPECTIONS} 项检查。", 409)
            uid = immutable_id()
            conn.execute(
                "INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, panel_id, capture["id"], comparison["id"] if comparison else None, tool,
                 title, encode(request), encode(report), "pending", now()),
            )
            for name, (data, mime) in files.items():
                checksum = self.blob(conn, data, mime)
                conn.execute("INSERT INTO inspection_files VALUES (?, ?, ?)", (uid, name, checksum))
            self.touch(conn, panel_id)
            return self.inspection_view(conn, self.require_inspection(conn, uid))

    @staticmethod
    def require_inspection(conn, inspection_id: str):
        return require_row(conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone(), "检查记录")

    @staticmethod
    def inspection_view(conn, row) -> dict:
        result = row_json(row, decode=("request", "report"))
        files = conn.execute(
            "SELECT f.name, f.checksum, b.mime, b.size FROM inspection_files f "
            "JOIN blobs b ON f.checksum = b.checksum WHERE inspection_id = ? ORDER BY f.name",
            (row["id"],),
        ).fetchall()
        result["files"] = [dict(file) for file in files]
        return result

    def inspection(self, inspection_id: str) -> dict:
        with self.connect() as conn:
            return self.inspection_view(conn, self.require_inspection(conn, inspection_id))

    def inspections(self, panel_id: str, state: str | None, capture_id: str | None) -> list[dict]:
        filters = ["panel_id = ?"]
        values = [panel_id]
        if state:
            filters.append("state = ?")
            values.append(state)
        if capture_id:
            filters.append("capture_id = ?")
            values.append(capture_id)
        where = " AND ".join(filters)
        with self.connect() as conn:
            self.require_panel(conn, panel_id)
            rows = conn.execute(
                "SELECT id, panel_id, capture_id, compare_id, tool, title, state, created "
                f"FROM inspections WHERE {where} ORDER BY created DESC, rowid DESC", values
            ).fetchall()
            return [dict(row) for row in rows]

    def inspection_file(self, inspection_id: str, name: str) -> tuple[bytes, str]:
        with self.connect() as conn:
            file = require_row(conn.execute(
                "SELECT checksum FROM inspection_files WHERE inspection_id = ? AND name = ?",
                (inspection_id, name),
            ).fetchone(), "检查附件")
            return self.read_blob(conn, file["checksum"])

    def decide(self, inspection_id: str, state: str, operator: str, note: str, expected: str) -> dict:
        with self.connect() as conn, transaction(conn):
            inspection = self.require_inspection(conn, inspection_id)
            previous = inspection["state"]
            if expected != previous:
                raise Problem("检查记录已由其他页面更新，请重新读取当前状态。", 409)
            if state == previous:
                raise Problem("请选择不同的复核结果。")
            if not note and state != "verified":
                raise Problem("请填写补拍或退回原因。")
            clock = now()
            conn.execute("UPDATE inspections SET state = ? WHERE id = ?", (state, inspection_id))
            conn.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (immutable_id(), inspection_id, previous, state, operator, note, clock),
            )
            self.touch(conn, inspection["panel_id"])
            return self.inspection_view(conn, self.require_inspection(conn, inspection_id))

    @staticmethod
    def decision_rows(conn, inspection_id: str) -> list[dict]:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM decisions WHERE inspection_id = ? ORDER BY created, rowid", (inspection_id,)
        ).fetchall()]

    def decisions(self, inspection_id: str) -> list[dict]:
        with self.connect() as conn:
            self.require_inspection(conn, inspection_id)
            return self.decision_rows(conn, inspection_id)

    def add_marker(self, capture_id: str, calibration_id: str | None, kind: str, coordinates: list,
                   label: str, info: dict) -> dict:
        with self.connect() as conn, transaction(conn):
            capture = self.require_capture(conn, capture_id)
            panel_id = capture["panel_id"]
            if conn.execute("SELECT count(*) FROM markers WHERE panel_id = ?", (panel_id,)).fetchone()[0] >= MAX_MARKERS:
                raise Problem(f"每个面板最多保存 {MAX_MARKERS} 项图像标记。", 409)
            uid = immutable_id()
            conn.execute(
                "INSERT INTO markers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, panel_id, capture_id, calibration_id, kind, encode(coordinates), label, encode(info), now()),
            )
            self.touch(conn, panel_id)
            row = conn.execute("SELECT * FROM markers WHERE id = ?", (uid,)).fetchone()
            return row_json(row, decode=("coordinates", "info"))

    def markers(self, capture_id: str) -> list[dict]:
        with self.connect() as conn:
            self.require_capture(conn, capture_id)
            rows = conn.execute(
                "SELECT * FROM markers WHERE capture_id = ? ORDER BY created DESC, rowid DESC", (capture_id,)
            ).fetchall()
            return [row_json(row, decode=("coordinates", "info")) for row in rows]

    def evidence_contents(self, panel_id: str, ids: list[str], include_sources: bool) -> tuple[dict, list[dict]]:
        result = []
        with self.connect() as conn:
            conn.execute("BEGIN")
            panel = self.panel_view(conn, self.require_panel(conn, panel_id))
            for inspection_id in ids:
                row = self.require_inspection(conn, inspection_id)
                if row["panel_id"] != panel_id or row["state"] != "verified":
                    raise Problem("交接包只支持当前面板已复核的记录。")
                inspection = self.inspection_view(conn, row)
                captures = []
                seen = set()
                for key, calibration_key in (("capture_id", "calibration_id"), ("compare_id", "comparison_calibration_id")):
                    if not row[key]:
                        continue
                    capture = self.capture_view(conn, self.require_capture(conn, row[key]))
                    checksum = capture["original"]
                    calibration_id = inspection["request"].get(calibration_key) or "identity"
                    calibration = self.calibration(conn, row[key], calibration_id)
                    actual = self.read_blob(conn, checksum)[0] if include_sources and checksum not in seen else None
                    seen.add(checksum)
                    captures.append((capture, calibration, actual))
                files = {item["name"]: self.read_blob(conn, item["checksum"])[0] for item in inspection["files"]}
                result.append({
                    "inspection": inspection,
                    "files": files,
                    "captures": captures,
                    "decisions": self.decision_rows(conn, inspection_id),
                })
            return panel, result
