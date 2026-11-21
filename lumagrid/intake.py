"""Portable camera capture bundles for moving selections between local panel libraries."""
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile
import json
import re

from .geometry import Calibration
from .handoff import checked_path, evidence_filename, json_payload
from .model import CaptureSpec, Problem, mapping, plain, sequence, source_name
from .raster import MAX_SOURCE_BYTES, Asset, decode_asset
from .storage import Store


CAPTURE_PACKET = "lumagrid-capture-bundle-v1"
MAX_INTAKE_FILES = 20
MAX_CAPTURE_PACKET_BYTES = 26 * 1024 * 1024
MAX_MANIFEST = 180 * 1024
HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_SUFFIXES = {".png", ".jpeg", ".jpg", ".tiff", ".tif"}


@dataclass
class IntakeItem:
    spec: CaptureSpec
    asset: Asset
    calibration: dict | None
    packet_key: str
    source_id: str | None
    original_hash: str

    def report(self):
        return {
            "title": self.spec.title,
            "role": self.spec.role,
            "filename": self.spec.filename,
            "dimensions": [self.asset.width, self.asset.height],
            "role_settings": asdict(self.spec.exposure),
            "sha256": self.original_hash,
            "has_calibration": bool(self.calibration),
            "source_id": self.source_id,
            "source_key": self.packet_key,
        }


def manifest_packet(archive: ZipFile):
    try:
        headers = archive.infolist()
        actual = {}
        expanded = 0
        for entry in headers:
            if entry.is_dir() or entry.flag_bits & 0x1:
                raise Problem("采集包不能包含目录项或加密成员。")
            name = checked_path(entry.filename)
            if name in actual:
                raise Problem("采集包存在同名文件。")
            expanded += entry.file_size
            actual[name] = entry
        if not 2 <= len(actual) <= MAX_INTAKE_FILES + 1 or expanded > MAX_CAPTURE_PACKET_BYTES:
            raise Problem("采集包成员数或展开字节数超出导入上限。", 413)
        if "captures.json" not in actual or actual["captures.json"].file_size > MAX_MANIFEST:
            raise Problem("需要包含 captures.json，且采集清单须小于 180 KiB。")
        manifest = json.loads(archive.read("captures.json"))
    except (BadZipFile, UnicodeError, json.JSONDecodeError) as exc:
        raise Problem("无法读取采集包的 ZIP 或 JSON 清单。") from exc
    manifest = mapping(manifest, "采集包清单")
    if manifest.get("version") != CAPTURE_PACKET:
        raise Problem("不支持这个采集包版本。")
    if not isinstance(manifest.get("synthetic"), bool):
        raise Problem("采集包必须声明是否是合成材料。")
    members = sequence(manifest.get("captures"), "采集包成员", minimum=1, maximum=MAX_INTAKE_FILES)
    paths = set(actual) - {"captures.json"}
    return manifest, members, paths


def prepare_preview(raw: bytes) -> tuple[dict, list[IntakeItem]]:
    if not raw or len(raw) > MAX_CAPTURE_PACKET_BYTES:
        raise Problem("采集包应非空且不超过 26 MiB。", 413)
    try:
        with ZipFile(BytesIO(raw)) as archive:
            manifest, members, actual_paths = manifest_packet(archive)
            expected_paths = set()
            seen_spec = set()
            items = []
            for entry in members:
                record = mapping(entry, "采集成员信息")
                file_key = record.get("path")
                if not isinstance(file_key, str):
                    raise Problem("采集成员缺少路径。")
                file_key = checked_path(file_key)
                if file_key not in actual_paths or file_key in expected_paths:
                    raise Problem("采集成员路径与 ZIP 文件不一致，或被多次登记。")
                suffix = PurePosixPath(file_key).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    raise Problem("采集包成员应为 PNG、JPEG 或 TIFF。")
                expected_paths.add(file_key)
                data = archive.read(file_key)
                hash_value = record.get("sha256")
                checksum = sha256(data).hexdigest()
                if not isinstance(hash_value, str) or not HEX_256.fullmatch(hash_value) or checksum != hash_value:
                    raise Problem("采集包有成员与 SHA-256 清单不一致。")
                metadata = mapping(record.get("attributes"), "采集成员属性")
                filename = source_name(metadata.get("filename") or PurePosixPath(file_key).name)
                spec = CaptureSpec.parse(metadata, filename)
                binding = (checksum, spec.role)
                if binding in seen_spec:
                    raise Problem("采集包包含同角色且字节完全一致的重复采集。")
                seen_spec.add(binding)
                asset = decode_asset(data)
                calibration = None
                if record.get("calibration") is not None:
                    article = mapping(record["calibration"], "采集成员校准")
                    cal = Calibration.from_raw(article.get("corners"))
                    calibration = {
                        "corners": cal.corners.tolist(),
                        "overview": cal.overview((asset.height, asset.width)),
                        "note": plain(article.get("note"), "角点说明", 400),
                    }
                key = plain(record.get("id"), "来源采集标识", 64)
                items.append(IntakeItem(spec, asset, calibration, file_key, key or None, checksum))
            if expected_paths != actual_paths:
                raise Problem("采集包含有未登记的附件，尚未导入。")
    except (BadZipFile, KeyError, RuntimeError, OSError) as exc:
        raise Problem("无法核对采集包，尚未导入任何成员。") from exc
    return manifest, items


def save_prepared(store: Store, panel_id: str, manifest: dict, items: list[IntakeItem]):
    from .storage import transaction

    with store.connect() as conn, transaction(conn):
        panel_row = store.require_panel(conn, panel_id)
        if manifest["synthetic"] is True and not panel_row["synthetic"]:
            raise Problem("请将合成采集包导入标记为合成的面板。")
        ids = []
        for member in items:
            capture_id = store.insert_capture(conn, panel_id, member.spec, member.asset)
            ids.append(capture_id)
            if member.calibration:
                cal = member.calibration
                mark = cal["note"] or "继承采集包中的实际校准"
                store.insert_calibration(conn, capture_id, cal["corners"], cal["overview"], mark)
        return [store.capture_view(conn, store.require_capture(conn, uid)) for uid in ids]


def packet_attempt_summary(store: Store, panel_id: str, body: bytes, save: bool, expected_sha256=None):
    actual_sha256 = sha256(body).hexdigest()
    if save:
        if not isinstance(expected_sha256, str) or not HEX_256.fullmatch(expected_sha256):
            raise Problem("保存前须先核对原始采集包。")
        if actual_sha256 != expected_sha256:
            raise Problem("当前文件与预检的采集包不同，请重新核对成员。", 409)
    panel = store.panel(panel_id)
    manifest, candidates = prepare_preview(body)
    invalid_provenance = manifest["synthetic"] is True and not panel["synthetic"]
    catalog = {(capture["original"], capture["role"]): capture
               for capture in store.captures(panel_id)}
    review = []
    conflicts = 0
    for candidate in candidates:
        conflict = catalog.get((candidate.original_hash, candidate.spec.role))
        conflicts += conflict is not None
        review.append({
            **candidate.report(),
            "existing_capture": conflict["id"] if conflict else None,
            "existing_title": conflict["title"] if conflict else None,
        })
    packet_details = {
        "synthetic": manifest["synthetic"],
        "original_panel": plain(manifest.get("panel_name"), "来源面板名称", 80),
        "sha256": actual_sha256,
        "importable_count": 0 if invalid_provenance or conflicts else len(candidates),
        "conflict_count": conflicts,
        "provenance_conflict": invalid_provenance,
        "members": review,
    }
    if not save:
        return packet_details
    if conflicts:
        raise Problem("面板中已有同角色同字节的采集，整个采集包均未写入。", 409)
    result = save_prepared(store, panel_id, manifest, candidates)
    return {**packet_details, "imported": result}


def capture_members(store: Store, panel_id: str, selected: list[str]):
    if not 1 <= len(selected) <= MAX_INTAKE_FILES or len(set(selected)) != len(selected):
        raise Problem("请选择 1 至 20 张不同的采集。")
    captured = []
    cumulative = 0
    with store.connect() as conn:
        # Snapshot both immutable source references and the current calibration version.
        conn.execute("BEGIN")
        panel = store.panel_view(conn, store.require_panel(conn, panel_id))
        for capture_id in selected:
            row = store.require_capture(conn, capture_id)
            if row["panel_id"] != panel_id:
                raise Problem("同一个采集包内的图片必须属于当前面板。")
            capture = store.capture_view(conn, row)
            content, mime = store.read_blob(conn, row["original"])
            checksum = sha256(content).hexdigest()
            if checksum != row["original"]:
                raise Problem("库中采集数据与已登记哈希不一致。")
            cumulative += len(content)
            if cumulative + MAX_MANIFEST > MAX_CAPTURE_PACKET_BYTES:
                raise Problem("采集包超出 26 MiB，请减少大图。", 413)
            calibration = store.calibration(conn, capture_id)
            captured.append((capture, content, mime, calibration))
    return panel, captured


def package_captures(store: Store, panel_id: str, selected: list[str]):
    panel, collected = capture_members(store, panel_id, selected)
    metadata = []
    archive_content = []
    for i, (capture, payload, mime, cal) in enumerate(collected, start=1):
        suffix = {"image/tiff": ".tiff", "image/png": ".png", "image/jpeg": ".jpg"}.get(mime)
        if suffix is None:
            raise Problem("当前采集的 MIME 无法导出为采集包成员。")
        path = f"captures/{i:02d}-{capture['id'][:8]}{suffix}"
        attributes = {
            **capture["exposure"],
            "title": capture["title"],
            "role": capture["role"],
            "filename": capture["filename"],
            "note": capture["note"],
        }
        member = {
            "id": capture["id"],
            "path": path,
            "sha256": capture["original"],
            "attributes": attributes,
            "calibration": {"corners": cal["corners"], "note": cal["note"]} if cal else None,
        }
        metadata.append(member)
        archive_content.append((path, payload))
    manifest = {
        "version": CAPTURE_PACKET,
        "panel_name": panel["name"],
        "synthetic": panel["synthetic"],
        "native_size": [panel["native_width"], panel["native_height"]],
        "captures": metadata,
    }
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
        archive.writestr("captures.json", json_payload(manifest))
        for name, data in archive_content:
            archive.writestr(name, data)
    return evidence_filename(panel).replace("-evidence.zip", "-captures.zip"), output.getvalue()
