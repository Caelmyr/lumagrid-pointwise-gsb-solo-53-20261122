"""Portable evidence bundles with content identifiers and independently checkable manifests."""
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import PurePosixPath
import csv
import json
import re
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from .model import Problem, now
from .storage import Store


MAX_EXPORT_BYTES = 132 * 1024 * 1024
BUNDLE_VERSION = "lumagrid-evidence-v1"
SEPARATOR_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def json_payload(data) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def checked_path(name: str) -> str:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise Problem("材料文件名不合法。")
    if str(path) != name or len(name) > 180:
        raise Problem("材料文件名超出交接包约定。")
    return name


def safe_slug(text: str, fallback: str, maximum: int = 60) -> str:
    reduced = SEPARATOR_RE.sub("-", text).strip("-").lower()
    return (reduced or fallback)[:maximum]


def verified_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is None:
        raise Problem("交接记录编号不合法。")
    return value


def safe_csv_text(value):
    if isinstance(value, str) and value.lstrip("\t\r\n ").startswith(("=", "-", "+", "@")):
        return "'" + value
    return value


def csv_payload(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    columns = list(dict.fromkeys(label for row in rows for label in row))
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: safe_csv_text(value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8-sig")


@dataclass
class Bundle:
    contents: dict[str, bytes] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)
    size: int = 0

    def add(self, name: str, data: bytes, media_type: str):
        name = checked_path(name)
        digest = sha256(data).hexdigest()
        if name in self.contents:
            if self.metadata[name]["sha256"] == digest:
                return
            raise Problem("同名材料内容不同，无法生成可验证交接包。")
        if len(data) + self.size > MAX_EXPORT_BYTES:
            raise Problem("交接内容超过 132 MiB，请缩小记录选择，或取消包含原图。", 413)
        self.contents[name] = data
        self.metadata[name] = {"sha256": digest, "bytes": len(data), "media_type": media_type}
        self.size += len(data)

    def add_json(self, name: str, data):
        self.add(name, json_payload(data), "application/json")

    def to_zip(self, details: dict) -> bytes:
        manifest = {"version": BUNDLE_VERSION, "files": self.metadata, **details}
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
            records = {**self.contents, "manifest.json": json_payload(manifest)}
            for name, content in sorted(records.items()):
                info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content, compresslevel=4)
        return output.getvalue()


def extension(capture: dict) -> str:
    suffix = PurePosixPath(capture.get("filename", "")).suffix.lower()
    return suffix if suffix in {".png", ".jpeg", ".jpg", ".tif", ".tiff"} else ".img"


def source_registration(bundle: Bundle, capture: dict, calibration: dict | None, payload: bytes | None):
    source_hash = capture["original"]
    key = verified_id(capture["id"])
    if payload is not None and sha256(payload).hexdigest() != source_hash:
        raise Problem("原图哈希与项目引用不一致，交接已停止。")
    if payload is not None:
        path = f"originals/{source_hash}{extension(capture)}"
        bundle.add(path, payload, "application/octet-stream")
    else:
        path = next((name for name in bundle.contents if name.startswith(f"originals/{source_hash}.")), None)
    current_revision = calibration["id"] if calibration else "identity"
    record = {
        "capture_id": key,
        "title": capture["title"],
        "role": capture["role"],
        "width": capture["width"],
        "height": capture["height"],
        "exposure": capture["exposure"],
        "sha256": source_hash,
        "original_bundle_path": path,
        "calibration": calibration,
    }
    bundle.add_json(f"source-records/{key}-{current_revision}.json", record)
    return record


def evidence_filename(panel: dict) -> str:
    root_name = safe_slug(panel["name"], "panel")
    return f"lumagrid-{root_name}-{panel['id'][:8]}-evidence.zip"


def prepare_bundle(store: Store, panel_id: str, inspections: list[str], include_sources: bool):
    if not inspections or len(inspections) > 18:
        raise Problem("请从当前面板选择 1 至 18 项已复核检查。")
    if len(set(inspections)) != len(inspections):
        raise Problem("交接选择中有重复检查记录。")
    checked = [verified_id(key) for key in inspections]
    panel, groups = store.evidence_contents(panel_id, checked, include_sources)
    bundle = Bundle()
    created = now()
    meta = {
        "panel": panel,
        "synthetic": panel["synthetic"],
        "requested_source_inclusion": include_sources,
        "generated_at": created,
        "inspections": [],
    }
    tracking = []
    for group in groups:
        record = group["inspection"]
        key = verified_id(record["id"])
        prefix = f"inspection/{key}"
        bundle.add_json(f"{prefix}/inspection.json", record)
        bundle.add_json(f"{prefix}/decisions.json", group["decisions"])
        attachments = []
        for item in record["files"]:
            name = item["name"]
            if name not in group["files"]:
                raise Problem("检查记录的附件不完整，交接已停止。")
            content = group["files"][name]
            if sha256(content).hexdigest() != item["checksum"]:
                raise Problem("检查记录的附件校验失败，交接已停止。")
            target = f"{prefix}/attachments/{name}"
            bundle.add(target, content, item["mime"])
            attachments.append(target)
        sources = [source_registration(bundle, capture, calibration, payload)
                   for capture, calibration, payload in group["captures"]]
        sessions = [source_registration(bundle, capture, calibration, payload)
                    for capture, calibration, payload in group.get("sessions", [])]
        meta["inspections"].append({
            "id": key,
            "title": record["title"],
            "tool": record["tool"],
            "state": record["state"],
            "source_ids": [source["capture_id"] for source in sources + sessions],
            "attachments": attachments,
        })
        last = group["decisions"][-1] if group["decisions"] else {}
        tracking.append({
            "inspection_id": key, "title": record["title"], "tool": record["tool"],
            "source_image": sources[0]["title"] if sources else None,
            "source_sha256": sources[0]["sha256"] if sources else None,
            "last_reviewer": last.get("operator"),
            "reviewed_utc": last.get("created"),
            "synthetic": panel["synthetic"],
        })
    bundle.add("inspection-index.csv", csv_payload(tracking), "text/csv")
    bundle.add("readme.txt", usage_message(panel, include_sources), "text/plain")
    return evidence_filename(panel), bundle.to_zip(meta)


def usage_message(panel: dict, include_sources: bool):
    scope = "合成材料。仅用于调试和培训。" if panel["synthetic"] else "材料来自当前本地面板记录。"
    sources = "记录附带原图。" if include_sources else "未附带原图；来源记录含原始内容的 SHA-256。"
    lines = [
        "LumaGrid 面板检查交接包",
        scope,
        sources,
        "manifest.json 中的 files 对每个成员提供字节数、媒体类型和 SHA-256。",
        "inspection-index.csv 为查阅入口，inspection 中保留结果及复核决定顺序。",
        "source-records 中保留相机采集元数据以及检查时实际使用的四角版本。",
        "本包中的相对码值与图像检测记录不构成面板出厂认证。",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def inspect_bundle_content(data: bytes):
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 2400 or sum(item.file_size for item in entries) > MAX_EXPORT_BYTES + 2 ** 20:
                raise Problem("交接包超出离线验证的成员和展开体积上限。", 413)
            names = [checked_path(entry.filename) for entry in entries if not entry.is_dir()]
            if len(names) != len(entries) or len(names) != len(set(names)):
                raise Problem("交接包包含重复名称或额外目录。")
            manifest = json.loads(archive.read("manifest.json"))
            if not isinstance(manifest, dict) or manifest.get("version") != BUNDLE_VERSION:
                raise Problem("不支持该交接清单版本。")
            fields = manifest.get("files")
            if not isinstance(fields, dict) or set(fields) != set(names) - {"manifest.json"}:
                raise Problem("归档中实际文件与清单不一致。")
            rows = []
            for name, proof in fields.items():
                if not isinstance(proof, dict):
                    raise Problem("归档校验声明不是对象。")
                hasher = sha256()
                length = 0
                with archive.open(name) as entry:
                    for block in iter(lambda: entry.read(1024 * 1024), b""):
                        length += len(block)
                        if length > MAX_EXPORT_BYTES:
                            raise Problem("单一材料解包长度超出安全限制。", 413)
                        hasher.update(block)
                actual = hasher.hexdigest()
                path_parts = PurePosixPath(name).parts
                checksum_name = (
                    PurePosixPath(name).stem
                    if len(path_parts) == 2 and path_parts[0] == "originals" else None
                )
                matches_name = checksum_name is None or checksum_name == actual
                rows.append({
                    "path": name,
                    "bytes": length,
                    "sha256": actual,
                    "valid": matches_name and length == proof.get("bytes") and actual == proof.get("sha256"),
                })
    except (BadZipFile, OSError, ValueError, KeyError, RuntimeError) as exc:
        if isinstance(exc, Problem):
            raise
        raise Problem("读取交接包失败，或缺少 manifest.json。") from exc
    return {
        "manifest_version": BUNDLE_VERSION,
        "valid": all(row["valid"] for row in rows),
        "checked_files": len(rows),
        "synthetic": manifest.get("synthetic"),
        "failed": [row for row in rows if not row["valid"]],
    }
