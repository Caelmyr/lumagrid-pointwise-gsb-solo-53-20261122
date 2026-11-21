from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any


class Problem(ValueError):
    """Input or workflow error presented to the inspection operator."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


CAPTURE_ROLES = {
    "uniform": "均匀灰场",
    "black": "低灰 / 黑场",
    "white": "白场",
    "checker": "棋盘格",
    "ramp": "灰阶",
    "edge": "斜边",
    "macro": "次像素近摄",
    "reference": "参考条件",
    "dark": "镜头暗场",
    "flat": "镜头平场",
    "other": "其他",
}
CONTROLLED_ROLES = {"dark", "flat"}
PNG_VERSION = 1


def plain(value: Any, label: str, limit: int, *, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise Problem(f"{label}应为文本。")
    value = value.strip()
    if any(char in value for char in "\x00\r"):
        raise Problem(f"{label}含控制字符。")
    if len(value) > limit:
        raise Problem(f"{label}最多 {limit} 个字符。")
    return value or default


def required_text(value: Any, label: str, limit: int) -> str:
    result = plain(value, label, limit)
    if not result:
        raise Problem(f"请填写{label}。")
    return result


def numeric(
    value: Any,
    label: str,
    lo: float,
    hi: float,
    *,
    default: float | None = None,
) -> float:
    if value is None or value == "":
        if default is not None:
            return default
        raise Problem(f"请填写{label}。")
    if isinstance(value, bool):
        raise Problem(f"{label}应为数字。")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise Problem(f"{label}应为数字。") from None
    if not isfinite(result) or not lo <= result <= hi:
        raise Problem(f"{label}应在 {lo:g} 至 {hi:g} 之间。")
    return result


def integral(value: Any, label: str, lo: int, hi: int, *, default: int | None = None) -> int:
    result = numeric(value, label, lo, hi, default=default)
    if int(result) != result:
        raise Problem(f"{label}应为整数。")
    return int(result)


def mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise Problem(f"{label}应为对象。")
    return value


def sequence(value: Any, label: str, *, minimum: int, maximum: int) -> list:
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise Problem(f"{label}须包含 {minimum} 至 {maximum} 项。")
    return list(value)


def boolean(value: Any, label: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise Problem(f"{label}应是开关。")


def choice(value: Any, choices: dict | set | tuple, label: str, *, default: str) -> str:
    if value is None or value == "":
        return default
    if not isinstance(value, str) or value not in choices:
        raise Problem(f"{label}没有这个选项。")
    return value


def optional_numeric(value: Any, label: str, lo: float, hi: float) -> float | None:
    if value is None or value == "":
        return None
    return numeric(value, label, lo, hi)


def source_name(value: Any) -> str:
    name = required_text(value, "文件名", 200).replace("\\", "/").split("/")[-1]
    if name in {"", ".", ".."} or "\n" in name:
        raise Problem("文件名不合法。")
    return name


def recorded_time(value: Any) -> datetime | None:
    if not value:
        return None
    value = plain(value, "采集时间", 50)
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise Problem("采集时间请使用带时区的 ISO 8601 时间。") from None
    if stamp.utcoffset() is None:
        raise Problem("采集时间缺少时区。")
    return stamp.astimezone(timezone.utc)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="milliseconds") if value else None


def filename_title(name: str) -> str:
    return name.rsplit(".", 1)[0][:80]


@dataclass(frozen=True)
class PanelSpec:
    name: str
    station: str
    native_width: int
    native_height: int
    width_mm: float | None
    height_mm: float | None
    note: str
    synthetic: bool

    @classmethod
    def parse(cls, data: dict) -> "PanelSpec":
        data = mapping(data, "面板属性")
        width_mm = optional_numeric(data.get("width_mm"), "有效宽度", 1, 20_000)
        height_mm = optional_numeric(data.get("height_mm"), "有效高度", 1, 20_000)
        if (width_mm is None) != (height_mm is None):
            raise Problem("有效宽和高需同时填写。")
        return cls(
            name=required_text(data.get("name"), "面板编号", 80),
            station=plain(data.get("station"), "工位", 80),
            native_width=integral(data.get("native_width"), "物理像素宽", 16, 16_384),
            native_height=integral(data.get("native_height"), "物理像素高", 16, 16_384),
            width_mm=width_mm,
            height_mm=height_mm,
            note=plain(data.get("note"), "备注", 1000),
            synthetic=boolean(data.get("synthetic"), "合成素材"),
        )


@dataclass(frozen=True)
class Exposure:
    shutter_ms: float | None
    iso: float | None
    angle_deg: float | None
    level: float | None
    camera: str
    time: str | None

    @classmethod
    def parse(cls, data: dict) -> "Exposure":
        return cls(
            shutter_ms=optional_numeric(data.get("shutter_ms"), "快门时长", 0.001, 360_000),
            iso=optional_numeric(data.get("iso"), "相机 ISO", 1, 1_000_000),
            angle_deg=optional_numeric(data.get("angle_deg"), "观察角", -90, 90),
            level=optional_numeric(data.get("level"), "图案亮度", 0, 1),
            camera=plain(data.get("camera"), "相机", 80),
            time=iso(recorded_time(data.get("captured_at", data.get("time")))),
        )


@dataclass(frozen=True)
class CaptureSpec:
    title: str
    role: str
    exposure: Exposure
    note: str
    filename: str

    @classmethod
    def parse(cls, data: dict, filename: str) -> "CaptureSpec":
        data = mapping(data, "采集属性")
        normalized_name = source_name(filename)
        return cls(
            title=plain(data.get("title"), "采集名称", 80, default=filename_title(normalized_name)),
            role=choice(data.get("role"), CAPTURE_ROLES, "采集角色", default="other"),
            exposure=Exposure.parse(data),
            note=plain(data.get("note"), "备注", 1000),
            filename=normalized_name,
        )


@dataclass(frozen=True)
class Region:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def parse(cls, data: Any, *, minimum: float = 0.01) -> "Region":
        if data is None:
            return cls(0, 0, 1, 1)
        coords = sequence(data, "选区", minimum=4, maximum=4)
        values = [numeric(item, "选区坐标", 0, 1) for item in coords]
        left, top, right, bottom = values
        if right - left < minimum or bottom - top < minimum:
            raise Problem("选区太小，或起止位置颠倒。")
        return cls(*values)

    def values(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = max(0, min(width - 1, int(self.left * width)))
        y0 = max(0, min(height - 1, int(self.top * height)))
        x1 = max(x0 + 1, min(width, int(self.right * width + 0.5)))
        y1 = max(y0 + 1, min(height, int(self.bottom * height + 0.5)))
        return x0, y0, x1, y1


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def parse(cls, value: Any, *, lo: float = 0, hi: float = 1) -> "Point":
        x, y = sequence(value, "坐标", minimum=2, maximum=2)
        return cls(numeric(x, "横坐标", lo, hi), numeric(y, "纵坐标", lo, hi))

    def values(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True)
class ToolRequest:
    key: str
    comparison: str | None
    region: Region
    calibration_id: str | None
    options: dict

    @classmethod
    def parse(cls, data: Any) -> "ToolRequest":
        data = mapping(data, "检查参数")
        parameters = data.get("options", {})
        mapping(parameters, "检查选项")
        if len(parameters) > 25:
            raise Problem("检查选项过多。")
        return cls(
            required_text(data.get("tool"), "检查工具", 60),
            plain(data.get("comparison"), "第二张采集", 64) or None,
            Region.parse(data.get("region")),
            plain(data.get("calibration_id"), "角点校准", 64) or None,
            parameters,
        )
