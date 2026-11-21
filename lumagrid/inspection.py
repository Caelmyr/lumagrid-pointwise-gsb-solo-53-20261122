"""Physical panel sampling, stable revision selection, and report composition."""
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Callable
import csv
import io
import re

import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import (
    Calibration, panel_coordinates, planned_canvas, sample_rgb, sampling_grid, scaled_dimensions,
)
from .model import Problem, Region, ToolRequest, choice, mapping, plain
from .raster import (
    DecodedRaster, code_statistics, decode, heatmap, luma, masked_rgb, png_bytes, resize_report,
    rounding,
)
from .storage import Store


@dataclass
class Scene:
    panel: dict
    capture: dict
    revision: dict | None
    calibration: Calibration
    image: DecodedRaster

    @classmethod
    def load(cls, store: Store, capture_id: str, calibration_id: str | None = None) -> "Scene":
        panel, capture, raw, revision = store.read_for_analysis(capture_id, calibration_id)
        calibration = Calibration.from_revision(revision) if revision else Calibration.identity()
        return cls(panel, capture, revision, calibration, decode(raw))

    @property
    def revision_id(self) -> str:
        return self.revision["id"] if self.revision else "identity"

    @property
    def base_revision(self) -> str | None:
        """The four-corner version an internal-reference correction builds on."""
        if not self.revision:
            return None
        return (self.revision.get("info") or {}).get("base_calibration_id") or self.revision["id"]

    @property
    def uses_internal_reference(self) -> bool:
        return self.calibration.field is not None

    @property
    def scale(self) -> tuple[int, int]:
        return self.panel["native_width"], self.panel["native_height"]

    def describe(self) -> dict:
        return {
            "capture_id": self.capture["id"],
            "capture_title": self.capture["title"],
            "source_sha256": self.capture["original"],
            "source_size": list(self.image.size),
            "source_role": self.capture["role"],
            "source_bits": self.image.bits,
            "calibration_id": self.revision_id,
            "calibration_revision": self.revision["revision"] if self.revision else None,
            "base_calibration_id": self.base_revision,
            "geometry_model": "corners+internal-reference" if self.uses_internal_reference else "corners",
            "internal_reference": self.calibration.reference_summary() or None,
            "exposure": self.capture["exposure"],
        }

    def require_role(self, roles: set[str], message: str) -> None:
        if self.capture["role"] not in roles:
            raise Problem(message)

    def sample(self, coords: np.ndarray, *, nearest=False) -> tuple[np.ndarray, np.ndarray]:
        order = 0 if nearest else 1
        camera_points = self.calibration.panel_to_camera(coords)
        camera_grid, inside = sampling_grid(self.image.valid.shape, camera_points)
        alpha = map_coordinates(
            self.image.valid.astype(np.float32), camera_grid, order=order, mode="nearest"
        )
        planes = [
            map_coordinates(face, camera_grid, order=order, mode="nearest")
            for face in self.image.rgb.transpose(2, 0, 1)
        ]
        inside &= alpha >= 1 - 1e-6
        return np.stack(planes, axis=-1), inside

    def working_size(self, region: Region, max_side: int = 900) -> tuple[int, int]:
        width, height = planned_canvas(
            *self.scale,
            self.image.valid.shape,
            self.calibration,
        )
        crop_width = max(16, round(width * (region.right - region.left)))
        crop_height = max(16, round(height * (region.bottom - region.top)))
        return scaled_dimensions(crop_width, crop_height, maximum=max_side, limit_area=850_000)


@dataclass
class Plane:
    rgb: np.ndarray
    observed: np.ndarray
    points: np.ndarray
    region: Region

    @property
    def gray(self) -> np.ndarray:
        return luma(self.rgb)

    @property
    def shape(self) -> tuple[int, int]:
        return self.observed.shape

    @property
    def coverage(self) -> float:
        return float(self.observed.mean())

    def require_coverage(self, fraction: float = .30, *, min_samples: int = 20) -> None:
        if self.coverage < fraction or np.count_nonzero(self.observed) < min_samples:
            raise Problem("选区中有依据的像素太少，请调整角点或选区。")

    def usable(self, *, avoid_saturated: bool) -> np.ndarray:
        if not avoid_saturated:
            return self.observed
        return self.observed & np.all((self.rgb > .006) & (self.rgb < .994), axis=-1)

    def info(self) -> dict:
        height, width = self.shape
        return {
            "region": self.region.values(),
            "working_size": [width, height],
            "coverage_pct": rounding(self.coverage * 100, 2),
            "positive_samples": int(np.count_nonzero(self.observed)),
        }


def make_plane(scene: Scene, region: Region, shape=None, *, max_side=900, nearest=False) -> Plane:
    width, height = shape or scene.working_size(region, max_side)
    coords = panel_coordinates(width, height, region)
    rgb, valid = scene.sample(coords, nearest=nearest)
    return Plane(rgb, valid, coords, region)


def paired_plane(first: Scene, second: Scene, region: Region, *, max_side=900):
    if first.panel["id"] != second.panel["id"]:
        raise Problem("对照图必须来自同一尺寸记录的面板。")
    if first.capture["id"] == second.capture["id"]:
        raise Problem("请选择另一张采集作为对照。")
    first_width, first_height = first.working_size(region, max_side)
    second_width, second_height = second.working_size(region, max_side)
    size = min(first_width, second_width), min(first_height, second_height)
    first_plane = make_plane(first, region, size)
    second_plane = make_plane(second, region, size)
    overlap = first_plane.observed & second_plane.observed
    if overlap.mean() < .10 or overlap.sum() < 64:
        raise Problem("两张采集在该面板选区的共同可见像素不足。")
    return first_plane, second_plane, overlap


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.shape != weights.shape or values.size == 0:
        raise Problem("缺少支撑统计的有效样本。")
    ascending = np.argsort(values, kind="stable")
    xs, ws = values[ascending], weights[ascending]
    cumulative = ws.cumsum()
    if cumulative[-1] <= 0:
        raise Problem("缺少权重大于零的样本。")
    index = np.searchsorted(cumulative, cumulative[-1] * .5)
    return float(xs[index])


def stable_fraction(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-8:
        return None
    return rounding(numerator / denominator)


@dataclass
class Report:
    title: str
    summary: str
    metrics: dict
    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: dict[str, tuple[bytes, str]] = field(default_factory=dict)

    def add(self, name: str, data: bytes, mime: str) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,70}", name) or PurePath(name).name != name:
            raise ValueError(f"Invalid result file name {name!r}")
        if name in self.files:
            raise ValueError(f"Duplicate result file {name!r}")
        self.files[name] = data, mime

    def image(self, name: str, image) -> None:
        self.add(name, png_bytes(resize_report(image)), "image/png")

    def table(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        order = list(dict.fromkeys(key for row in rows for key in row))
        content = io.StringIO(newline="")
        writer = csv.DictWriter(content, fieldnames=order)
        writer.writeheader()
        for row in rows:
            fields = {}
            for key, value in row.items():
                if isinstance(value, str) and value.lstrip(" \t\n\r").startswith(("=", "-", "+", "@")):
                    value = "'" + value
                fields[key] = value
            writer.writerow(fields)
        self.add(name, content.getvalue().encode("utf-8-sig"), "text/csv")

    def as_json(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "metrics": self.metrics,
            "rows": self.rows,
            "warnings": self.warnings,
        }


def rectification(scene: Scene, comparison: Scene | None, region: Region, options: dict) -> Report:
    method = choice(options.get("sampling"), {"bilinear", "nearest"}, "采样方式", default="bilinear")
    plane = make_plane(scene, region, nearest=method == "nearest")
    plane.require_coverage(.05)
    stats = {
        **plane.info(),
        "sampling": method,
        "geometry": scene.calibration.overview(scene.image.valid.shape),
        "geometry_model": "corners+internal-reference" if scene.uses_internal_reference else "corners",
        "recorded_code": code_statistics(plane.gray[plane.observed]),
    }
    report = Report("面板展平", "在标定后的面板坐标中查看可采样区域。", stats)
    report.image("rectified.png", masked_rgb(plane.rgb, plane.observed))
    if scene.uses_internal_reference:
        support = scene.calibration.support(plane.points)
        summary = scene.calibration.reference_summary()
        stats["geometry_reference_support_pct"] = summary["support_area_pct"]
        report.image("geometry-support.png", heatmap(support, plane.observed, 0, 1))
        report.metrics["geometry_support_scale"] = "青色区使用内部参照；暗色区仍沿用四角投影"
        report.warnings.append(
            f"本版标定在约 {summary['support_area_pct']:.0f}% 面板面积使用内部参照，"
            "其余区域仍按四角投影展平；对照图为参照支撑强度，不是图像内容。"
        )
    if plane.coverage < .97:
        report.warnings.append("深灰区域在原始采集中缺少完整像素采样支撑。")
    if scene.revision is None:
        report.warnings.append("当前使用整张相机图作为面板范围，尚无单独保存的校准。")
    return report


def tool_config() -> dict:
    from .assays import configs as light_configs
    from .response import configs as response_configs
    from .structure import configs as structure_configs
    from .calibration_tools import configs as calibration_configs
    from .sensor import configs as sensor_configs
    from .seams import configs as seam_configs
    from .noise import configs as noise_configs
    from .probe import configs as probe_configs
    from .displacement import configs as displacement_configs
    from .vignetting import configs as vignetting_configs
    from .field import configs as field_configs

    return {
        "rectify": {
            "title": "面板展平",
            "group": "采集坐标",
            "description": "将透视采集转成面板坐标图，显示有效覆盖。",
            "run": rectification,
            "options": [
                {"key": "sampling", "label": "采样方式", "type": "select",
                 "values": {"bilinear": "双线性", "nearest": "最近邻"}, "default": "bilinear"},
            ],
        },
        **light_configs(),
        **response_configs(),
        **structure_configs(),
        **calibration_configs(),
        **sensor_configs(),
        **seam_configs(),
        **noise_configs(),
        **probe_configs(),
        **displacement_configs(),
        **vignetting_configs(),
        **field_configs(),
    }


def public_tools() -> list[dict]:
    return [{"key": name, **{k: v for k, v in data.items() if k != "run"}}
            for name, data in tool_config().items()]


def scene_quality_notes(scene: Scene, comparison: Scene | None, space: str) -> list[str]:
    notices = []
    for name, selected in (("主采集", scene), ("对照采集", comparison)):
        if selected is None:
            continue
        if space == "panel" and selected.revision is None:
            notices.append(f"{name}未单独标定，使用整张图作为面板范围。")
        if not selected.capture["exposure"].get("camera"):
            notices.append(f"{name}未填写相机标识。")
        if space == "camera" and selected.revision is not None:
            notices.append(f"{name}当前工具使用相机原像素，不使用四角插值。")
    return notices


def execute(store: Store, capture_id: str, data: dict) -> tuple[Scene, Scene | None, dict, Report]:
    data = mapping(data, "检查参数")
    request = ToolRequest.parse(data)
    tools = tool_config()
    if request.key not in tools:
        raise Problem("没有这个检查工具。", 404)
    config = tools[request.key]
    scene = Scene.load(store, capture_id, request.calibration_id)
    comparison_id = request.comparison if config.get("comparison") else None
    if not comparison_id and config.get("comparison") == "required":
        raise Problem("请选择这项检查使用的对照图。")
    comparison_rev = plain(data.get("comparison_calibration_id"), "对照图角点版本", 64) or None
    second = Scene.load(store, comparison_id, comparison_rev) if comparison_id else None
    if second and second.panel["id"] != scene.panel["id"]:
        raise Problem("检查只能使用当前面板的采集。")
    if second and second.capture["id"] == scene.capture["id"]:
        raise Problem("请选择另一张采集作为对照。")
    run: Callable = config["run"]
    outcome = run(scene, second, request.region, request.options)
    report, correction = outcome if isinstance(outcome, tuple) else (outcome, None)
    domain = config.get("coordinate_space", "panel")
    report.warnings.extend(scene_quality_notes(scene, second, domain))
    input_state = {
        "tool": request.key,
        "capture_id": capture_id,
        "comparison_id": comparison_id,
        "calibration_id": scene.revision_id,
        "comparison_calibration_id": second.revision_id if second else None,
        "region": request.region.values(),
        "coordinate_space": domain,
        "options": request.options,
        "source": scene.describe(),
        "comparison": second.describe() if second else None,
        "synthetic": scene.panel["synthetic"],
    }
    if correction is not None:
        input_state["correction"] = correction
    report.metrics["source_title"] = scene.capture["title"]
    report.metrics["synthetic"] = scene.panel["synthetic"]
    if scene.uses_internal_reference:
        report.metrics["geometry_model"] = "corners+internal-reference"
        report.metrics["geometry_reference_support_pct"] = scene.calibration.reference_summary()[
            "support_area_pct"
        ]
    return scene, second, input_state, report
