"""Sample a physical panel point without conflating a pixel with its neighborhood."""
from dataclasses import dataclass
from math import pi

import numpy as np
from PIL import ImageDraw

from .geometry import apply_homography
from .inspection import Report, Scene
from .model import Point, Problem, Region, boolean, integral, numeric, sequence
from .raster import chart_png, code_statistics, luma, masked_rgb, rounding


@dataclass(frozen=True)
class Footprint:
    center: Point
    radius_x: float
    radius_y: float
    radial_count: int

    def panel_positions(self):
        # Uniform area under polar coordinates requires the square root radii.
        angles = np.linspace(0, 2 * pi, self.radial_count, endpoint=False)
        radii = np.sqrt((np.arange(self.radial_count) + .5) / self.radial_count)
        along_x = self.radius_x * np.cos(angles)[None, :] * radii[:, None]
        along_y = self.radius_y * np.sin(angles)[None, :] * radii[:, None]
        positions = np.stack([self.center.x + along_x, self.center.y + along_y], axis=-1)
        return positions, radii

    def projected_outline(self, scene: Scene):
        angles = np.linspace(0, 2 * pi, 72, endpoint=True)
        aligned = np.stack([
            self.center.x + self.radius_x * np.cos(angles),
            self.center.y + self.radius_y * np.sin(angles),
        ], axis=-1)
        return scene.calibration.panel_to_camera(aligned)


def probe_position(region: Region, options: dict):
    value = options.get("panel_point")
    if value:
        return Point.parse(value)
    return Point((region.left + region.right) / 2, (region.top + region.bottom) / 2)


def extract_neighborhood(scene: Scene, footprint: Footprint, nearest: bool):
    points, radii = footprint.panel_positions()
    valid_panel = np.all((points >= 0) & (points <= 1), axis=-1)
    rgb, covered = scene.sample(points, nearest=nearest)
    keep = valid_panel & covered
    if np.count_nonzero(keep) < max(8, .12 * keep.size):
        raise Problem("这个物理探针的邻域落在采集有效范围之外。")
    source_coords = scene.calibration.panel_to_camera(np.asarray([footprint.center.values()]))[0]
    sample_values, sample_valid = scene.sample(np.asarray([footprint.center.values()]), nearest=nearest)
    width, height = scene.image.size
    projected = {
        "normalized_camera": [rounding(coord, 8) for coord in source_coords],
        "source_pixel": [
            rounding(source_coords[0] * (width - 1), 3), rounding(source_coords[1] * (height - 1), 3),
        ],
        "point_is_observed": bool(sample_valid[0]),
        "point_rgb": [rounding(part) for part in sample_values[0]] if sample_valid[0] else None,
        "projection_revision": scene.revision_id,
    }
    mean = np.mean(rgb[keep], axis=0)
    median = np.median(rgb[keep], axis=0)
    mad = np.median(np.abs(rgb[keep] - median), axis=0) * 1.4826
    aggregate = {
        "neighbor_rgb_avg": [rounding(channel) for channel in mean],
        "neighbor_rgb_median": [rounding(channel) for channel in median],
        "neighbor_rgb_mad": [rounding(channel) for channel in mad],
        "luma": code_statistics(luma(rgb)[keep]),
        "neighbor_coverage_pct": rounding(keep.mean() * 100, 3),
    }
    ring_rows = []
    for ring, radius in enumerate(radii, start=1):
        values = rgb[ring - 1][keep[ring - 1]]
        ring_rows.append({
            "ring": ring,
            "radius_fraction": rounding(radius, 4),
            "valid_samples": len(values),
            "mean_y": rounding(luma(values).mean()) if len(values) else None,
            "r": rounding(values[:, 0].mean()) if len(values) else None,
            "g": rounding(values[:, 1].mean()) if len(values) else None,
            "b": rounding(values[:, 2].mean()) if len(values) else None,
        })
    return projected, aggregate, ring_rows


def probe_overlay(scene: Scene, footprint: Footprint, projected: dict):
    width, height = scene.image.size
    photo = masked_rgb(scene.image.rgb, scene.image.valid)
    draw = ImageDraw.Draw(photo)
    outline = footprint.projected_outline(scene)
    scale = [width - 1, height - 1]
    coords = [(float(x), float(y)) for x, y in outline * scale if np.isfinite(x) and np.isfinite(y)]
    if len(coords) >= 3:
        draw.line(coords, fill=(120, 239, 210), width=max(2, width // 450))
    x, y = projected["source_pixel"]
    marker = max(5, min(35, width // 110))
    draw.line((x - marker, y, x + marker, y), fill=(252, 213, 110), width=2)
    draw.line((x, y - marker, x, y + marker), fill=(252, 213, 110), width=2)
    return photo


def contrast_row(primary: dict, secondary: dict, name: str):
    a, b = primary.get(name), secondary.get(name)
    if a is None or b is None:
        return None
    return [rounding(second - first) for first, second in zip(a, b, strict=True)]


def surface_probe(scene: Scene, second: Scene | None, region: Region, options: dict) -> Report:
    point = probe_position(region, options)
    radius_px = numeric(options.get("radius"), "面板半径像素", 1, 480, default=18)
    detail = integral(options.get("detail"), "每环采样数", 8, 120, default=32)
    native_w, native_h = scene.scale
    radius_x = radius_px / native_w
    radius_y = radius_px / native_h
    if max(radius_x, radius_y) > .3:
        raise Problem("邻域半径超过面板对应尺寸的 30%，请缩小邻域。")
    nearest = boolean(options.get("nearest"), "使用最近相机像素", default=False)
    aperture = Footprint(point, radius_x, radius_y, detail)
    current, cluster, rows = extract_neighborhood(scene, aperture, nearest)
    metrics = {
        "coordinate_panel": point.values(),
        "physical_radius_panel_px": radius_px,
        "samples_per_ring": detail,
        "sampling": "nearest" if nearest else "bilinear",
        "primary_point": current,
        "primary_aperture": cluster,
    }
    if scene.panel["width_mm"] is not None:
        metrics["radius_width_mm"] = rounding(radius_x * scene.panel["width_mm"], 4)
        metrics["radius_height_mm"] = rounding(radius_y * scene.panel["height_mm"], 4)
    report = Report("面板像素探针", "从同一物理位置探查点值和周边采样，保留相机位置及校准版本。", metrics, rows)
    report.image("primary-probe.png", probe_overlay(scene, aperture, current))
    plot_radii = np.asarray([row["radius_fraction"] for row in rows])
    primary_channel = np.asarray([row["mean_y"] if row["mean_y"] is not None else np.nan for row in rows])
    chart = [("Primary", plot_radii, primary_channel)]
    if second is not None:
        compared, neighborhood, second_rows = extract_neighborhood(second, aperture, nearest)
        metrics["comparison_point"] = compared
        metrics["comparison_aperture"] = neighborhood
        metrics["comparison_point_rgb_delta"] = contrast_row(current, compared, "point_rgb")
        metrics["comparison_aperture_median_delta"] = contrast_row(cluster, neighborhood, "neighbor_rgb_median")
        report.image("comparison-probe.png", probe_overlay(second, aperture, compared))
        other_channel = np.asarray([row["mean_y"] if row["mean_y"] is not None else np.nan for row in second_rows])
        chart.append(("Comparison", plot_radii, other_channel))
        for first, other in zip(rows, second_rows, strict=True):
            first["comparison_y"] = other["mean_y"]
            first["comparison_valid_samples"] = other["valid_samples"]
    report.add("probe-response.png", chart_png(
        chart, x_label="Fraction of physical radius", y_label="Mean observation in each ring",
        x_bounds=(0, 1),
    ), "image/png")
    report.table("probe-neighborhood.csv", rows)
    if not current["point_is_observed"]:
        report.warnings.append("中心点投影在不可见区域；邻域仍有部分实采像素。")
    if cluster["neighbor_coverage_pct"] < 85:
        report.warnings.append("邻域被图像范围裁开，统计仅反映可见部分。")
    return report


def configs():
    from .assays import number, tick
    return {
        "probe": {
            "title": "面板像素探针",
            "group": "采集坐标",
            "description": "采样物理位置及其邻域，也可复核第二张采集的同一点。",
            "run": surface_probe,
            "comparison": "optional",
            "options": [
                {"key": "panel_point", "label": "可选面板坐标 [x,y]，留 null 用选区中心", "type": "json",
                 "default": None},
                number("radius", "物理半径（面板像素）", 18, 1, 480),
                number("detail", "邻域每环采样数", 32, 8, 120),
                tick("nearest", "使用最近相机像素", False),
            ],
        },
    }
