from dataclasses import dataclass
from itertools import combinations
from math import hypot

import numpy as np
from scipy.ndimage import map_coordinates

from .model import Point, Problem, Region, numeric, sequence


UNIT_CORNERS = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
MAX_CANVAS = 1600


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float((x @ np.roll(y, -1) - y @ np.roll(x, -1)) * 0.5)


def parse_quad(raw: list) -> np.ndarray:
    points = [Point.parse(point, lo=-0.10, hi=1.10).values() for point in sequence(
        raw, "面板角点", minimum=4, maximum=4
    )]
    corners = np.asarray(points, dtype=np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if np.any(cross <= 1e-4) or polygon_area(corners) < 0.008:
        raise Problem("请按左上、右上、右下、左下依次标注足够大的四边形。")
    if np.any(np.linalg.norm(edges, axis=1) < 0.025):
        raise Problem("一对面板角点过近。")
    return corners


def similarity_normalization(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance < 1e-9:
        raise Problem("重合的参考点不能建立坐标关系。")
    scale = np.sqrt(2) / distance
    transform = np.array([
        [scale, 0.0, -center[0] * scale],
        [0.0, scale, -center[1] * scale],
        [0.0, 0.0, 1.0],
    ])
    return (points - center) * scale, transform


def estimate_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) != len(target) or len(source) < 4:
        raise Problem("至少需要四组匹配的坐标。")
    src, from_matrix = similarity_normalization(source)
    dst, to_matrix = similarity_normalization(target)
    x, y = src.T
    u, v = dst.T
    ones = np.ones(len(src))
    zeros = np.zeros(len(src))
    equations = np.zeros((2 * len(src), 9))
    equations[::2] = np.column_stack([-x, -y, -ones, zeros, zeros, zeros, u*x, u*y, u])
    equations[1::2] = np.column_stack([zeros, zeros, zeros, -x, -y, -ones, v*x, v*y, v])
    _, singular, right = np.linalg.svd(equations, full_matrices=len(src) == 4)
    if len(singular) < 8 or singular[7] < 1e-7:
        raise Problem("参考点位于同一直线附近，无法校准面板。")
    result = np.linalg.inv(to_matrix) @ right[-1].reshape(3, 3) @ from_matrix
    scale = result[2, 2]
    if abs(scale) < 1e-9 or abs(np.linalg.det(result)) < 1e-8:
        raise Problem("角点产生了退化的投影。")
    result = result / scale
    denominator = UNIT_CORNERS @ result[2, :2] + result[2, 2]
    if np.any(denominator < 1e-4):
        raise Problem("投影的消失线穿过面板，请复核四角。")
    return result


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = points @ matrix[2, :2] + matrix[2, 2]
        numerators = points @ matrix[:2, :2].T + matrix[:2, 2]
        return numerators / denominator[..., None]


@dataclass(frozen=True)
class Calibration:
    corners: np.ndarray
    transform: np.ndarray
    inverse: np.ndarray

    @classmethod
    def from_raw(cls, raw: list) -> "Calibration":
        corners = parse_quad(raw)
        matrix = estimate_homography(UNIT_CORNERS, corners)
        return cls(corners, matrix, np.linalg.inv(matrix))

    @classmethod
    def identity(cls) -> "Calibration":
        return cls(UNIT_CORNERS, np.eye(3), np.eye(3))

    def panel_to_camera(self, points: np.ndarray) -> np.ndarray:
        return apply_homography(self.transform, points)

    def camera_to_panel(self, points: np.ndarray) -> np.ndarray:
        return apply_homography(self.inverse, points)

    def coverage_estimate(self, samples: int = 100) -> float:
        u = (np.arange(samples) + 0.5) / samples
        grid = np.stack(np.meshgrid(u, u), axis=-1)
        camera_points = self.panel_to_camera(grid)
        valid = np.all((camera_points >= 0) & (camera_points <= 1), axis=-1)
        return float(valid.mean())

    def pixels_per_panel_step(self, image_shape: tuple[int, int]) -> tuple[float, float]:
        height, width = image_shape[:2]
        camera_points = self.corners * [width - 1, height - 1]
        top = np.linalg.norm(camera_points[1] - camera_points[0])
        bottom = np.linalg.norm(camera_points[2] - camera_points[3])
        left = np.linalg.norm(camera_points[3] - camera_points[0])
        right = np.linalg.norm(camera_points[2] - camera_points[1])
        return float((top + bottom) / 2), float((left + right) / 2)

    def camera_rectangle(self, region: Region) -> list[list[float]]:
        polygon = np.array([
            [region.left, region.top],
            [region.right, region.top],
            [region.right, region.bottom],
            [region.left, region.bottom],
        ])
        return self.panel_to_camera(polygon).tolist()

    def overview(self, image_shape: tuple[int, int]) -> dict:
        coverage = self.coverage_estimate()
        resolution = self.pixels_per_panel_step(image_shape)
        return {
            "corners": self.corners.tolist(),
            "projected_camera_area_pct": round(100 * polygon_area(self.corners), 2),
            "coverage_pct": round(100 * coverage, 2),
            "camera_sampling": [round(value, 1) for value in resolution],
            "projective_determinant": round(float(np.linalg.det(self.transform)), 8),
            "may_reach_source_extent": bool(np.any(self.corners < 0) or np.any(self.corners > 1)),
        }


def scaled_dimensions(
    width: int, height: int, *, maximum: int = MAX_CANVAS, limit_area: int = 1_500_000
) -> tuple[int, int]:
    scale = min(1, maximum / max(width, height), np.sqrt(limit_area / (width * height)))
    return max(2, round(scale * width)), max(2, round(scale * height))


def planned_canvas(
    native_width: int,
    native_height: int,
    shape: tuple[int, int],
    calibration: Calibration,
    *,
    max_side: int = MAX_CANVAS,
) -> tuple[int, int]:
    camera_width, camera_height = calibration.pixels_per_panel_step(shape)
    native_scale = min(1, max(camera_width / native_width, camera_height / native_height))
    width = max(16, round(native_scale * native_width))
    height = max(16, round(native_scale * native_height))
    return scaled_dimensions(width, height, maximum=max_side)


def panel_coordinates(width: int, height: int, region: Region) -> np.ndarray:
    x = region.left + (np.arange(width, dtype=np.float64) + 0.5) * (region.right - region.left) / width
    y = region.top + (np.arange(height, dtype=np.float64) + 0.5) * (region.bottom - region.top) / height
    columns, rows = np.meshgrid(x, y)
    return np.stack([columns, rows], axis=-1)


def sampling_grid(
    camera_shape: tuple[int, int], camera_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = camera_shape[:2]
    valid = np.all((camera_points >= 0) & (camera_points <= 1), axis=-1)
    if not np.all(np.isfinite(camera_points)):
        camera_points = np.nan_to_num(camera_points, nan=-1, neginf=-1, posinf=2)
    coordinates = np.array([
        np.clip(camera_points[..., 1], 0, 1) * (height - 1),
        np.clip(camera_points[..., 0], 0, 1) * (width - 1),
    ])
    return coordinates, valid


def sample_rgb(
    raster: np.ndarray, calibration: Calibration, panel_points: np.ndarray, *, order: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = calibration.panel_to_camera(panel_points)
    coordinates, valid = sampling_grid(raster.shape[:2], camera_points)
    planes = [map_coordinates(raster[..., band], coordinates, order=order, mode="nearest")
              for band in range(raster.shape[2])]
    return np.stack(planes, axis=-1), valid


def intersection(first: Region, second: Region) -> Region | None:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if left >= right or top >= bottom:
        return None
    return Region(left, top, right, bottom)


def point_distance(a: Point, b: Point, scales: tuple[float, float]) -> float:
    return hypot((a.x - b.x) * scales[0], (a.y - b.y) * scales[1])


def local_to_panel(x: float, y: float, width: int, height: int, region: Region) -> tuple[float, float]:
    u = region.left + (x + 0.5) / width * (region.right - region.left)
    v = region.top + (y + 0.5) / height * (region.bottom - region.top)
    return float(u), float(v)


def segment_statistics(start: Point, stop: Point, panel: dict, calibration: Calibration,
                       source_width: int, source_height: int) -> dict:
    panel_size = panel["native_width"], panel["native_height"]
    mapped = calibration.panel_to_camera(np.asarray([start.values(), stop.values()]))
    first, second = (Point(*xy) for xy in mapped)
    measured = {
        "panel_px": round(point_distance(start, stop, panel_size), 3),
        "source_px": round(point_distance(first, second, (source_width, source_height)), 3),
        "panel_mm": None,
        "start_camera": first.values(),
        "end_camera": second.values(),
    }
    if panel["width_mm"] is not None:
        size_mm = panel["width_mm"], panel["height_mm"]
        measured["panel_mm"] = round(point_distance(start, stop, size_mm), 3)
    return measured


def line_samples(start: Point, stop: Point, detail: int = 240) -> np.ndarray:
    return np.linspace([start.x, start.y], [stop.x, stop.y], detail)


def overlapping_quads(first: Calibration, second: Calibration, steps: int = 64) -> float:
    grid = panel_coordinates(steps, steps, Region(0, 0, 1, 1))
    first_pts = first.panel_to_camera(grid)
    second_pts = second.panel_to_camera(grid)
    first_valid = np.all((first_pts >= 0) & (first_pts <= 1), axis=-1)
    second_valid = np.all((second_pts >= 0) & (second_pts <= 1), axis=-1)
    return float((first_valid & second_valid).mean())


def choose_nearest(points: list[dict], x: float, y: float, limit: float) -> dict | None:
    result = None
    minimum = numeric(limit, "搜索阈值", 0.0001, 1)
    for point in points:
        distance = hypot(point["u"] - x, point["v"] - y)
        if distance < minimum:
            minimum = distance
            result = point
    return result


def spaced_points(points: list[dict], min_distance: float) -> list[dict]:
    chosen = []
    for point in points:
        if all(hypot(point["u"] - other["u"], point["v"] - other["v"]) >= min_distance
               for other in chosen):
            chosen.append(point)
    return chosen


def distance_summary(points: list[Point], scales: tuple[float, float]) -> dict:
    distances = [point_distance(a, b, scales) for a, b in combinations(points, 2)]
    if not distances:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": float(min(distances)),
        "median": float(np.median(distances)),
        "maximum": float(max(distances)),
    }
