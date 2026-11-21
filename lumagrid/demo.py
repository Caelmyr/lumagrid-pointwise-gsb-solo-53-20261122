"""Deterministic, explicitly synthetic camera samples for the inspection tour."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
from io import StringIO

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import Calibration
from .model import CaptureSpec, PanelSpec
from .raster import decode_asset, png_bytes, display_code
from .storage import Store


DEMO_KEY = "lumagrid-optical-room-tour-1"
PANEL_WIDTH = 1080
PANEL_HEIGHT = 648
WIDTH = 850
HEIGHT = 555
FIRST_CAPTURE = datetime(2026, 5, 12, 4, 5, tzinfo=timezone.utc)
CAMERA = "演示相机 C7"
CORNERS = (
    ((.100, .140), (.880, .093), (.926, .865), (.095, .902)),
    ((.148, .097), (.876, .153), (.831, .887), (.103, .826)),
    ((.119, .142), (.934, .135), (.952, .855), (.103, .896)),
    ((.066, .122), (.914, .078), (.982, .842), (.093, .940)),
    ((-.027, .119), (.872, .097), (.921, .885), (-.038, .893)),
    ((.050, .300), (.930, .060), (.970, .920), (.030, .660)),
)


@dataclass(frozen=True)
class DemoSource:
    key: str
    title: str
    role: str
    photo: str
    shutter: float
    level: float
    angle: float = 0.0
    pose: int = 0
    seed: int = 21


SOURCES = [
    DemoSource("uniform", "合成·中灰定位", "uniform", "uniform", 4.0, .50),
    DemoSource("uniform_oblique", "合成·中灰斜拍", "reference", "oblique", 4.0, .50, 26.0, 1, 31),
    DemoSource("uniform_clipped", "合成·中灰左沿出画", "uniform", "uniform", 4.0, .50, -10.0, 4, 61),
    DemoSource("black", "合成·暗室低灰", "black", "leak", 4.0, .035, seed=44),
    DemoSource("checker", "合成·6×4 棋盘格", "checker", "checker", 4.0, .50, pose=3, seed=23),
    DemoSource("ramp", "合成·九阶灰卡", "ramp", "steps", 4.0, .52, seed=37),
    DemoSource("edge", "合成·右倾斜边", "edge", "edge", 4.0, .60, 9.0, 2, 41),
    DemoSource("edge_card", "合成·多向斜边测试卡", "edge", "edge_card", 4.0, .60, 14.0, 5, 53),
    DemoSource("reflect", "合成·斜侧补光", "reference", "reflection", 4.0, .50, 2.0, 3, 47),
    DemoSource("band_a", "合成·条纹 4 毫秒", "uniform", "band_a", 4.0, .50, seed=11),
    DemoSource("band_b", "合成·条纹 8 毫秒", "uniform", "band_b", 8.0, .50, pose=1, seed=29),
    DemoSource("band_c", "合成·条纹 13 毫秒", "uniform", "band_c", 13.0, .50, pose=2, seed=39),
    DemoSource("micro", "合成·RGB 子像素近摄", "macro", "macro", 4.0, .70, seed=42),
    DemoSource("dark", "合成·镜头暗场", "dark", "dark", 4.0, 0.0, seed=77),
    DemoSource("flat", "合成·镜头平场", "flat", "flat", 4.0, .85, seed=78),
    DemoSource("white", "合成·亮场参考", "white", "white", 4.0, .90, 0.0, 0, 19),
    DemoSource("gray45", "合成·稳态灰场 45%", "uniform", "gray45", 4.0, .45, 0.0, 2, 119),
    DemoSource("fade1", "合成·移除图案后 0.2 秒", "uniform", "fade1", 4.0, .45, 0.0, 1, 121),
    DemoSource("fade2", "合成·移除图案后 1 秒", "uniform", "fade2", 4.0, .45, 0.0, 2, 123),
    DemoSource("fade3", "合成·移除图案后 3 秒", "uniform", "fade3", 4.0, .45, 0.0, 3, 131),
    DemoSource("spot_alt", "合成·中灰重复点检", "uniform", "spot_repeat", 4.0, .50, 8.0, 2, 59),
    DemoSource("module_seam", "合成·拼接模块画面", "uniform", "module_seam", 4.0, .50, seed=62),
    DemoSource("display_rim", "合成·发光画面内沿", "uniform", "display_rim", 4.0, .55, 0.0, 2, 73),
    DemoSource("dark_repeat", "合成·镜头暗场复拍", "dark", "dark", 4.0, 0.0, seed=101),
]


def bump(x, y, cx, cy, sx, sy):
    return np.exp(-.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def background(u, v):
    falloff = .06 * ((u - .54) ** 2 + 1.2 * (v - .48) ** 2)
    left = bump(u, v, .24, .34, .24, .23)
    corner = bump(u, v, .89, .82, .15, .18)
    return .355 + .09 * left - .055 * corner - falloff + .004 * np.sin(2 * np.pi * v * 2.4)


def as_color(gray: np.ndarray, offsets=(0.0, 0.0, 0.0)):
    return np.stack([gray + color for color in offsets], axis=-1).astype(np.float32)


def exercise_minor_markers(u: np.ndarray, v: np.ndarray, gray: np.ndarray) -> np.ndarray:
    result = gray.copy()
    sample_points = ((.26, .34), (.51, .74), (.79, .32), (.72, .68))
    for column, row in sample_points:
        result += .055 * bump(u, v, column, row, .0026, .0037)
        result -= .015 * bump(u, v, column + .010, row + .011, .0027, .0042)
    return result


def optics_panel(pattern: str, u: np.ndarray, v: np.ndarray):
    gray = background(u, v)
    if pattern == "uniform":
        return as_color(exercise_minor_markers(u, v, gray), (.012, 0, -.010))
    if pattern == "spot_repeat":
        gray = exercise_minor_markers(u, v, gray)
        gray += .050 * bump(u, v, .11, .52, .0028, .004)
        gray += .043 * bump(u, v, .63, .83, .0034, .004)
        return as_color(gray, (.010, 0, -.005))
    if pattern == "oblique":
        spread = bump(u, v, .80, .44, .27, .33)
        baseline = gray * .92 + .012
        red = baseline + .060 * spread
        green = baseline - .010 * (u - .35)
        blue = baseline - .038 * spread + .027 * bump(u, v, .24, .38, .30, .25)
        return np.stack([red, green, blue], axis=-1)
    if pattern == "leak":
        dark = .033 + .088 * bump(u, v, .09, .92, .082, .059)
        dark += .037 * bump(u, v, .81, .13, .12, .045)
        dark += .059 * bump(u, v, .65, .53, .063, .069)
        return as_color(dark, (.006, 0, .002))
    if pattern == "checker":
        columns = np.clip(np.floor(u * 6), 0, 5).astype(np.int32)
        rows = np.clip(np.floor(v * 4), 0, 3).astype(np.int32)
        expected_white = (columns + rows) % 2 == 0
        brightness = np.where(expected_white, .715, .117)
        du = np.abs((u * 6) % 1 - .5)
        dv = np.abs((v * 4) % 1 - .5)
        glow = np.clip((np.maximum(du, dv) - .39) * 7.5, 0, .70)
        brightness = brightness * (1 - glow) + .43 * glow + (gray - .355) * .34
        return as_color(brightness, (.014, .005, 0))
    if pattern == "steps":
        steps = np.clip(np.floor(u * 9), 0, 8) / 8
        response = .019 + .873 * steps ** 1.77
        response += .003 * np.cos(v * 6) + .012 * (gray - .355)
        return as_color(response, (.003, 0, -.001))
    if pattern == "edge":
        distance = u - (.49 + (v - .5) * .16)
        response = .14 + .66 * (.5 + .5 * np.tanh(distance * 270))
        response += .025 * np.cos(v * 2 * np.pi) * (distance > .013)
        return as_color(response, (0, .008, .003))
    if pattern == "edge_card":
        result = gray * .82 + .045
        patches = (
            (.50, .50, .10, .16, "v", .12),
            (.18, .22, .10, .10, "h", .10),
            (.80, .24, .12, .10, "h", -.12),
            (.24, .76, .10, .14, "v", -.10),
            (.78, .74, .12, .14, "v", .11),
        )
        for cu, cv, half_u, half_v, orientation, slope in patches:
            inside = (np.abs(u - cu) < half_u) & (np.abs(v - cv) < half_v)
            if orientation == "v":
                distance = (u - cu) - slope * (v - cv)
            else:
                distance = (v - cv) - slope * (u - cu)
            edge = .16 + .68 * (.5 + .5 * np.tanh(distance * 1000))
            result = np.where(inside, edge, result)
        return as_color(result, (0, .006, .002))
    if pattern == "reflection":
        bright = gray + .19 * bump(u, v, .21, .36, .017, .034)
        bright += .20 * bump(u, v, .64, .35, .024, .017)
        bright += .12 * bump(u, v, .68, .62, .016, .019)
        return as_color(bright, (.010, 0, -.003))
    if pattern.startswith("band"):
        harmonics = {"band_a": (21.4, .059, .30), "band_b": (17.6, .039, 1.10),
                     "band_c": (24.5, .050, 2.04)}
        frequency, amplitude, phase = harmonics[pattern]
        wave = np.sin(2 * np.pi * (frequency * v + .50 * u) + phase)
        second = np.sin(2 * np.pi * (2 * frequency * v + .55 * u) - .40 * phase)
        banded = gray + amplitude * wave + .014 * second
        return as_color(banded, (.012, 0, -.010))
    if pattern == "module_seam":
        axis = .51 + .006 * (v - .58) + .003 * np.sin(2 * np.pi * v)
        width = .006 + .004 * bump(u, v, .5, .64, .25, .21)
        delta = np.exp(-.5 * ((u - axis) / width) ** 2)
        support = .92 - .36 * bump(u, v, .52, .41, .32, .035)
        step = .025 * (u > axis)
        return as_color(gray * .94 + step - .13 * delta * support, (.013, .004, 0))
    if pattern == "display_rim":
        left = .043 - .011 * np.sin(np.pi * v)
        right = .949 + .010 * np.sin(np.pi * v)
        top = .057 - .007 * np.sin(np.pi * u)
        bottom = .943 + .006 * np.sin(np.pi * u)
        inward = np.minimum.reduce((u - left, right - u, v - top, bottom - v))
        border = .5 + .5 * np.tanh(inward * 500)
        shading = gray + .065
        return as_color(.060 * (1 - border) + shading * border, (.008, 0, -.01))
    if pattern == "white":
        panel = .79 + .08 * (gray - .31) + .024 * np.sin(np.pi * v)
        return as_color(panel, (.008, 0, -.012))
    if pattern in {"gray45", "fade1", "fade2", "fade3"}:
        level = {"gray45": 0, "fade1": .078, "fade2": .041, "fade3": .010}[pattern]
        shape = (u > .27) & (u < .68) & (v > .33) & (v < .60)
        modulation = .85 + .15 * np.sin(2 * np.pi * (3 * u + .14))
        retained = gaussian_filter((shape * modulation).astype(float), 1.7)
        return as_color(gray * .86 + retained * level, (.009, 0, -.007))
    raise ValueError(f"Unknown synthetic optical pattern {pattern!r}")


def panel_environment(x, y):
    border = .047 + .023 * bump(x, y, .54, .68, .53, .40)
    line = np.round(y * 35)
    grain = np.sin(line * 3.3) * .002
    return as_color(border + grain, (.008, .002, .010))


def lens_grid(width: int, height: int):
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    u, v = np.meshgrid(x, y)
    falloff = .85 + .13 * bump(u, v, .51, .42, .37, .36)
    gain = np.stack([falloff * 1.012, falloff, falloff * .993], axis=-1)
    bias = as_color(.016 + .006 * v + .002 * np.sin(np.pi * u), (.003, 0, .004))
    return u, v, gain, bias


def sensor_reference(pattern: str, seed: int) -> np.ndarray:
    _, _, gain, bias = lens_grid(WIDTH, HEIGHT)
    generator = np.random.default_rng(seed)
    if pattern == "dark":
        frame = bias + generator.normal(0, .0014, bias.shape)
        for y, x, amplitude in ((41, 202, .12), (132, 520, .16), (311, 480, .15), (400, 630, .13)):
            frame[y, x, :] += amplitude
        sporadic = {77: (421, 240), 101: (243, 264)}.get(seed)
        if sporadic is not None:
            frame[sporadic[0], sporadic[1], :] += .10
        return np.clip(frame, 0, 1)
    field = gain * .81 + bias
    frame = field + generator.normal(0, .0019, field.shape)
    return np.clip(frame, 0, 1)


def synth_macro(seed: int) -> np.ndarray:
    height, width = 350, 640
    grid_y, grid_x = np.indices((height, width))
    period_x, period_y = 15.4, 12.3
    phase_x = ((grid_x + .10 * grid_y) % period_x) / period_x
    phase_y = ((grid_y - .035 * grid_x) % period_y) / period_y
    bright = np.exp(-.5 * ((phase_y - .50) / .29) ** 2)
    rgb = []
    for center, scale in ((.19, .76), (.50, .91), (.80, .79)):
        streak = np.exp(-.5 * ((phase_x - center) / .082) ** 2)
        modulation = .90 + .1 * np.sin(2 * np.pi * grid_x / width)
        rgb.append(.051 + scale * streak * bright * modulation)
    stacked = np.stack(rgb, axis=-1)
    photo = gaussian_filter(stacked, (1.05, 1.05, 0))
    noise = np.random.default_rng(seed).normal(0, .007, size=photo.shape)
    return np.clip(photo + noise, 0, 1)


def synth_capture(source: DemoSource):
    if source.photo in {"dark", "flat"}:
        photo = sensor_reference(source.photo, source.seed)
        return photo, None
    if source.photo == "macro":
        return synth_macro(source.seed), None
    calibration = Calibration.from_raw(CORNERS[source.pose])
    x, y, gain, bias = lens_grid(WIDTH, HEIGHT)
    uv = calibration.camera_to_panel(np.stack([x, y], axis=-1))
    valid_panel = np.all((uv >= 0) & (uv <= 1), axis=-1)
    u = np.clip(uv[..., 0], 0, 1)
    v = np.clip(uv[..., 1], 0, 1)
    rendered = optics_panel(source.photo, u, v)
    camera = np.where(valid_panel[..., None], rendered, panel_environment(x, y))
    camera = gain * camera + bias
    noise = np.random.default_rng(source.seed).normal(0, .0025, camera.shape)
    camera = gaussian_filter(camera, (.56, .56, 0)) + noise
    return np.clip(camera, 0, 1), calibration


def timestamp(index):
    offsets = {15: 28, 16: 30.2, 17: 31.0, 18: 33.0}
    return FIRST_CAPTURE + timedelta(seconds=offsets.get(index, index * 1.35 + .021))


def ordered_specs():
    for index, source in enumerate(SOURCES):
        image, calibration = synth_capture(source)
        png = png_bytes(np.round(255 * image).astype(np.uint8))
        asset = decode_asset(png)
        kind = "用于界面练习的程序合成图，无实体面板或相机测量。"
        exposure = {
            "title": source.title,
            "role": source.role,
            "shutter_ms": source.shutter,
            "iso": 100,
            "camera": CAMERA,
            "captured_at": timestamp(index).isoformat(timespec="milliseconds"),
            "angle_deg": source.angle,
            "level": source.level,
            "note": kind,
        }
        spec = CaptureSpec.parse(exposure, f"synthetic-{source.key}.png")
        overview = calibration.overview((asset.height, asset.width)) if calibration else None
        yield spec, asset, overview


def install(store: Store):
    older = store.find_seed(DEMO_KEY)
    if older:
        return older
    panel = PanelSpec.parse({
        "name": "DG-1080 / 合成光学台",
        "station": "演示工位",
        "native_width": PANEL_WIDTH,
        "native_height": PANEL_HEIGHT,
        "width_mm": 260,
        "height_mm": 156,
        "note": "所有采集均为有标签的合成素材，可操作完整检查流程。",
        "synthetic": True,
    })
    return store.install_demo(DEMO_KEY, panel, list(ordered_specs()))


def demo_trigger_log():
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("timestamp", "event", "level"))
    offsets_ms = [0, -4, 2, 5, -2, 4, -3]
    for index, source in enumerate(SOURCES):
        time = timestamp(index) + timedelta(seconds=.001 * (source.shutter * .5 + offsets_ms[index % len(offsets_ms)]))
        writer.writerow((time.isoformat(timespec="milliseconds"), f"synthetic-{source.key}", source.level))
    return stream.getvalue()
