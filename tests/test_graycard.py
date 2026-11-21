"""Gray reference card correction: sampling, rejection, models and persistence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lumagrid.assays import color_shift, configs
from lumagrid.graycard import (
    ANCHOR_BAND, STATUS_BRACKETED, STATUS_EXTRAPOLATED, STATUS_NEAR, STATUS_UNREFERENCED,
    GrayBox, evaluate, parse_gray_boxes, read_patch,
)
from lumagrid.geometry import Calibration
from lumagrid.inspection import Scene, execute
from lumagrid.model import Problem, Region
from lumagrid.raster import DecodedRaster, decode


def make_raster(rgb, valid=None):
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    if valid is None:
        valid = np.ones(rgb.shape[:2], dtype=bool)
    return DecodedRaster(rgb, valid, "RGB", 8, "PNG")


def make_scene(rgb, scene_id, panel_id="panel-1", identity=True):
    height, width = rgb.shape[:2]
    capture = {"id": scene_id, "title": scene_id, "original": "x", "role": "uniform",
               "exposure": {}}
    panel = {"id": panel_id, "native_width": width, "native_height": height,
             "width_mm": None, "height_mm": None, "synthetic": False}
    calibration = Calibration.identity() if identity else None
    revision = {"id": "identity", "revision": None, "corners": calibration.corners.tolist()}
    return Scene(panel, capture, revision, calibration, make_raster(rgb))


def fill_panel(width=160, height=120, value=(0.4, 0.4, 0.4)):
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[:, :] = np.asarray(value, dtype=np.float32)
    return rgb


def card(rgb, box, value, noise=0.001, seed=0):
    x0, y0, x1, y1 = box
    height, width = rgb.shape[:2]
    px0, px1 = int(x0 * width), int(x1 * width)
    py0, py1 = int(y0 * height), int(y1 * height)
    rng = np.random.default_rng(seed)
    patch = np.asarray(value, dtype=np.float32) + rng.normal(0, noise, (py1 - py0, px1 - px0, 3))
    rgb[py0:py1, px0:px1] = patch


class GrayBoxParsingTests(unittest.TestCase):
    def test_parse_groups_by_scene(self):
        boxes = parse_gray_boxes({
            "primary": [{"level": "light", "left": .1, "top": .1, "right": .2, "bottom": .2}],
            "comparison": [
                {"level": "light", "left": .1, "top": .1, "right": .2, "bottom": .2},
                {"level": "dark", "left": .1, "top": .3, "right": .2, "bottom": .4},
            ],
        })
        self.assertEqual([(b.scene, b.level) for b in boxes],
                         [("primary", "light"), ("comparison", "light"), ("comparison", "dark")])

    def test_duplicate_level_rejected(self):
        with self.assertRaises(Problem):
            parse_gray_boxes({"primary": [
                {"level": "light", "left": .1, "top": .1, "right": .2, "bottom": .2},
                {"level": "light", "left": .3, "top": .1, "right": .4, "bottom": .2},
            ], "comparison": []})

    def test_tiny_box_rejected(self):
        with self.assertRaises(Problem):
            GrayBox.parse({"level": "light", "left": .1, "top": .1, "right": .1001, "bottom": .2},
                          "primary")


class PatchReadingTests(unittest.TestCase):
    def test_uniform_patch_is_usable(self):
        rgb = fill_panel()
        box = GrayBox("primary", "light", .1, .1, .25, .3)
        card(rgb, (.1, .1, .25, .3), (.6, .6, .6))
        scene = make_scene(rgb, "c1")
        sample = read_patch(scene, box)
        self.assertTrue(sample.usable, sample.reason)
        self.assertGreater(sample.inlier_count, 24)
        self.assertLess(sample.uniform_cv, .02)

    def test_specular_reflection_rejected(self):
        rgb = fill_panel()
        box = GrayBox("primary", "light", .1, .1, .4, .5)
        card(rgb, (.1, .1, .4, .5), (.5, .5, .5), noise=.001)
        height, width = rgb.shape[:2]
        # Specular streak covering roughly a tenth of the card patch.
        cx0, cx1 = int(.18 * width), int(.33 * width)
        cy0, cy1 = int(.22 * height), int(.38 * height)
        rgb[cy0:cy1, cx0:cx1] = (.97, .95, .92)
        scene = make_scene(rgb, "c1")
        sample = read_patch(scene, box)
        self.assertFalse(sample.usable)
        self.assertIn("反光", sample.reason)

    def test_saturated_card_rejected(self):
        rgb = fill_panel()
        box = GrayBox("primary", "light", .1, .1, .35, .4)
        card(rgb, (.1, .1, .35, .4), (.999, .999, .999), noise=0.0)
        scene = make_scene(rgb, "c1")
        sample = read_patch(scene, box)
        self.assertFalse(sample.usable)
        self.assertIn("极限", sample.reason)

    def test_clipped_card_rejected_and_keeps_inlier_stats(self):
        rgb = fill_panel(width=160, height=120)
        box = GrayBox("primary", "light", -.06, .1, .04, .4)
        card(rgb, (0, .1, .04, .4), (.55, .55, .55))
        scene = make_scene(rgb, "c1")
        sample = read_patch(scene, box)
        self.assertFalse(sample.usable)
        self.assertIn("出画", sample.reason)
        self.assertLess(sample.coverage_pct, .55)
        self.assertGreater(sample.inlier_count, 0)

    def test_alpha_invalid_region_rejected(self):
        rgb = fill_panel()
        valid = np.ones(rgb.shape[:2], dtype=bool)
        height, width = rgb.shape[:2]
        valid[:, : int(.18 * width)] = False
        box = GrayBox("primary", "light", .05, .1, .16, .4)
        scene = Scene(make_scene(rgb, "c1").panel, make_scene(rgb, "c1").capture,
                      {"id": "identity"}, Calibration.identity(), make_raster(rgb, valid))
        sample = read_patch(scene, box)
        self.assertFalse(sample.usable)


class CorrectionModelTests(unittest.TestCase):
    def _pair_with_warm_fill(self):
        width, height = 160, 120
        primary = fill_panel(width, height, (.42, .40, .38))
        secondary = fill_panel(width, height, (.42, .40, .38))
        # Secondary viewpoint receives a warm fill: reds up, blues down.
        secondary = np.clip(secondary * np.array([1.12, 1.0, .86], np.float32), 0, 1)
        light_box = (.08, .08, .22, .26)
        dark_box = (.08, .32, .22, .5)
        card(primary, light_box, (.62, .62, .62), seed=1)
        card(primary, dark_box, (.18, .18, .18), seed=2)
        card(secondary, light_box, (.62 * 1.12, .62, .62 * .86), seed=3)
        card(secondary, dark_box, (.18 * 1.12, .18, .18 * .86), seed=4)
        boxes = [
            GrayBox("primary", "light", *light_box), GrayBox("primary", "dark", *dark_box),
            GrayBox("comparison", "light", *light_box), GrayBox("comparison", "dark", *dark_box),
        ]
        return make_scene(primary, "primary"), make_scene(secondary, "secondary"), boxes

    def test_two_level_affine_recovers_warm_fill(self):
        first, second, boxes = self._pair_with_warm_fill()
        correction = evaluate(boxes, first, second)
        self.assertTrue(correction.available)
        self.assertEqual(correction.method, "two_level_affine")
        self.assertEqual(correction.reliable_levels, ["light", "dark"])
        # In sRGB-linear space a pure per-channel multiplier k maps source
        # medians s to target medians t = k*s; with two anchors the affine fit
        # must recover k and near-zero offsets on every channel.
        primary = {p.level: p.median_linear for p in correction.patches if p.scene == "primary"}
        source = {p.level: p.median_linear for p in correction.patches if p.scene == "comparison"}
        expected_gains = (primary["light"] - primary["dark"]) / (source["light"] - source["dark"])
        np.testing.assert_allclose(correction.gains, expected_gains, atol=.01)
        np.testing.assert_array_less(correction.gains[[0, 1]], [1.0, 1.001])
        np.testing.assert_array_less(1.0, correction.gains[2])
        np.testing.assert_allclose(correction.offsets, 0, atol=.02)
        # Applying the correction maps the secondary card onto the primary card.
        fixed = correction.apply(source["light"])
        np.testing.assert_allclose(fixed, primary["light"], atol=.01)

    def test_color_shift_shrinks_after_correction(self):
        first, second, boxes = self._pair_with_warm_fill()
        options = {"grid": 4, "neutralize": True,
                   "gray_boxes": {
                       "primary": [b.as_value() for b in boxes if b.scene == "primary"],
                       "comparison": [b.as_value() for b in boxes if b.scene == "comparison"],
                   }}
        report = color_shift(first, second, Region(0, 0, 1, 1), options)
        self.assertLess(report.metrics["corrected_median_chromatic_shift"],
                        report.metrics["median_chromatic_shift"])
        names = set(report.files)
        self.assertIn("color-delta-before-after.png", names)
        self.assertIn("graycard-evidence.png", names)
        self.assertIn("graycard-patches.csv", names)
        statuses = {row["reference_status"] for row in report.rows}
        self.assertIn(STATUS_BRACKETED, statuses)

    def test_single_level_gain_marks_extrapolation(self):
        width, height = 160, 120
        primary = fill_panel(width, height, (.7, .68, .66))
        secondary = np.clip(fill_panel(width, height, (.7, .68, .66))
                            * np.array([1.1, 1.0, .9], np.float32), 0, 1)
        light_box = (.08, .08, .24, .28)
        card(primary, light_box, (.55, .55, .55), seed=5)
        card(secondary, light_box, (.55 * 1.1, .55, .55 * .9), seed=6)
        boxes = [GrayBox("primary", "light", *light_box),
                 GrayBox("comparison", "light", *light_box)]
        correction = evaluate(boxes, make_scene(primary, "p"), make_scene(secondary, "s"))
        self.assertEqual(correction.method, "single_level_gain")
        anchor = correction.target_luma["light"]
        self.assertEqual(correction.cell_status(anchor), STATUS_NEAR)
        self.assertEqual(correction.cell_status(anchor * (1 + ANCHOR_BAND * .5)), STATUS_NEAR)
        self.assertEqual(correction.cell_status(anchor * 1.6), STATUS_EXTRAPOLATED)
        self.assertEqual(correction.cell_status(None), STATUS_UNREFERENCED)

    def test_inverted_levels_raise(self):
        first, second, boxes = self._pair_with_warm_fill()
        # Swap the primary level labels so light is actually darker.
        broken = [
            GrayBox("primary", "dark", .08, .08, .22, .26),
            GrayBox("primary", "light", .08, .32, .22, .5),
            GrayBox("comparison", "light", .08, .08, .22, .26),
            GrayBox("comparison", "dark", .08, .32, .22, .5),
        ]
        with self.assertRaises(Problem):
            evaluate(broken, first, second)

    def test_no_shared_level_returns_unavailable_with_warning(self):
        width, height = 160, 120
        primary = fill_panel(width, height)
        secondary = fill_panel(width, height)
        good = (.08, .08, .24, .28)
        card(primary, good, (.6, .6, .6), seed=7)
        # Secondary card is fully clipped out of frame.
        bad = GrayBox("comparison", "light", -.2, .1, -.05, .4)
        boxes = [GrayBox("primary", "light", *good), bad]
        correction = evaluate(boxes, make_scene(primary, "p"), make_scene(secondary, "s"))
        self.assertFalse(correction.available)
        self.assertTrue(any("共同可靠" in w for w in correction.warnings))

    def test_reflection_on_one_capture_falls_back_to_other_level(self):
        first, second, boxes = self._pair_with_warm_fill()
        # Ruin the primary light patch with a strong glare.
        height, width = first.image.rgb.shape[:2]
        cx, cy = int(.15 * width), int(.17 * height)
        first.image.rgb[cy - 5:cy + 5, cx - 7:cx + 7] = (.98, .97, .95)
        correction = evaluate(boxes, first, second)
        self.assertTrue(correction.available)
        self.assertEqual(correction.reliable_levels, ["dark"])
        self.assertEqual(correction.method, "single_level_gain")
        self.assertTrue(any("反光" in p.reason for p in correction.patches if not p.usable))

    def test_implausible_gain_disables_correction(self):
        width, height = 160, 120
        primary = fill_panel(width, height, (.4, .4, .4))
        secondary = fill_panel(width, height, (.4, .4, .4))
        light_box = (.08, .08, .24, .28)
        card(primary, light_box, (.7, .7, .7), seed=11)
        # Near-black "card" region on the second capture forces a huge gain.
        card(secondary, light_box, (.012, .012, .012), noise=0.0, seed=12)
        boxes = [GrayBox("primary", "light", *light_box),
                 GrayBox("comparison", "light", *light_box)]
        correction = evaluate(boxes, make_scene(primary, "p"), make_scene(secondary, "s"))
        self.assertFalse(correction.available)
        self.assertTrue(any("0.5～2.0" in w for w in correction.warnings))


class ColorShiftBackwardCompatibilityTests(unittest.TestCase):
    def test_without_gray_boxes_keeps_legacy_artifact(self):
        primary = fill_panel(160, 120, (.4, .4, .4))
        secondary = np.clip(fill_panel(160, 120, (.4, .4, .4)) * 1.08, 0, 1)
        report = color_shift(make_scene(primary, "p"), make_scene(secondary, "s"),
                             Region(0, 0, 1, 1), {"grid": 3, "neutralize": True})
        self.assertIn("color-delta.png", report.files)
        self.assertNotIn("graycard-evidence.png", report.files)
        self.assertNotIn("graycard_mode", report.metrics)
        self.assertNotIn("reference_status", report.rows[0])

    def test_color_tool_registers_graybox_option(self):
        option_types = {item["type"] for item in configs()["color"]["options"]}
        self.assertIn("graybox", option_types)


class PersistenceTests(unittest.TestCase):
    def test_saved_inspection_keeps_gray_boxes_and_evidence(self):
        from lumagrid.demo import GRAY_CARDS
        from lumagrid.storage import Store
        from lumagrid.demo import install as install_demo

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            panel = install_demo(store)
            captures = store.captures(panel["id"])
            primary = next(c for c in captures if c["filename"] == "synthetic-uniform.png")
            comparison = next(c for c in captures if c["filename"] == "synthetic-uniform_oblique.png")
            options = {
                "grid": 4,
                "neutralize": True,
                "gray_boxes": {
                    "primary": [
                        {"level": level, **dict(zip(("left", "top", "right", "bottom"), box))}
                        for level, box in GRAY_CARDS[0].items()
                        if level in ("light", "dark")
                    ],
                    "comparison": [
                        {"level": level, **dict(zip(("left", "top", "right", "bottom"), box))}
                        for level, box in GRAY_CARDS[1].items()
                        if level in ("light", "dark")
                    ],
                },
            }
            payload = {
                "tool": "color",
                "comparison": comparison["id"],
                "calibration_id": primary["calibration"]["id"],
                "comparison_calibration_id": comparison["calibration"]["id"],
                "region": None,
                "options": options,
                "save": True,
            }
            _, _, snapshot, report = execute(store, primary["id"], payload)
            self.assertEqual(snapshot["options"]["gray_boxes"]["primary"][0]["level"], "light")
            saved = store.add_inspection(
                store.capture(primary["id"]), store.capture(comparison["id"]),
                "color", "灰片校正色偏", snapshot, report.as_json(), report.files,
            )
            record = store.inspection(saved["id"])
            stored_boxes = record["request"]["options"]["gray_boxes"]
            self.assertEqual(len(stored_boxes["primary"]), 2)
            self.assertEqual(len(stored_boxes["comparison"]), 2)
            files = {item["name"] for item in record["files"]}
            self.assertIn("graycard-evidence.png", files)
            self.assertIn("graycard-patches.csv", files)
            self.assertIn("color-delta-before-after.png", files)
            evidence, mime = store.inspection_file(saved["id"], "graycard-evidence.png")
            self.assertEqual(mime, "image/png")
            decoded = decode(evidence)
            self.assertGreater(decoded.rgb.size, 0)


if __name__ == "__main__":
    unittest.main()
