from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elrslang import Renderer, RendererConfig
from elrslang.device import SlangPyUnavailable
from elrslang.passes import FeatureUnavailable


class GpuSmokeTests(unittest.TestCase):
    def test_slangpy_preview_non_black(self):
        self._assert_graph_non_black("slangpy_preview")

    def test_raster_forward_non_black(self):
        self._assert_graph_non_black("raster_forward")

    def test_raster_forward_draws_falcor_teapot_mesh(self):
        try:
            renderer = Renderer(
                RendererConfig(
                    scene_path=ROOT / "assets" / "scenes" / "falcor" / "meshes" / "teapot.obj",
                    graph_name="raster_forward",
                    backend="automatic",
                    width=64,
                    height=64,
                    interactive=False,
                )
            )
            renderer.frame()
        except (SlangPyUnavailable, FeatureUnavailable) as exc:
            raise unittest.SkipTest(str(exc)) from exc
        except RuntimeError as exc:
            raise unittest.SkipTest(f"GPU smoke unavailable in this environment: {exc}") from exc

        texture = renderer.context.resources.get("HardwareRasterForward.color")
        if not hasattr(texture, "to_numpy"):
            raise unittest.SkipTest("Texture readback is unavailable in this SlangPy backend.")
        pixels = np.asarray(texture.to_numpy()).view(np.float32).reshape(64, 64, 4)
        colored = np.any(pixels[:, :, :3] > 0.01, axis=2)
        self.assertGreater(int(colored.sum()), 0)
        self.assertLess(int(colored.sum()), colored.size)

    def test_raster_forward_draws_falcor_pyscene(self):
        try:
            renderer = Renderer(
                RendererConfig(
                    scene_path=ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "cornell_box.pyscene",
                    graph_name="raster_forward",
                    backend="automatic",
                    width=64,
                    height=64,
                    interactive=False,
                )
            )
            renderer.frame()
        except (SlangPyUnavailable, FeatureUnavailable) as exc:
            raise unittest.SkipTest(str(exc)) from exc
        except RuntimeError as exc:
            raise unittest.SkipTest(f"GPU smoke unavailable in this environment: {exc}") from exc

        texture = renderer.context.resources.get("HardwareRasterForward.color")
        if not hasattr(texture, "to_numpy"):
            raise unittest.SkipTest("Texture readback is unavailable in this SlangPy backend.")
        pixels = np.asarray(texture.to_numpy()).view(np.float32).reshape(64, 64, 4)
        colored = np.any(pixels[:, :, :3] > 0.01, axis=2)
        self.assertGreater(int(colored.sum()), 0)
        self.assertLess(int(colored.sum()), colored.size)
        self._assert_color_present(pixels, (0.63, 0.065, 0.05))
        self._assert_color_present(pixels, (0.14, 0.45, 0.091))
        self._assert_color_present(pixels, (0.725, 0.71, 0.68))
        self._assert_color_present(pixels, (1.0, 1.0, 1.0))

    def test_dxr_pathtrace_non_black_when_supported(self):
        self._assert_graph_non_black("dxr_pathtrace")

    def test_hybrid_debug_non_black_when_supported(self):
        self._assert_graph_non_black("hybrid_debug")

    def _assert_graph_non_black(self, graph_name: str) -> None:
        try:
            renderer = Renderer(
                RendererConfig(
                    graph_name=graph_name,
                    backend="automatic",
                    width=32,
                    height=32,
                    interactive=False,
                )
            )
            renderer.frame()
        except (SlangPyUnavailable, FeatureUnavailable) as exc:
            raise unittest.SkipTest(str(exc)) from exc
        except RuntimeError as exc:
            raise unittest.SkipTest(f"GPU smoke unavailable in this environment: {exc}") from exc

        output = renderer.context.output
        if not hasattr(output, "to_numpy"):
            raise unittest.SkipTest("Texture readback is unavailable in this SlangPy backend.")
        pixels = np.asarray(output.to_numpy()).view(np.float32)
        self.assertTrue(np.any(pixels != 0.0), f"{graph_name} produced an all-zero image")

    def _assert_color_present(self, pixels: np.ndarray, color: tuple[float, float, float]) -> None:
        matches = np.all(np.isclose(pixels[:, :, :3], color, atol=0.02), axis=2)
        self.assertTrue(np.any(matches), f"Expected color {color} was not present in raster output")


if __name__ == "__main__":
    unittest.main()
