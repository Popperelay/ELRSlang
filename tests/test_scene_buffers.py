from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elrslang.scene import Material, Mesh, MeshInstance, Scene, SceneLoader
from elrslang.scene_buffers import (
    RasterBakeSettings,
    RasterDrawList,
    build_raster_scene_buffers,
    build_world_scene_buffers,
    raster_scene_cache_key,
)


class SceneBufferTests(unittest.TestCase):
    def test_falcor_diffuse_raster_bake_contains_cornell_material_colors(self):
        scene = SceneLoader().load(
            ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "cornell_box.pyscene"
        )
        settings = RasterBakeSettings(64, 64, (0.1, 0.45, 0.9, 1.0), "falcor_diffuse")

        vertices, indices = build_raster_scene_buffers(scene, settings)
        packed_vertices = vertices.reshape(-1, 7)

        self.assertEqual(indices.size, packed_vertices.shape[0])
        self._assert_color_present(packed_vertices[:, 3:6], (0.63, 0.065, 0.05))
        self._assert_color_present(packed_vertices[:, 3:6], (0.14, 0.45, 0.091))
        self._assert_color_present(packed_vertices[:, 3:6], (0.725, 0.71, 0.68))
        self._assert_color_present(packed_vertices[:, 3:6], (1.0, 1.0, 1.0))

    def test_raster_cache_key_tracks_render_mode_and_fallback_color(self):
        scene = Scene.default()
        base = raster_scene_cache_key(
            scene, RasterBakeSettings(32, 32, (0.1, 0.2, 0.3, 1.0), "lit")
        )
        different_mode = raster_scene_cache_key(
            scene, RasterBakeSettings(32, 32, (0.1, 0.2, 0.3, 1.0), "falcor_diffuse")
        )
        different_fallback = raster_scene_cache_key(
            scene, RasterBakeSettings(32, 32, (0.3, 0.2, 0.1, 1.0), "lit")
        )

        self.assertNotEqual(base, different_mode)
        self.assertNotEqual(base, different_fallback)

    def test_world_scene_buffers_are_indexable(self):
        scene = SceneLoader().load(
            ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "cornell_box.pyscene"
        )

        vertices, indices = build_world_scene_buffers(scene)

        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(vertices.shape[1], 3)
        self.assertGreater(indices.size, 0)
        self.assertLess(int(indices.max()), vertices.shape[0])

    def test_raster_draw_list_filters_meshes(self):
        scene = Scene(
            meshes=[
                Mesh(
                    name="RedMesh",
                    positions=[(-0.8, -0.6, 0.0), (-0.1, -0.6, 0.0), (-0.45, 0.4, 0.0)],
                    normals=[(0.0, 0.0, 1.0)] * 3,
                    indices=[0, 1, 2],
                    material_index=0,
                ),
                Mesh(
                    name="GreenMesh",
                    positions=[(0.1, -0.6, 0.0), (0.8, -0.6, 0.0), (0.45, 0.4, 0.0)],
                    normals=[(0.0, 0.0, 1.0)] * 3,
                    indices=[0, 1, 2],
                    material_index=1,
                ),
            ],
            materials=[
                Material(name="RedMaterial", base_color=(1.0, 0.0, 0.0, 1.0)),
                Material(name="GreenMaterial", base_color=(0.0, 1.0, 0.0, 1.0)),
            ],
            instances=[
                MeshInstance(name="RedInstance", mesh_index=0),
                MeshInstance(name="GreenInstance", mesh_index=1),
            ],
        )
        settings = RasterBakeSettings(
            64,
            64,
            (0.1, 0.2, 0.3, 1.0),
            "falcor_diffuse",
            RasterDrawList(meshes=("GreenMesh",)),
        )

        vertices, _ = build_raster_scene_buffers(scene, settings)
        colors = vertices.reshape(-1, 7)[:, 3:6]

        self.assertFalse(np.any(np.all(np.isclose(colors, (1.0, 0.0, 0.0)), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(colors, (0.0, 1.0, 0.0)), axis=1)))

    def _assert_color_present(
        self, colors: np.ndarray, expected: tuple[float, float, float]
    ) -> None:
        matches = np.all(np.isclose(colors, expected, atol=0.02), axis=1)
        self.assertTrue(np.any(matches), f"Expected color {expected} was not baked")


if __name__ == "__main__":
    unittest.main()
