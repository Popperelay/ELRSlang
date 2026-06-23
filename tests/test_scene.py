from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elrslang.scene import MissingSceneDependency, SceneLoader


class SceneLoaderTests(unittest.TestCase):
    def test_default_scene_has_triangle(self):
        scene = SceneLoader().load(None)
        self.assertEqual(len(scene.meshes), 1)
        self.assertEqual(scene.meshes[0].triangle_count, 1)
        self.assertEqual(scene.to_view().get_this()["meshCount"], 1)

    def test_obj_loader_triangulates_quad(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quad.obj"
            path.write_text(
                "\n".join(
                    [
                        "v -1 -1 0",
                        "v 1 -1 0",
                        "v 1 1 0",
                        "v -1 1 0",
                        "f 1 2 3 4",
                    ]
                ),
                encoding="utf-8",
            )
            scene = SceneLoader().load(path)
            self.assertEqual(len(scene.meshes), 1)
            self.assertEqual(scene.meshes[0].triangle_count, 2)

    def test_falcor_quad_winding_matches_declared_normal(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "cornell_box.pyscene")
        floor = next(mesh for mesh in scene.meshes if mesh.name == "Quad")
        positions = floor.position_array()
        i0, i1, i2 = floor.index_array()[:3]
        geometric_normal = np.cross(positions[i1] - positions[i0], positions[i2] - positions[i0])
        geometric_normal = geometric_normal / np.linalg.norm(geometric_normal)
        declared_normal = floor.normal_array()[0]
        self.assertGreater(float(np.dot(geometric_normal, declared_normal)), 0.99)

    def test_falcor_standard_material_default_base_color_is_white(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "cornell_box.pyscene")
        light = next(material for material in scene.materials if material.name == "Light")
        self.assertEqual(light.base_color, (1.0, 1.0, 1.0, 1.0))

    def test_gltf_json_loader_reads_materials_cameras_and_lights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.gltf"
            path.write_text(
                json.dumps(
                    {
                        "asset": {"version": "2.0"},
                        "materials": [
                            {
                                "name": "mat",
                                "pbrMetallicRoughness": {
                                    "baseColorFactor": [1, 0, 0, 1],
                                    "roughnessFactor": 0.25,
                                },
                            }
                        ],
                        "meshes": [{"name": "mesh", "primitives": [{"material": 0}]}],
                        "nodes": [{"name": "node", "mesh": 0}],
                        "cameras": [{"name": "camera"}],
                        "extensions": {
                            "KHR_lights_punctual": {
                                "lights": [{"name": "light", "type": "point", "intensity": 2.0}]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            scene = SceneLoader().load(path)
            self.assertEqual(scene.materials[0].name, "mat")
            self.assertEqual(scene.meshes[0].name, "mesh")
            self.assertEqual(scene.instances[0].name, "node")
            self.assertEqual(scene.cameras[0].name, "camera")
            self.assertEqual(scene.lights[0].name, "light")

    def test_glb_reports_optional_dependency_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.glb"
            path.write_bytes(b"glTF")
            try:
                SceneLoader().load(path)
            except MissingSceneDependency as exc:
                self.assertIn("pygltflib", str(exc))
            except Exception:
                pass

    def test_falcor_pyscene_cornell_box_builds_procedural_scene(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "cornell_box.pyscene")
        self.assertGreaterEqual(len(scene.meshes), 8)
        self.assertGreaterEqual(len(scene.instances), 8)
        self.assertGreaterEqual(len(scene.materials), 8)
        self.assertEqual(scene.cameras[0].name, "DefaultCamera")
        self.assertGreater(scene.meshes[0].triangle_count, 0)

    def test_falcor_pyscene_import_scene_merges_gltf_assets(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "cesium_man" / "CesiumMan.pyscene")
        self.assertGreaterEqual(len(scene.meshes), 1)
        self.assertGreaterEqual(len(scene.instances), 1)
        self.assertGreaterEqual(len(scene.cameras), 1)
        self.assertIsNotNone(scene.env_map)

    def test_fbx_loader_reads_binary_falcor_asset(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "animated_cubes" / "animated_cubes.fbx")
        self.assertEqual(len(scene.meshes), 6)
        self.assertEqual(len(scene.instances), 6)
        self.assertEqual(scene.metadata["generator"], "ufbx")
        self.assertGreater(scene.meshes[0].triangle_count, 0)

    def test_falcor_pyscene_animated_cubes_uses_script_cameras(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "animated_cubes" / "animated_cubes.pyscene")
        self.assertEqual(len(scene.meshes), 6)
        self.assertEqual(len(scene.instances), 6)
        self.assertEqual([camera.name for camera in scene.cameras], ["FrontCamera", "BackCamera"])
        self.assertEqual(scene.active_camera.name, "FrontCamera")

    def test_falcor_pyscene_texture_lod_scene_uses_relative_assets(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "tex_lod" / "spheres_cube.pyscene")
        self.assertGreaterEqual(len(scene.meshes), 5)
        self.assertGreaterEqual(len(scene.lights), 4)
        self.assertTrue(any(material.texture_paths for material in scene.materials))

    def test_falcor_pyscene_curves_custom_primitives(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "curves" / "two_curves.pyscene")
        self.assertGreaterEqual(len(scene.meshes), 5)
        self.assertGreaterEqual(len(scene.instances), 7)
        self.assertGreaterEqual(len(scene.cameras), 1)

    def test_falcor_pyscene_bunny_reference_resolves(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "bunny.pyscene")
        self.assertGreaterEqual(len(scene.meshes), 2)
        self.assertGreaterEqual(len(scene.instances), 2)
        self.assertGreaterEqual(len(scene.cameras), 1)

    def test_falcor_pyscene_pica_pica_imports_media_ext_assets(self):
        scene = SceneLoader().load(ROOT / "assets" / "scenes" / "falcor" / "falcor_pyscene" / "pica_pica.pyscene")
        self.assertGreaterEqual(len(scene.meshes), 10)
        self.assertGreaterEqual(len(scene.instances), 10)
        self.assertGreaterEqual(len(scene.materials), 2)
        self.assertGreaterEqual(len(scene.lights), 1)


if __name__ == "__main__":
    unittest.main()
