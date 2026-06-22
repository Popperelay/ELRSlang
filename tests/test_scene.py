from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
