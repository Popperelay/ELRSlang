from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elrslang.camera import FirstPersonCameraController
from elrslang.scene import Camera


class CameraControllerTests(unittest.TestCase):
    def test_keyboard_movement_changes_camera_position(self):
        camera = Camera()
        controller = FirstPersonCameraController(speed=2.0)

        controller.on_keyboard_event(SimpleNamespace(type="key_press", key="w"))
        changed = controller.update(camera, 0.5)

        self.assertTrue(changed)
        self.assertLess(camera.position[2], 4.0)

    def test_mouse_look_changes_camera_target(self):
        camera = Camera()
        controller = FirstPersonCameraController()
        controller.update(camera, 0.0)
        old_target = camera.target

        controller.on_mouse_event(
            SimpleNamespace(type="button_down", button="right", pos=SimpleNamespace(x=10, y=10))
        )
        controller.on_mouse_event(SimpleNamespace(type="move", pos=SimpleNamespace(x=50, y=30)))
        changed = controller.update(camera, 0.0)

        self.assertTrue(changed)
        self.assertNotEqual(old_target, camera.target)


if __name__ == "__main__":
    unittest.main()
