"""Interactive camera controller helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np

from .scene import Camera, normalize


def _event_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name is None:
        name = str(value)
    return str(name).lower().replace("keycode.", "").replace("key_", "")


@dataclass
class FirstPersonCameraController:
    speed: float = 1.0
    mouse_sensitivity: float = 0.0025
    keys_down: set[str] = field(default_factory=set)
    mouse_look: bool = False
    last_mouse_pos: tuple[float, float] | None = None
    yaw: float | None = None
    pitch: float | None = None

    def on_keyboard_event(self, event: Any) -> None:
        key = _event_name(getattr(event, "key", ""))
        event_type = _event_name(getattr(event, "type", ""))
        if "press" in event_type or "down" in event_type:
            self.keys_down.add(key)
        elif "release" in event_type or "up" in event_type:
            self.keys_down.discard(key)

    def on_mouse_event(self, event: Any) -> None:
        event_type = _event_name(getattr(event, "type", ""))
        pos = getattr(event, "pos", None)
        if pos is not None:
            if hasattr(pos, "x") and hasattr(pos, "y"):
                current = (float(pos.x), float(pos.y))
            else:
                current = (float(pos[0]), float(pos[1]))
        else:
            current = self.last_mouse_pos
        if "button" in event_type:
            button = _event_name(getattr(event, "button", ""))
            if "right" in button or button in {"1", "2"}:
                self.mouse_look = "down" in event_type or "press" in event_type
                self.last_mouse_pos = current
        elif "move" in event_type and self.mouse_look and current and self.last_mouse_pos:
            dx = current[0] - self.last_mouse_pos[0]
            dy = current[1] - self.last_mouse_pos[1]
            if self.yaw is not None and self.pitch is not None:
                self.yaw -= dx * self.mouse_sensitivity
                self.pitch = max(-1.55, min(1.55, self.pitch - dy * self.mouse_sensitivity))
            self.last_mouse_pos = current
        elif "move" in event_type:
            self.last_mouse_pos = current

    def update(self, camera: Camera, dt: float) -> bool:
        self._ensure_angles(camera)
        assert self.yaw is not None and self.pitch is not None
        forward = np.asarray(
            [
                math.cos(self.pitch) * math.sin(self.yaw),
                math.sin(self.pitch),
                -math.cos(self.pitch) * math.cos(self.yaw),
            ],
            dtype=np.float32,
        )
        forward = normalize(forward, (0.0, 0.0, -1.0))
        right = normalize(np.cross(forward, np.asarray(camera.up, dtype=np.float32)), (1.0, 0.0, 0.0))
        up = normalize(np.cross(right, forward), (0.0, 1.0, 0.0))
        move = np.zeros(3, dtype=np.float32)
        if "w" in self.keys_down:
            move += forward
        if "s" in self.keys_down:
            move -= forward
        if "a" in self.keys_down:
            move -= right
        if "d" in self.keys_down:
            move += right
        if "q" in self.keys_down:
            move -= up
        if "e" in self.keys_down:
            move += up
        if np.linalg.norm(move) > 1e-5:
            move = normalize(move, (0.0, 0.0, 0.0))
        boost = 4.0 if "leftshift" in self.keys_down or "shift" in self.keys_down else 1.0
        slow = 0.25 if "leftcontrol" in self.keys_down or "ctrl" in self.keys_down else 1.0
        old_position = np.asarray(camera.position, dtype=np.float32)
        new_position = old_position + move * float(self.speed) * boost * slow * max(dt, 0.0)
        changed = bool(np.linalg.norm(new_position - old_position) > 1e-7)
        new_target = new_position + forward
        if changed or self.mouse_look:
            camera.position = tuple(float(v) for v in new_position)
            camera.target = tuple(float(v) for v in new_target)
            camera.up = tuple(float(v) for v in up)
            return True
        return False

    def _ensure_angles(self, camera: Camera) -> None:
        if self.yaw is not None and self.pitch is not None:
            return
        direction = np.asarray(camera.target, dtype=np.float32) - np.asarray(camera.position, dtype=np.float32)
        direction = normalize(direction, (0.0, 0.0, -1.0))
        self.pitch = math.asin(float(np.clip(direction[1], -1.0, 1.0)))
        self.yaw = math.atan2(float(direction[0]), float(-direction[2]))


class FrameTimer:
    def __init__(self) -> None:
        self._last = time.perf_counter()

    def tick(self, interactive: bool) -> float:
        if not interactive:
            return 1.0 / 60.0
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        return max(0.0, min(dt, 0.25))
