"""High-level renderer facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app import ELRApp
from .camera import FirstPersonCameraController, FrameTimer
from .device import DeviceBackend, DeviceConfig, create_slangpy_device
from .paths import GRAPH_DIR, SHADER_DIR
from .passes import pass_from_config
from .render_graph import RenderContext, RenderGraph
from .scene import SceneLoader


@dataclass
class FrameStats:
    frame_index: int
    graph_name: str
    output: Any
    timings: dict[str, float] | None = None


@dataclass
class RendererConfig:
    scene_path: str | Path | None = None
    graph_name: str = "slangpy_preview"
    backend: DeviceBackend | str = DeviceBackend.automatic
    width: int = 1280
    height: int = 720
    enable_debug: bool = False
    enable_raytracing: bool = False
    interactive: bool = False


class Renderer:
    def __init__(self, config: RendererConfig) -> None:
        self.config = config
        self.scene = SceneLoader().load(config.scene_path)
        self.app = None
        self.camera_controller = None
        self._timer = FrameTimer()
        device_config = DeviceConfig(backend=config.backend, enable_debug=config.enable_debug)
        if config.interactive:
            self.app = ELRApp(
                width=config.width,
                height=config.height,
                device_config=device_config,
                include_paths=[SHADER_DIR],
            )
            self.device = self.app.device
            self.camera_controller = FirstPersonCameraController(speed=max(self.scene.camera_speed, 0.1))
            self.app.on_keyboard_event = self.camera_controller.on_keyboard_event
            self.app.on_mouse_event = self.camera_controller.on_mouse_event
        else:
            self.device = create_slangpy_device(device_config, include_paths=[SHADER_DIR])
        self.graph = load_graph(config.graph_name)
        self.context = RenderContext(
            device=self.device,
            width=config.width,
            height=config.height,
            scene=self.scene,
            app=self.app,
            shader_paths=[SHADER_DIR],
            settings={"enable_raytracing": config.enable_raytracing},
        )

    def load_scene(self, path: str | Path | None) -> None:
        self.scene = SceneLoader().load(path)
        self.context.scene = self.scene
        if self.camera_controller is not None:
            self.camera_controller.speed = max(self.scene.camera_speed, 0.1)

    def load_graph(self, name_or_path: str | Path) -> None:
        self.graph = load_graph(name_or_path)

    def frame(self) -> FrameStats:
        if self.app is not None:
            self.context.width = self.app.width
            self.context.height = self.app.height
        dt = self._timer.tick(self.app is not None)
        self.context.frame.time_seconds += dt
        if self.camera_controller is not None:
            self.camera_controller.update(self.scene.active_camera, dt)
        self.context.resources.clear_transient()
        self.context.timings.clear()
        self.graph.execute(self.context)
        return FrameStats(
            frame_index=self.context.frame.frame_index - 1,
            graph_name=self.graph.name,
            output=self.context.output,
            timings=dict(self.context.timings),
        )


def load_graph(name_or_path: str | Path) -> RenderGraph:
    candidate = Path(name_or_path)
    if not candidate.suffix:
        candidate = GRAPH_DIR / f"{candidate}.json"
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate if candidate.exists() else candidate
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    graph = RenderGraph.from_json(candidate, pass_from_config)
    graph.compile()
    return graph
