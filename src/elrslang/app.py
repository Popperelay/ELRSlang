"""Window and swapchain helper for the interactive viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .device import DeviceConfig, create_slangpy_device, import_slangpy


class ELRApp:
    def __init__(
        self,
        title: str = "ELRSlang",
        width: int = 1280,
        height: int = 720,
        device_config: DeviceConfig | None = None,
        include_paths: list[Path] | None = None,
        output_format: str = "rgba32_float",
    ) -> None:
        spy = import_slangpy()
        self._spy = spy
        self._window = spy.Window(width=width, height=height, title=title, resizable=True)
        self._device = create_slangpy_device(device_config or DeviceConfig(), include_paths or [])
        self._surface = self._device.create_surface(self._window)
        self._surface.configure(width=self._window.width, height=self._window.height)
        self._output_format = getattr(spy.Format, output_format)
        self._output_texture = self._create_output_texture(width, height)
        self._mouse_pos = spy.float2()
        self.on_keyboard_event: Callable | None = None
        self.on_mouse_event: Callable | None = None

        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event
        self._window.on_resize = self._on_resize

    @property
    def device(self):
        return self._device

    @property
    def window(self):
        return self._window

    @property
    def output(self):
        return self._output_texture

    @property
    def width(self) -> int:
        return self._window.width

    @property
    def height(self) -> int:
        return self._window.height

    def process_events(self) -> bool:
        if self._window.should_close():
            return False
        self._window.process_events()
        return True

    def present(self, source=None) -> None:
        if not self._surface.config:
            return
        image = self._surface.acquire_next_image()
        if not image:
            return
        source = source or self._output_texture
        command_encoder = self._device.create_command_encoder()
        command_encoder.blit(image, source)
        command_encoder.set_texture_state(image, self._spy.ResourceState.present)
        self._device.submit_command_buffer(command_encoder.finish())
        del image
        self._surface.present()

    def screenshot(self, path: str | Path = "screenshot.png") -> None:
        bitmap = self._output_texture.to_bitmap()
        bitmap.convert(
            self._spy.Bitmap.PixelFormat.rgb,
            self._spy.Bitmap.ComponentType.uint8,
            srgb_gamma=True,
        ).write_async(str(path))

    def _create_output_texture(self, width: int, height: int):
        return self._device.create_texture(
            format=self._output_format,
            width=width,
            height=height,
            mip_count=1,
            usage=self._spy.TextureUsage.shader_resource | self._spy.TextureUsage.unordered_access,
            label="app.output",
        )

    def _on_keyboard_event(self, event) -> None:
        if event.type == self._spy.KeyboardEventType.key_press:
            if event.key == self._spy.KeyCode.escape:
                self._window.close()
                return
            if event.key == self._spy.KeyCode.f2:
                self.screenshot()
                return
        if self.on_keyboard_event:
            self.on_keyboard_event(event)

    def _on_mouse_event(self, event) -> None:
        if event.type == self._spy.MouseEventType.move:
            self._mouse_pos = event.pos
        if self.on_mouse_event:
            self.on_mouse_event(event)

    def _on_resize(self, width: int, height: int) -> None:
        self._device.wait()
        if width <= 0 or height <= 0:
            self._surface.unconfigure()
            return
        self._surface.configure(width=width, height=height)
        self._output_texture = self._create_output_texture(width, height)
