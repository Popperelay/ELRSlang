"""Small SlangPy device helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .paths import SHADER_DIR


class DeviceBackend(str, Enum):
    automatic = "automatic"
    d3d12 = "d3d12"
    vulkan = "vulkan"
    metal = "metal"
    wgpu = "wgpu"
    cpu = "cpu"
    cuda = "cuda"


@dataclass(frozen=True)
class DeviceConfig:
    backend: DeviceBackend | str = DeviceBackend.automatic
    enable_debug: bool = False
    enable_hot_reload: bool = True
    include_paths: tuple[Path, ...] = ()


class SlangPyUnavailable(RuntimeError):
    """Raised when SlangPy is required but cannot be imported."""


def import_slangpy():
    try:
        import slangpy as spy  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local GPU package install
        raise SlangPyUnavailable(
            "SlangPy is required for GPU execution. Install with `python -m pip install -e .`."
        ) from exc
    return spy


def parse_backend(backend: DeviceBackend | str):
    spy = import_slangpy()
    value = backend.value if isinstance(backend, DeviceBackend) else str(backend)
    try:
        return getattr(spy.DeviceType, value)
    except AttributeError as exc:
        valid = ", ".join(item.name for item in spy.DeviceType)
        raise ValueError(f"Unsupported SlangPy backend `{value}`. Valid values: {valid}") from exc


def create_slangpy_device(config: DeviceConfig, include_paths: Iterable[Path] = ()):
    spy = import_slangpy()
    merged_paths = [SHADER_DIR, *config.include_paths, *include_paths]
    return spy.create_device(
        parse_backend(config.backend),
        enable_debug_layers=config.enable_debug,
        include_paths=[str(path) for path in merged_paths],
        enable_hot_reload=config.enable_hot_reload,
    )


def has_feature(device, feature_name: str) -> bool:
    spy = import_slangpy()
    feature = getattr(spy.Feature, feature_name, None)
    if feature is None or device is None:
        return False
    try:
        return bool(device.has_feature(feature))
    except Exception:
        return False
