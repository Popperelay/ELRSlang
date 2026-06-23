"""GPU helper utilities shared by render passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import SHADER_DIR
from .render_graph import RenderContext
from .resources import TextureDesc


def create_texture(
    context: RenderContext,
    spy,
    key: str,
    fmt_name: str = "rgba32_float",
    usage=None,
    width: int | None = None,
    height: int | None = None,
):
    texture_width = int(width or context.width)
    texture_height = int(height or context.height)
    desc = context.resources.describe(key)
    existing = context.resources.get(key)
    if (
        existing is not None
        and desc is not None
        and desc.width == texture_width
        and desc.height == texture_height
        and desc.format == fmt_name
    ):
        return existing

    texture = context.device.create_texture(
        width=texture_width,
        height=texture_height,
        format=getattr(spy.Format, fmt_name),
        mip_count=1,
        usage=usage or (spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access),
        label=key,
    )
    context.resources.set(key, texture, TextureDesc(texture_width, texture_height, fmt_name, key))
    return texture


def build_acceleration_structure(device, spy, inputs, label: str, keepalive: list[Any] | None = None):
    desc = spy.AccelerationStructureBuildDesc()
    desc.inputs = inputs
    sizes = device.get_acceleration_structure_sizes(desc)
    scratch = device.create_buffer(
        size=sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        label=f"{label}.scratch",
    )
    accel = device.create_acceleration_structure(size=sizes.acceleration_structure_size, label=label)
    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(desc=desc, dst=accel, src=None, scratch_buffer=scratch)
    device.submit_command_buffer(command_encoder.finish())
    if keepalive is not None:
        keepalive.append(scratch)
    return accel


def resolve_shader_path(path: str | Path, extra_paths: list[Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in [Path.cwd(), SHADER_DIR, *extra_paths]:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return SHADER_DIR / candidate


def slang_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(SHADER_DIR.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def enum_value(enum_scope, name: str):
    try:
        return getattr(enum_scope, name)
    except AttributeError as exc:
        enum_name = getattr(enum_scope, "__name__", str(enum_scope))
        raise ValueError(f"Unsupported {enum_name} value `{name}`.") from exc
