"""Reusable hardware raster pipeline primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .device import has_feature, import_slangpy
from .gpu import create_texture, enum_value, resolve_shader_path, slang_source_path
from .pipeline import FeatureUnavailable, PipelinePass
from .render_graph import PassReflection, RenderContext, ResourceDesc


@dataclass(frozen=True)
class VertexElementDesc:
    semantic_name: str
    format: str
    offset: int
    semantic_index: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VertexElementDesc":
        if isinstance(config, cls):
            return config
        return cls(
            semantic_name=str(config["semantic_name"]),
            semantic_index=int(config.get("semantic_index", 0)),
            format=str(config["format"]),
            offset=int(config["offset"]),
        )

    def to_rhi(self, spy) -> dict[str, Any]:
        return {
            "semantic_name": self.semantic_name,
            "semantic_index": self.semantic_index,
            "format": getattr(spy.Format, self.format),
            "offset": self.offset,
        }


@dataclass(frozen=True)
class VertexStreamDesc:
    stride: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VertexStreamDesc":
        if isinstance(config, cls):
            return config
        return cls(stride=int(config["stride"]))

    def to_rhi(self) -> dict[str, Any]:
        return {"stride": self.stride}


@dataclass(frozen=True)
class RasterTargetDesc:
    name: str = "color"
    format: str = "rgba32_float"
    clear_value: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    load_op: str = "clear"
    store_op: str = "store"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RasterTargetDesc":
        if isinstance(config, cls):
            return config
        return cls(
            name=str(config.get("name", "color")),
            format=str(config.get("format", "rgba32_float")),
            clear_value=tuple(config.get("clear_value", (0.0, 0.0, 0.0, 1.0))),
            load_op=str(config.get("load_op", "clear")),
            store_op=str(config.get("store_op", "store")),
        )

    def to_attachment(self, spy, texture) -> dict[str, Any]:
        return {
            "view": texture.create_view({}),
            "clear_value": list(self.clear_value),
            "load_op": enum_value(spy.LoadOp, self.load_op),
            "store_op": enum_value(spy.StoreOp, self.store_op),
        }


@dataclass(frozen=True)
class DepthStencilDesc:
    format: str = "d32_float"
    depth_test_enable: bool = True
    depth_write_enable: bool = True
    depth_func: str = "less"
    clear_depth: float | None = 1.0
    load_op: str = "load"
    store_op: str = "store"

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DepthStencilDesc | None":
        if config is None:
            return cls()
        if config is False:
            return None
        return cls(
            format=str(config.get("format", "d32_float")),
            depth_test_enable=bool(config.get("depth_test_enable", True)),
            depth_write_enable=bool(config.get("depth_write_enable", True)),
            depth_func=str(config.get("depth_func", "less")),
            clear_depth=config.get("clear_depth", 1.0),
            load_op=str(config.get("load_op", "load")),
            store_op=str(config.get("store_op", "store")),
        )

    def to_pipeline(self, spy) -> dict[str, Any]:
        return {
            "format": getattr(spy.Format, self.format),
            "depth_test_enable": self.depth_test_enable,
            "depth_write_enable": self.depth_write_enable,
            "depth_func": enum_value(spy.ComparisonFunc, self.depth_func),
        }

    def to_attachment(self, spy, texture) -> dict[str, Any]:
        return {
            "view": texture.create_view({}),
            "depth_load_op": enum_value(spy.LoadOp, self.load_op),
            "depth_store_op": enum_value(spy.StoreOp, self.store_op),
        }


@dataclass(frozen=True)
class RasterizerDesc:
    cull_mode: str = "back"

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "RasterizerDesc":
        if config is None:
            return cls()
        return cls(cull_mode=str(config.get("cull_mode", "back")))

    def to_rhi(self, spy) -> dict[str, Any]:
        return {"cull_mode": enum_value(spy.CullMode, self.cull_mode)}


DEFAULT_VERTEX_ELEMENTS = (
    VertexElementDesc("POSITION", "rgb32_float", 0),
    VertexElementDesc("COLOR", "rgba32_float", 12),
)
DEFAULT_VERTEX_STREAMS = (VertexStreamDesc(stride=28),)
DEFAULT_RASTER_TARGETS = (RasterTargetDesc(),)


@dataclass(frozen=True)
class RasterPipelineDesc:
    shader: str | Path = "raster_forward.slang"
    vertex_entry: str = "vertex_main"
    fragment_entry: str = "fragment_main"
    vertex_elements: tuple[VertexElementDesc, ...] = DEFAULT_VERTEX_ELEMENTS
    vertex_streams: tuple[VertexStreamDesc, ...] = DEFAULT_VERTEX_STREAMS
    primitive_topology: str = "triangle_list"
    targets: tuple[RasterTargetDesc, ...] = DEFAULT_RASTER_TARGETS
    depth_stencil: DepthStencilDesc | None = field(default_factory=DepthStencilDesc)
    rasterizer: RasterizerDesc = field(default_factory=RasterizerDesc)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RasterPipelineDesc":
        if isinstance(config, cls):
            return config
        default_target = {
            "name": config.get("output", "color"),
            "format": config.get("output_format", "rgba32_float"),
        }
        targets = tuple(
            RasterTargetDesc.from_config(item)
            for item in config.get("targets", (default_target,))
        )
        vertex_elements = tuple(
            VertexElementDesc.from_config(item)
            for item in config.get("vertex_elements", DEFAULT_VERTEX_ELEMENTS)
        )
        vertex_streams = tuple(
            VertexStreamDesc.from_config(item)
            for item in config.get("vertex_streams", DEFAULT_VERTEX_STREAMS)
        )
        return cls(
            shader=config.get("shader", "raster_forward.slang"),
            vertex_entry=str(config.get("vertex_entry", "vertex_main")),
            fragment_entry=str(config.get("fragment_entry", "fragment_main")),
            vertex_elements=vertex_elements,
            vertex_streams=vertex_streams,
            primitive_topology=str(config.get("primitive_topology", "triangle_list")),
            targets=targets,
            depth_stencil=DepthStencilDesc.from_config(config.get("depth_stencil")),
            rasterizer=RasterizerDesc.from_config(config.get("rasterizer")),
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)


@dataclass(frozen=True)
class RasterDrawData:
    vertex_buffers: tuple[Any, ...]
    index_buffer: Any
    index_count: int
    index_format: str = "uint32"

    @classmethod
    def from_single_buffer(
        cls,
        vertex_buffer,
        index_buffer,
        index_count: int,
        index_format: str = "uint32",
    ) -> "RasterDrawData":
        return cls((vertex_buffer,), index_buffer, int(index_count), index_format)


class RasterPipelinePass(PipelinePass):
    """Reusable base class for fixed-function hardware raster passes."""

    def __init__(
        self,
        name: str,
        pipeline_desc: RasterPipelineDesc,
        inputs: tuple[str, ...] = ("scene",),
    ) -> None:
        super().__init__(name)
        self.pipeline_desc = pipeline_desc
        self.inputs = tuple(inputs)
        self._pipeline = None
        self._input_layout = None
        self._depth_texture = None
        self._depth_cache_key: tuple[int, int, str] | None = None

    def reflect(self) -> PassReflection:
        resources = {
            target.name: ResourceDesc(kind="texture", format=target.format)
            for target in self.pipeline_desc.targets
        }
        if self.pipeline_desc.depth_stencil is not None:
            resources["depth"] = ResourceDesc(
                kind="texture",
                format=self.pipeline_desc.depth_stencil.format,
                internal=True,
            )
        return PassReflection.from_iterables(
            inputs=self.inputs,
            outputs=self.pipeline_desc.output_names,
            resources=resources,
        )

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        spy = import_slangpy()
        if context.device is None:
            raise RuntimeError(f"Pass `{self.name}` requires a SlangPy device.")
        if not has_feature(context.device, "rasterization"):
            raise FeatureUnavailable("Current device does not support hardware rasterization.")

        targets = self._ensure_targets(context, spy)
        self._ensure_pipeline(context, spy)
        self._ensure_depth_texture(context, spy)
        draw_data = self.prepare_draw_data(context, inputs, spy)

        command_encoder = context.device.create_command_encoder()
        if self._depth_texture is not None and self.pipeline_desc.depth_stencil is not None:
            if self.pipeline_desc.depth_stencil.clear_depth is not None:
                command_encoder.clear_texture_depth_stencil(
                    self._depth_texture,
                    depth_value=float(self.pipeline_desc.depth_stencil.clear_depth),
                )
        with command_encoder.begin_render_pass(self._render_pass_desc(spy, targets)) as encoder:
            encoder.set_render_state(
                {
                    "vertex_buffers": list(draw_data.vertex_buffers),
                    "index_buffer": draw_data.index_buffer,
                    "index_format": enum_value(spy.IndexFormat, draw_data.index_format),
                    "viewports": [spy.Viewport.from_size(context.width, context.height)],
                    "scissor_rects": [spy.ScissorRect.from_size(context.width, context.height)],
                }
            )
            shader_object = encoder.bind_pipeline(self._pipeline)
            self.bind_shader_data(shader_object, context, inputs, spy)
            self.draw(encoder, draw_data, spy)
        context.device.submit_command_buffer(command_encoder.finish())
        return targets

    def prepare_draw_data(
        self, context: RenderContext, inputs: Mapping[str, Any], spy
    ) -> RasterDrawData:
        raise NotImplementedError

    def bind_shader_data(
        self, shader_object, context: RenderContext, inputs: Mapping[str, Any], spy
    ) -> None:
        pass

    def draw(self, encoder, draw_data: RasterDrawData, spy) -> None:
        encoder.draw_indexed({"vertex_count": draw_data.index_count})

    def _ensure_targets(self, context: RenderContext, spy) -> dict[str, Any]:
        usage = (
            spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access
            | spy.TextureUsage.render_target
        )
        return {
            target.name: create_texture(
                context,
                spy,
                f"{self.name}.{target.name}",
                fmt_name=target.format,
                usage=usage,
            )
            for target in self.pipeline_desc.targets
        }

    def _ensure_pipeline(self, context: RenderContext, spy) -> None:
        if self._pipeline is not None:
            return
        self._input_layout = context.device.create_input_layout(
            input_elements=[item.to_rhi(spy) for item in self.pipeline_desc.vertex_elements],
            vertex_streams=[item.to_rhi() for item in self.pipeline_desc.vertex_streams],
        )
        shader_path = resolve_shader_path(Path(self.pipeline_desc.shader), context.shader_paths)
        program = context.device.load_program(
            slang_source_path(shader_path),
            [self.pipeline_desc.vertex_entry, self.pipeline_desc.fragment_entry],
        )
        depth_stencil = (
            self.pipeline_desc.depth_stencil.to_pipeline(spy)
            if self.pipeline_desc.depth_stencil is not None
            else None
        )
        self._pipeline = context.device.create_render_pipeline(
            program=program,
            input_layout=self._input_layout,
            primitive_topology=enum_value(spy.PrimitiveTopology, self.pipeline_desc.primitive_topology),
            targets=[{"format": getattr(spy.Format, target.format)} for target in self.pipeline_desc.targets],
            depth_stencil=depth_stencil,
            rasterizer=self.pipeline_desc.rasterizer.to_rhi(spy),
        )

    def _ensure_depth_texture(self, context: RenderContext, spy) -> None:
        desc = self.pipeline_desc.depth_stencil
        if desc is None:
            return
        key = (int(context.width), int(context.height), desc.format)
        if self._depth_texture is not None and self._depth_cache_key == key:
            return
        self._depth_texture = context.device.create_texture(
            format=getattr(spy.Format, desc.format),
            width=context.width,
            height=context.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.depth_stencil,
            label=f"{self.name}.depth",
        )
        self._depth_cache_key = key

    def _render_pass_desc(self, spy, targets: Mapping[str, Any]) -> dict[str, Any]:
        render_pass_desc: dict[str, Any] = {
            "color_attachments": [
                target.to_attachment(spy, targets[target.name])
                for target in self.pipeline_desc.targets
            ]
        }
        if self._depth_texture is not None and self.pipeline_desc.depth_stencil is not None:
            render_pass_desc["depth_stencil_attachment"] = (
                self.pipeline_desc.depth_stencil.to_attachment(spy, self._depth_texture)
            )
        return render_pass_desc
