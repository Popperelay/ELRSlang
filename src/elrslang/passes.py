"""Built-in render passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .device import has_feature, import_slangpy
from .paths import SHADER_DIR
from .render_graph import PassReflection, RenderContext, RenderPass
from .resources import TextureDesc
from .scene import Scene


class FeatureUnavailable(RuntimeError):
    pass


class SceneUploadPass(RenderPass):
    def reflect(self) -> PassReflection:
        return PassReflection.from_iterables(outputs=("scene",))

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        scene = context.scene or Scene.default()
        return {"scene": scene.to_view()}


class SlangFunctionPass(RenderPass):
    """Call a Slang function directly through SlangPy.

    Bindings are intentionally declarative so a graph JSON can express most
    full-screen and compute work without a custom Python subclass.
    """

    def __init__(
        self,
        name: str,
        module_path: str | Path,
        function_name: str,
        bindings: Mapping[str, Any],
        result: str,
        inputs: tuple[str, ...] = (),
        outputs: tuple[str, ...] | None = None,
        output_format: str = "rgba32_float",
    ) -> None:
        super().__init__(name)
        self.module_path = Path(module_path)
        self.function_name = function_name
        self.bindings = dict(bindings)
        self.result = result
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs or (result,))
        self.output_format = output_format
        self._module = None

    def reflect(self) -> PassReflection:
        return PassReflection.from_iterables(inputs=self.inputs, outputs=self.outputs)

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        spy = import_slangpy()
        if context.device is None:
            raise RuntimeError(f"Pass `{self.name}` requires a SlangPy device.")
        module = self._load_module(spy, context)
        function = getattr(module, self.function_name)
        kwargs = {
            name: self._resolve_binding(spec, context, inputs, spy)
            for name, spec in self.bindings.items()
        }
        result_resource = self._ensure_result_texture(context, spy)
        function(**kwargs, _result=result_resource)
        return {self.result: result_resource}

    def _load_module(self, spy, context: RenderContext):
        if self._module is not None:
            return self._module
        shader_path = resolve_shader_path(self.module_path, context.shader_paths)
        self._module = spy.Module.load_from_file(context.device, slang_source_path(shader_path))
        return self._module

    def _ensure_result_texture(self, context: RenderContext, spy):
        existing = context.resources.get(f"{self.name}.{self.result}")
        if existing is not None:
            return existing
        fmt = getattr(spy.Format, self.output_format)
        texture = context.device.create_texture(
            width=context.width,
            height=context.height,
            format=fmt,
            mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"{self.name}.{self.result}",
        )
        context.resources.set(
            f"{self.name}.{self.result}",
            texture,
            TextureDesc(context.width, context.height, self.output_format, f"{self.name}.{self.result}"),
        )
        return texture

    def _resolve_binding(self, spec: Any, context: RenderContext, inputs: Mapping[str, Any], spy):
        if isinstance(spec, str):
            if spec.startswith("$input."):
                return to_slangpy_value(inputs[spec.removeprefix("$input.")], spy)
            if spec == "$frame":
                return slang_frame_constants(context, spy)
            if spec == "$scene":
                scene = context.scene or Scene.default()
                return to_slangpy_value(scene.to_view(), spy)
            if spec.startswith("$resource."):
                return context.resources.require(spec.removeprefix("$resource."))
            return spec
        if isinstance(spec, list):
            return [self._resolve_binding(item, context, inputs, spy) for item in spec]
        if isinstance(spec, dict):
            if "input" in spec:
                return to_slangpy_value(inputs[spec["input"]], spy)
            if "resource" in spec:
                return context.resources.require(spec["resource"])
            if spec.get("special") == "frame":
                return slang_frame_constants(context, spy)
            if spec.get("special") == "sampler":
                return default_sampler(context)
            if "generator" in spec:
                return resolve_generator(spec, spy)
            return {key: self._resolve_binding(value, context, inputs, spy) for key, value in spec.items()}
        return spec


class ToneMapPass(SlangFunctionPass):
    def __init__(self, name: str = "ToneMap", output: str = "color") -> None:
        super().__init__(
            name=name,
            module_path="tonemap.slang",
            function_name="tonemap",
            inputs=("input",),
            outputs=(output,),
            bindings={
                "pixel": {"generator": "call_id"},
                "hdr": {"input": "input"},
                "samplerState": {"special": "sampler"},
                "frame": {"special": "frame"},
            },
            result=output,
        )


class PresentPass(RenderPass):
    def __init__(self, name: str = "Present") -> None:
        super().__init__(name)

    def reflect(self) -> PassReflection:
        return PassReflection.from_iterables(inputs=("input",), outputs=("presented",))

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        image = inputs["input"]
        context.output = image
        if context.app is not None:
            context.app.present(image)
        return {"presented": image}


class PipelinePass(RenderPass):
    pass


class HardwareRasterPass(PipelinePass):
    """Minimal hardware raster pass for the first mesh in the scene."""

    def __init__(
        self,
        name: str = "HardwareRasterForward",
        shader: str | Path = "raster_forward.slang",
        output: str = "color",
        color: tuple[float, float, float, float] = (0.1, 0.45, 0.9, 1.0),
    ) -> None:
        super().__init__(name)
        self.shader = Path(shader)
        self.output = output
        self.color = color
        self._pipeline = None
        self._input_layout = None
        self._vertex_buffer = None
        self._index_buffer = None
        self._index_count = 0
        self._mesh_cache_key: tuple[Any, ...] | None = None

    def reflect(self) -> PassReflection:
        return PassReflection.from_iterables(inputs=("scene",), outputs=(self.output,))

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        spy = import_slangpy()
        if context.device is None:
            raise RuntimeError(f"Pass `{self.name}` requires a SlangPy device.")
        if not has_feature(context.device, "rasterization"):
            raise FeatureUnavailable("Current device does not support hardware rasterization.")

        target = create_texture(
            context,
            spy,
            f"{self.name}.{self.output}",
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access
            | spy.TextureUsage.render_target,
        )
        self._ensure_pipeline(context, spy)
        scene_view = inputs["scene"]
        self._ensure_scene_mesh(context, spy, scene_view.scene)

        command_encoder = context.device.create_command_encoder()
        render_pass_desc = {
            "color_attachments": [
                {
                    "view": target.create_view({}),
                    "clear_value": [0.0, 0.0, 0.0, 1.0],
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                }
            ]
        }
        with command_encoder.begin_render_pass(render_pass_desc) as encoder:
            encoder.set_render_state(
                {
                    "vertex_buffers": [self._vertex_buffer],
                    "index_buffer": self._index_buffer,
                    "index_format": spy.IndexFormat.uint32,
                    "viewports": [spy.Viewport.from_size(context.width, context.height)],
                    "scissor_rects": [spy.ScissorRect.from_size(context.width, context.height)],
                }
            )
            shader_object = encoder.bind_pipeline(self._pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.vert_offset = spy.float2(0.0, 0.0)
            cursor.vert_scale = spy.float2(1.0, 1.0)
            cursor.vert_z = 0.0
            cursor.frag_color = spy.float4(*self.color)
            encoder.draw_indexed({"vertex_count": self._index_count})
        context.device.submit_command_buffer(command_encoder.finish())
        return {self.output: target}

    def _ensure_pipeline(self, context: RenderContext, spy) -> None:
        if self._pipeline is not None:
            return
        self._input_layout = context.device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                    "offset": 0,
                }
            ],
            vertex_streams=[{"stride": 12}],
        )
        shader_path = resolve_shader_path(self.shader, context.shader_paths)
        program = context.device.load_program(
            slang_source_path(shader_path), ["vertex_main", "fragment_main"]
        )
        self._pipeline = context.device.create_render_pipeline(
            program=program,
            input_layout=self._input_layout,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[{"format": spy.Format.rgba32_float}],
            rasterizer={"cull_mode": spy.CullMode.none},
        )

    def _ensure_scene_mesh(self, context: RenderContext, spy, scene: Scene) -> None:
        scene.ensure_defaults()
        mesh = scene.meshes[0] if scene.meshes else Scene.default().meshes[0]
        cache_key = (
            scene.source_path,
            mesh.name,
            mesh.vertex_count,
            len(mesh.indices),
            context.width,
            context.height,
        )
        if self._mesh_cache_key == cache_key and self._vertex_buffer is not None:
            return

        vertices = fit_mesh_to_screen(mesh.position_array())
        indices = mesh.index_array().astype(np.uint32, copy=False).reshape(-1)
        if indices.size == 0:
            indices = np.arange(vertices.shape[0], dtype=np.uint32)

        self._vertex_buffer = context.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
            label=f"{self.name}.{mesh.name}.vertex_buffer",
            data=vertices,
        )
        self._index_buffer = context.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.index_buffer,
            label=f"{self.name}.{mesh.name}.index_buffer",
            data=indices,
        )
        self._index_count = int(indices.size)
        self._mesh_cache_key = cache_key


class BuildAccelerationStructurePass(PipelinePass):
    def __init__(self, name: str = "BuildAS", output: str = "tlas") -> None:
        super().__init__(name)
        self.output = output
        self._keepalive: list[Any] = []

    def reflect(self) -> PassReflection:
        return PassReflection.from_iterables(inputs=("scene",), outputs=(self.output,))

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        spy = import_slangpy()
        if context.device is None:
            raise RuntimeError(f"Pass `{self.name}` requires a SlangPy device.")
        if not has_feature(context.device, "acceleration_structure"):
            raise FeatureUnavailable("Current device does not support acceleration structures.")

        scene_view = inputs["scene"]
        scene = scene_view.scene
        if scene.source_path is None:
            vertices, indices = dxr_smoke_quad()
        else:
            mesh = scene.meshes[0] if scene.meshes else Scene.default().meshes[0]
            vertices = mesh.position_array().astype(np.float32)
            indices = mesh.index_array().astype(np.uint32)

        vertex_buffer = context.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.acceleration_structure_build_input,
            label=f"{self.name}.vertex_buffer",
            data=vertices,
        )
        index_buffer = context.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.acceleration_structure_build_input,
            label=f"{self.name}.index_buffer",
            data=indices,
        )
        self._keepalive = [vertex_buffer, index_buffer]
        triangle_input = spy.AccelerationStructureBuildInputTriangles()
        triangle_input.flags = spy.AccelerationStructureGeometryFlags.opaque
        triangle_input.vertex_buffers = [spy.BufferOffsetPair(vertex_buffer)]
        triangle_input.vertex_format = spy.Format.rgb32_float
        triangle_input.vertex_count = vertices.size // 3
        triangle_input.vertex_stride = vertices.itemsize * 3
        triangle_input.index_buffer = index_buffer
        triangle_input.index_format = spy.IndexFormat.uint32
        triangle_input.index_count = indices.size
        blas = build_acceleration_structure(
            context.device,
            spy,
            [triangle_input],
            f"{self.name}.blas",
            self._keepalive,
        )
        self._keepalive.append(blas)
        instance_list = context.device.create_acceleration_structure_instance_list(1)
        transform = spy.float3x4.identity()
        if scene.source_path is None:
            scale = spy.float3(max(float(context.width) * 0.5 - 0.9, 1.0), max(float(context.height) * 0.5 - 0.9, 1.0), 1.0)
            transform = spy.float3x4(
                spy.math.mul(
                    spy.math.matrix_from_translation(spy.float3(-0.05, -0.05, 1.0)),
                    spy.math.matrix_from_scaling(scale),
                )
            )
        instance_list.write(
            0,
            {
                "transform": transform,
                "instance_id": 0,
                "instance_mask": 0xFF,
                "instance_contribution_to_hit_group_index": 0,
                "flags": spy.AccelerationStructureInstanceFlags.none,
                "acceleration_structure": blas.handle,
            },
        )
        tlas = build_acceleration_structure(
            context.device,
            spy,
            [instance_list.build_input_instances()],
            f"{self.name}.tlas",
            self._keepalive,
        )
        self._keepalive.extend([instance_list, tlas])
        return {self.output: tlas}


class HardwareDXRPass(PipelinePass):
    def __init__(
        self,
        name: str = "HardwareDXRTrace",
        shader: str | Path = "dxr_pathtrace.slang",
        output: str = "color",
    ) -> None:
        super().__init__(name)
        self.shader = Path(shader)
        self.output = output
        self._program = None
        self._pipeline = None
        self._shader_table = None

    def reflect(self) -> PassReflection:
        return PassReflection.from_iterables(inputs=("tlas",), outputs=(self.output,))

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        spy = import_slangpy()
        if context.device is None:
            raise RuntimeError(f"Pass `{self.name}` requires a SlangPy device.")
        if not has_feature(context.device, "ray_tracing"):
            raise FeatureUnavailable("Current device does not support hardware ray tracing.")

        target = create_texture(
            context,
            spy,
            f"{self.name}.{self.output}",
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access
            | spy.TextureUsage.render_target,
        )
        self._ensure_ray_pipeline(context, spy)
        command_encoder = context.device.create_command_encoder()
        with command_encoder.begin_ray_tracing_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self._pipeline, self._shader_table)
            cursor = spy.ShaderCursor(shader_object)
            cursor.rt_tlas = inputs["tlas"]
            cursor.rt_render_texture = target
            pass_encoder.dispatch_rays(0, [context.width, context.height, 1])
        context.device.submit_command_buffer(command_encoder.finish())
        return {self.output: target}

    def _ensure_ray_pipeline(self, context: RenderContext, spy) -> None:
        if self._pipeline is not None:
            return
        shader_path = resolve_shader_path(self.shader, context.shader_paths)
        self._program = context.device.load_program(
            slang_source_path(shader_path), ["rt_ray_gen", "rt_miss", "rt_closest_hit"]
        )
        self._pipeline = context.device.create_ray_tracing_pipeline(
            program=self._program,
            hit_groups=[
                spy.HitGroupDesc(
                    hit_group_name="hit_group",
                    closest_hit_entry_point="rt_closest_hit",
                )
            ],
            max_recursion=1,
            max_ray_payload_size=16,
        )
        self._shader_table = context.device.create_shader_table(
            program=self._program,
            ray_gen_entry_points=["rt_ray_gen"],
            miss_entry_points=["rt_miss"],
            hit_group_names=["hit_group"],
        )


def create_texture(context: RenderContext, spy, key: str, fmt_name: str = "rgba32_float", usage=None):
    existing = context.resources.get(key)
    if existing is not None:
        return existing
    texture = context.device.create_texture(
        width=context.width,
        height=context.height,
        format=getattr(spy.Format, fmt_name),
        mip_count=1,
        usage=usage or (spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access),
        label=key,
    )
    context.resources.set(key, texture, TextureDesc(context.width, context.height, fmt_name, key))
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


def dxr_smoke_quad() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
    return vertices, indices


def fit_mesh_to_screen(positions: np.ndarray) -> np.ndarray:
    source = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if source.size == 0:
        return np.asarray(
            [(-0.8, -0.7, 0.0), (0.8, -0.7, 0.0), (0.0, 0.8, 0.0)],
            dtype=np.float32,
        )

    bounds_min = source.min(axis=0)
    bounds_max = source.max(axis=0)
    extents = bounds_max - bounds_min
    axes = (0, 1)
    if extents[1] < max(extents[0], extents[2]) * 0.05 and extents[2] > 0:
        axes = (0, 2)

    projected = source[:, axes]
    projected_min = projected.min(axis=0)
    projected_max = projected.max(axis=0)
    center = (projected_min + projected_max) * 0.5
    scale = float(np.max(projected_max - projected_min))
    if scale <= 1e-8:
        scale = 1.0

    normalized = (projected - center) * (1.7 / scale)
    vertices = np.zeros((source.shape[0], 3), dtype=np.float32)
    vertices[:, 0:2] = normalized
    vertices[:, 2] = 0.0
    return vertices.reshape(-1)


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


def resolve_generator(spec: Mapping[str, Any], spy):
    generator = spec["generator"]
    if generator == "call_id":
        return spy.call_id()
    if generator == "thread_id":
        return spy.thread_id()
    if generator == "grid":
        return spy.grid(shape=tuple(spec["shape"]), stride=tuple(spec.get("stride", (1,) * len(spec["shape"]))))
    raise ValueError(f"Unsupported SlangPy generator `{generator}`.")


def default_sampler(context: RenderContext):
    sampler = context.resources.get("_default_sampler")
    if sampler is None:
        sampler = context.device.create_sampler()
        context.resources.set("_default_sampler", sampler)
    return sampler


def slang_frame_constants(context: RenderContext, spy):
    return {
        "_type": "FrameConstants",
        "frameIndex": int(context.frame.frame_index),
        "resolution": spy.float2(float(context.width), float(context.height)),
        "timeSeconds": float(context.frame.time_seconds),
    }


def to_slangpy_value(value: Any, spy):
    if hasattr(value, "get_this"):
        return to_slangpy_value(value.get_this(), spy)
    if isinstance(value, dict):
        return {key: to_slangpy_value(item, spy) for key, item in value.items()}
    if isinstance(value, tuple) and all(isinstance(item, (int, float)) for item in value):
        if len(value) == 2:
            return spy.float2(*value)
        if len(value) == 3:
            return spy.float3(*value)
        if len(value) == 4:
            return spy.float4(*value)
    if isinstance(value, list):
        return [to_slangpy_value(item, spy) for item in value]
    return value


def pass_from_config(config: Mapping[str, Any]) -> RenderPass:
    pass_type = config["type"]
    name = config["name"]
    if pass_type == "SceneUploadPass":
        return SceneUploadPass(name)
    if pass_type == "SlangFunctionPass":
        required = {"module", "function", "bindings", "result"}
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"SlangFunctionPass `{name}` is missing fields: {', '.join(missing)}")
        return SlangFunctionPass(
            name=name,
            module_path=config["module"],
            function_name=config["function"],
            bindings=config["bindings"],
            result=config["result"],
            inputs=tuple(config.get("inputs", ())),
            outputs=tuple(config.get("outputs", (config["result"],))),
            output_format=config.get("output_format", "rgba32_float"),
        )
    if pass_type == "ToneMapPass":
        return ToneMapPass(name=name, output=config.get("output", "color"))
    if pass_type == "PresentPass":
        return PresentPass(name=name)
    if pass_type == "HardwareRasterPass":
        return HardwareRasterPass(
            name=name,
            shader=config.get("shader", "raster_forward.slang"),
            output=config.get("output", "color"),
            color=tuple(config.get("color", (0.1, 0.45, 0.9, 1.0))),
        )
    if pass_type == "BuildAccelerationStructurePass":
        return BuildAccelerationStructurePass(name=name, output=config.get("output", "tlas"))
    if pass_type == "HardwareDXRPass":
        return HardwareDXRPass(
            name=name,
            shader=config.get("shader", "dxr_pathtrace.slang"),
            output=config.get("output", "color"),
        )
    raise ValueError(f"Unknown render pass type `{pass_type}`.")
