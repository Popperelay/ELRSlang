"""Built-in render passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .device import has_feature, import_slangpy
from .paths import SHADER_DIR
from .render_graph import PassReflection, RenderContext, RenderPass
from .resources import TextureDesc
from .scene import Scene
from .scene_buffers import (
    RasterBakeSettings,
    build_raster_scene_buffers,
    build_world_scene_buffers,
    dxr_smoke_quad,
    fit_mesh_to_screen,
    raster_scene_cache_key,
)


class FeatureUnavailable(RuntimeError):
    pass


PASS_REGISTRY: dict[str, Any] = {}


def register_pass_type(pass_type: str, factory: Any) -> None:
    PASS_REGISTRY[pass_type] = factory


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
    """Hardware raster pass that draws all scene instances with simple material colors."""

    def __init__(
        self,
        name: str = "HardwareRasterForward",
        shader: str | Path = "raster_forward.slang",
        output: str = "color",
        color: tuple[float, float, float, float] = (0.1, 0.45, 0.9, 1.0),
        mode: str = "lit",
    ) -> None:
        super().__init__(name)
        self.shader = Path(shader)
        self.output = output
        self.color = color
        self.mode = mode
        self._pipeline = None
        self._input_layout = None
        self._vertex_buffer = None
        self._index_buffer = None
        self._depth_texture = None
        self._depth_size: tuple[int, int] | None = None
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
        self._ensure_depth_texture(context, spy)
        scene_view = inputs["scene"]
        self._ensure_scene_buffers(context, spy, scene_view.scene)

        command_encoder = context.device.create_command_encoder()
        command_encoder.clear_texture_depth_stencil(self._depth_texture, depth_value=1.0)
        render_pass_desc = {
            "color_attachments": [
                {
                    "view": target.create_view({}),
                    "clear_value": [0.0, 0.0, 0.0, 1.0],
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                }
            ],
            "depth_stencil_attachment": {
                "view": self._depth_texture.create_view({}),
                "depth_load_op": spy.LoadOp.load,
                "depth_store_op": spy.StoreOp.store,
            },
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
                },
                {
                    "semantic_name": "COLOR",
                    "semantic_index": 0,
                    "format": spy.Format.rgba32_float,
                    "offset": 12,
                }
            ],
            vertex_streams=[{"stride": 28}],
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
            depth_stencil={
                "format": spy.Format.d32_float,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
            rasterizer={"cull_mode": spy.CullMode.back},
        )

    def _ensure_depth_texture(self, context: RenderContext, spy) -> None:
        size = (int(context.width), int(context.height))
        if self._depth_texture is not None and self._depth_size == size:
            return
        self._depth_texture = context.device.create_texture(
            format=spy.Format.d32_float,
            width=context.width,
            height=context.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.depth_stencil,
            label=f"{self.name}.depth",
        )
        self._depth_size = size

    def _ensure_scene_buffers(self, context: RenderContext, spy, scene: Scene) -> None:
        settings = RasterBakeSettings(context.width, context.height, self.color, self.mode)
        cache_key = raster_scene_cache_key(scene, settings)
        if self._mesh_cache_key == cache_key and self._vertex_buffer is not None:
            return

        vertices, indices = build_raster_scene_buffers(scene, settings)

        self._vertex_buffer = context.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
            label=f"{self.name}.scene.vertex_buffer",
            data=vertices,
        )
        self._index_buffer = context.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.index_buffer,
            label=f"{self.name}.scene.index_buffer",
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
        vertices, indices = build_world_scene_buffers(scene)

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
            camera = (context.scene or Scene.default()).active_camera
            forward, right, up = camera.basis()
            cursor.rt_camera_position = spy.float3(*camera.position)
            cursor.rt_camera_forward = spy.float3(*[float(v) for v in forward])
            cursor.rt_camera_right = spy.float3(*[float(v) for v in right])
            cursor.rt_camera_up = spy.float3(*[float(v) for v in up])
            cursor.rt_camera_vfov_degrees = float(camera.vfov_degrees)
            cursor.rt_resolution = spy.float2(float(context.width), float(context.height))
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
    if pass_type in PASS_REGISTRY:
        return PASS_REGISTRY[pass_type](config)
    raise ValueError(f"Unknown render pass type `{pass_type}`.")


def _scene_upload_from_config(config: Mapping[str, Any]) -> RenderPass:
    return SceneUploadPass(config["name"])


def _slang_function_from_config(config: Mapping[str, Any]) -> RenderPass:
    name = config["name"]
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


def _tone_map_from_config(config: Mapping[str, Any]) -> RenderPass:
    return ToneMapPass(name=config["name"], output=config.get("output", "color"))


def _present_from_config(config: Mapping[str, Any]) -> RenderPass:
    return PresentPass(name=config["name"])


def _hardware_raster_from_config(config: Mapping[str, Any]) -> RenderPass:
    return HardwareRasterPass(
        name=config["name"],
        shader=config.get("shader", "raster_forward.slang"),
        output=config.get("output", "color"),
        color=tuple(config.get("color", (0.1, 0.45, 0.9, 1.0))),
        mode=config.get("mode", "lit"),
    )


def _build_as_from_config(config: Mapping[str, Any]) -> RenderPass:
    return BuildAccelerationStructurePass(name=config["name"], output=config.get("output", "tlas"))


def _hardware_dxr_from_config(config: Mapping[str, Any]) -> RenderPass:
    return HardwareDXRPass(
        name=config["name"],
        shader=config.get("shader", "dxr_pathtrace.slang"),
        output=config.get("output", "color"),
    )


register_pass_type("SceneUploadPass", _scene_upload_from_config)
register_pass_type("SlangFunctionPass", _slang_function_from_config)
register_pass_type("ToneMapPass", _tone_map_from_config)
register_pass_type("PresentPass", _present_from_config)
register_pass_type("HardwareRasterPass", _hardware_raster_from_config)
register_pass_type("BuildAccelerationStructurePass", _build_as_from_config)
register_pass_type("HardwareDXRPass", _hardware_dxr_from_config)
