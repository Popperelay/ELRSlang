"""CPU-side scene baking helpers shared by graphics and ray tracing passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scene import Material, Scene, matrix_to_numpy, normalize, transform_points

Color4 = tuple[float, float, float, float]

FALCOR_DIFFUSE_MODES = frozenset({"falcor_diffuse", "diffuse_opacity", "albedo"})


@dataclass(frozen=True)
class RasterDrawList:
    instances: tuple[str, ...] = ()
    meshes: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Any | None) -> "RasterDrawList":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        return cls(
            instances=tuple(str(item) for item in config.get("instances", ())),
            meshes=tuple(str(item) for item in config.get("meshes", ())),
            materials=tuple(str(item) for item in config.get("materials", ())),
        )

    def includes(self, instance_name: str, mesh_name: str, material_name: str) -> bool:
        if self.instances and instance_name not in self.instances:
            return False
        if self.meshes and mesh_name not in self.meshes:
            return False
        if self.materials and material_name not in self.materials:
            return False
        return True


@dataclass(frozen=True)
class RasterBakeSettings:
    width: int
    height: int
    fallback_color: Color4
    mode: str = "lit"
    draw_list: RasterDrawList = field(default_factory=RasterDrawList)

    @property
    def aspect(self) -> float:
        return float(self.width) / max(float(self.height), 1.0)


def raster_scene_cache_key(scene: Scene, settings: RasterBakeSettings) -> tuple[Any, ...]:
    scene.ensure_defaults()
    instances = tuple(
        (instance.name, instance.mesh_index, instance.transform) for instance in scene.instances
    )
    materials = tuple(
        (material.name, material.base_color, material.emissive_color, material.emissive_factor)
        for material in scene.materials
    )
    cameras = tuple(
        (camera.position, camera.target, camera.up, camera.vfov_degrees)
        for camera in scene.cameras
    )
    return (
        scene.source_path,
        tuple(
            (mesh.name, mesh.vertex_count, len(mesh.indices), mesh.material_index)
            for mesh in scene.meshes
        ),
        instances,
        materials,
        cameras,
        scene.selected_camera,
        tuple(float(c) for c in settings.fallback_color),
        settings.mode,
        int(settings.width),
        int(settings.height),
        settings.draw_list.instances,
        settings.draw_list.meshes,
        settings.draw_list.materials,
    )


def build_raster_scene_buffers(
    scene: Scene, settings: RasterBakeSettings
) -> tuple[np.ndarray, np.ndarray]:
    scene.ensure_defaults()
    view_projection = (
        scene.active_camera.projection_matrix(settings.aspect) @ scene.active_camera.view_matrix()
    )
    triangles: list[np.ndarray] = []
    light = scene.lights[0] if scene.lights else None

    for instance in scene.instances:
        if not (0 <= instance.mesh_index < len(scene.meshes)):
            continue
        mesh = scene.meshes[instance.mesh_index]
        material_name = ""
        if 0 <= mesh.material_index < len(scene.materials):
            material_name = scene.materials[mesh.material_index].name
        if not settings.draw_list.includes(instance.name, mesh.name, material_name):
            continue
        positions = mesh.position_array()
        normals = mesh.normal_array()
        indices = mesh.index_array().astype(np.uint32, copy=False).reshape(-1)
        model = matrix_to_numpy(instance.transform)
        world_positions = transform_points(model, positions)
        normal_matrix = np.linalg.pinv(model[:3, :3]).T
        world_normals = np.asarray(normals @ normal_matrix.T, dtype=np.float32)
        for i in range(0, len(indices) - 2, 3):
            tri_indices = indices[i : i + 3]
            tri_world = world_positions[tri_indices]
            tri_normals = world_normals[tri_indices]
            ndc = _project_triangle_to_ndc(tri_world, view_projection)
            if ndc is None or _is_outside_view(ndc):
                continue
            color = shade_triangle(
                scene,
                mesh.material_index,
                tri_world,
                tri_normals,
                light,
                settings,
            )
            tri_vertices = np.zeros((3, 7), dtype=np.float32)
            tri_vertices[:, 0:3] = ndc.astype(np.float32)
            tri_vertices[:, 3:7] = np.asarray(color, dtype=np.float32)
            triangles.append(tri_vertices)

    if not triangles:
        return fallback_raster_triangle(settings.fallback_color)

    vertices = np.concatenate(triangles, axis=0)
    indices = np.arange(vertices.shape[0], dtype=np.uint32)
    return vertices.reshape(-1), indices


def shade_triangle(
    scene: Scene,
    material_index: int,
    positions: np.ndarray,
    normals: np.ndarray,
    light: Any,
    settings: RasterBakeSettings,
) -> Color4:
    material = (
        scene.materials[material_index]
        if 0 <= material_index < len(scene.materials)
        else Material(base_color=settings.fallback_color)
    )
    base = np.asarray(material.base_color, dtype=np.float32)
    if settings.mode in FALCOR_DIFFUSE_MODES:
        return (float(base[0]), float(base[1]), float(base[2]), 1.0)

    normal = normalize(np.mean(normals, axis=0), (0.0, 1.0, 0.0))
    lambert = 0.8
    light_color = np.ones(3, dtype=np.float32)
    if light is not None:
        if light.kind in {"directional", "distant"}:
            light_dir = -np.asarray(light.direction, dtype=np.float32)
        else:
            light_dir = np.asarray(light.position, dtype=np.float32) - np.mean(positions, axis=0)
        light_dir = normalize(light_dir, (0.0, 1.0, 0.0))
        light_color_tuple, intensity = light.color_intensity()
        light_color = np.asarray(light_color_tuple, dtype=np.float32) * min(float(intensity), 10.0)
        lambert = max(float(np.dot(normal, light_dir)), 0.0)
    shaded = base[:3] * (0.22 + 0.78 * lambert) * np.clip(light_color, 0.0, 4.0)
    emissive = np.asarray(material.emissive_color, dtype=np.float32) * float(
        material.emissive_factor
    )
    shaded = np.clip(shaded + emissive, 0.0, 20.0)
    return (float(shaded[0]), float(shaded[1]), float(shaded[2]), float(base[3]))


def build_world_scene_buffers(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    scene.ensure_defaults()
    vertices: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    vertex_offset = 0
    for instance in scene.instances:
        if not (0 <= instance.mesh_index < len(scene.meshes)):
            continue
        mesh = scene.meshes[instance.mesh_index]
        world_positions = transform_points(instance.transform, mesh.position_array()).astype(
            np.float32
        )
        mesh_indices = mesh.index_array().astype(np.uint32, copy=False).reshape(-1)
        if mesh_indices.size == 0:
            mesh_indices = np.arange(world_positions.shape[0], dtype=np.uint32)
        vertices.append(world_positions)
        indices.append(mesh_indices + vertex_offset)
        vertex_offset += world_positions.shape[0]
    if not vertices:
        return build_world_scene_buffers(Scene.default())
    return (
        np.concatenate(vertices, axis=0).astype(np.float32),
        np.concatenate(indices, axis=0).astype(np.uint32),
    )


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


def fallback_raster_triangle(color: Color4) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-0.8, -0.7, 0.5, *color],
            [0.8, -0.7, 0.5, *color],
            [0.0, 0.8, 0.5, *color],
        ],
        dtype=np.float32,
    )
    return vertices.reshape(-1), np.arange(3, dtype=np.uint32)


def _project_triangle_to_ndc(
    tri_world: np.ndarray, view_projection: np.ndarray
) -> np.ndarray | None:
    homogeneous = np.concatenate([tri_world, np.ones((3, 1), dtype=np.float32)], axis=1)
    clip = homogeneous @ view_projection.T
    if np.all(clip[:, 3] <= 1e-5):
        return None
    safe_w = np.where(np.abs(clip[:, 3:4]) > 1e-5, clip[:, 3:4], 1.0)
    return clip[:, :3] / safe_w


def _is_outside_view(ndc: np.ndarray) -> bool:
    return bool(
        np.all(ndc[:, 0] < -1.5)
        or np.all(ndc[:, 0] > 1.5)
        or np.all(ndc[:, 1] < -1.5)
        or np.all(ndc[:, 1] > 1.5)
    )
