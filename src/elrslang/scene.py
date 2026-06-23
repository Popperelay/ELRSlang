"""Scene schema, importers, and a small Falcor `.pyscene` compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
import math
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

import numpy as np


Matrix4 = tuple[tuple[float, float, float, float], ...]


IDENTITY_4X4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _Vec(tuple):
    def _binary(self, other: Any, op: Any) -> "_Vec":
        if isinstance(other, (int, float)):
            values = [op(float(a), float(other)) for a in self]
        else:
            values = [op(float(a), float(b)) for a, b in zip(self, other)]
        return type(self)(*values)

    def __add__(self, other: Any) -> "_Vec":
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> "_Vec":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "_Vec":
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> "_Vec":
        if isinstance(other, (int, float)):
            values = [float(other) - float(a) for a in self]
        else:
            values = [float(a) - float(b) for a, b in zip(other, self)]
        return type(self)(*values)

    def __mul__(self, other: Any) -> "_Vec":
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> "_Vec":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "_Vec":
        return self._binary(other, lambda a, b: a / b)


class Vec2(_Vec):
    def __new__(cls, x: float = 0.0, y: float = 0.0):
        return tuple.__new__(cls, (float(x), float(y)))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]


class Vec3(_Vec):
    def __new__(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        return tuple.__new__(cls, (float(x), float(y), float(z)))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]


class Vec4(_Vec):
    def __new__(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 0.0):
        return tuple.__new__(cls, (float(x), float(y), float(z), float(w)))


def float2(x: float = 0.0, y: float | None = None) -> tuple[float, float]:
    if y is None:
        y = x
    return Vec2(float(x), float(y))


def float3(
    x: float = 0.0,
    y: float | None = None,
    z: float | None = None,
) -> tuple[float, float, float]:
    if y is None and z is None:
        y = z = x
    if y is None or z is None:
        raise ValueError("float3 requires either 1 or 3 values.")
    return Vec3(float(x), float(y), float(z))


def float4(
    x: float = 0.0,
    y: float | None = None,
    z: float | None = None,
    w: float | None = None,
) -> tuple[float, float, float, float]:
    if y is None and z is None and w is None:
        y = z = w = x
    if y is None or z is None or w is None:
        raise ValueError("float4 requires either 1 or 4 values.")
    return Vec4(float(x), float(y), float(z), float(w))


def _vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    values = list(value)
    if len(values) < 3:
        raise ValueError(f"Expected 3 values, got `{value}`.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _vec4(
    value: Any,
    default: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value), float(value))
    values = list(value)
    if len(values) == 3:
        values.append(1.0)
    if len(values) < 4:
        raise ValueError(f"Expected 4 values, got `{value}`.")
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _matrix(value: Any = None) -> Matrix4:
    if value is None:
        return IDENTITY_4X4
    arr = np.asarray(value, dtype=np.float32).reshape(4, 4)
    return tuple(tuple(float(c) for c in row) for row in arr)  # type: ignore[return-value]


def matrix_to_numpy(value: Matrix4) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(4, 4)


def matrix_from_numpy(value: np.ndarray) -> Matrix4:
    arr = np.asarray(value, dtype=np.float32).reshape(4, 4)
    return tuple(tuple(float(c) for c in row) for row in arr)  # type: ignore[return-value]


def normalize(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (value / length).astype(np.float32)


def transform_points(matrix: Matrix4 | np.ndarray, positions: np.ndarray) -> np.ndarray:
    mat = matrix_to_numpy(matrix) if isinstance(matrix, tuple) else np.asarray(matrix, dtype=np.float32)
    points = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return points.reshape(0, 3)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ mat.T
    w = transformed[:, 3:4]
    safe_w = np.where(np.abs(w) > 1e-8, w, 1.0)
    return (transformed[:, :3] / safe_w).astype(np.float32)


def compose_matrix(
    translation: Any = None,
    scaling: Any = None,
    rotation_euler: Any = None,
    rotation_euler_degrees: Any = None,
) -> Matrix4:
    t = _vec3(translation)
    s = _vec3(1.0 if scaling is None else scaling, (1.0, 1.0, 1.0))
    if rotation_euler_degrees is not None:
        rx, ry, rz = (math.radians(v) for v in _vec3(rotation_euler_degrees))
    else:
        rx, ry, rz = _vec3(rotation_euler)

    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    scale = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
    rot_x = np.asarray(
        [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    rot_y = np.asarray(
        [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    rot_z = np.asarray(
        [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    translate = np.eye(4, dtype=np.float32)
    translate[:3, 3] = np.asarray(t, dtype=np.float32)
    return matrix_from_numpy(translate @ rot_z @ rot_y @ rot_x @ scale)


def compose_trs(translation: Any, rotation_quat: Any, scale: Any) -> Matrix4:
    t = _vec3(translation)
    s = _vec3(scale, (1.0, 1.0, 1.0))
    qx, qy, qz, qw = _vec4(rotation_quat, (0.0, 0.0, 0.0, 1.0))
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rotation = np.asarray(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    scale_m = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
    translate = np.eye(4, dtype=np.float32)
    translate[:3, 3] = np.asarray(t, dtype=np.float32)
    return matrix_from_numpy(translate @ rotation @ scale_m)


@dataclass
class Camera:
    name: str = "DefaultCamera"
    position: tuple[float, float, float] = (0.0, 0.0, 4.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vfov_degrees: float = 45.0

    @property
    def focalLength(self) -> float:
        return 24.0 / (2.0 * math.tan(math.radians(self.vfov_degrees) * 0.5))

    @focalLength.setter
    def focalLength(self, value: float) -> None:
        focal = max(float(value), 1e-4)
        self.vfov_degrees = math.degrees(2.0 * math.atan(24.0 / (2.0 * focal)))

    def view_matrix(self) -> np.ndarray:
        eye = np.asarray(self.position, dtype=np.float32)
        target = np.asarray(self.target, dtype=np.float32)
        up = normalize(np.asarray(self.up, dtype=np.float32), (0.0, 1.0, 0.0))
        forward = normalize(target - eye, (0.0, 0.0, -1.0))
        right = normalize(np.cross(forward, up), (1.0, 0.0, 0.0))
        true_up = normalize(np.cross(right, forward), (0.0, 1.0, 0.0))
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[0, 3] = -float(np.dot(right, eye))
        view[1, 3] = -float(np.dot(true_up, eye))
        view[2, 3] = float(np.dot(forward, eye))
        return view

    def projection_matrix(self, aspect: float, near: float = 0.01, far: float = 1000.0) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.vfov_degrees) * 0.5)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / max(aspect, 1e-6)
        proj[1, 1] = f
        proj[2, 2] = far / (near - far)
        proj[2, 3] = (far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye = np.asarray(self.position, dtype=np.float32)
        forward = normalize(np.asarray(self.target, dtype=np.float32) - eye, (0.0, 0.0, -1.0))
        right = normalize(np.cross(forward, np.asarray(self.up, dtype=np.float32)), (1.0, 0.0, 0.0))
        up = normalize(np.cross(right, forward), (0.0, 1.0, 0.0))
        return forward, right, up

    def get_this(self) -> dict[str, Any]:
        return {
            "_type": "Camera",
            "position": self.position,
            "target": self.target,
            "up": self.up,
            "vfovDegrees": self.vfov_degrees,
        }


@dataclass
class Light:
    name: str = "KeyLight"
    kind: str = "directional"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float | tuple[float, float, float] = 1.0
    direction: tuple[float, float, float] = (-0.4, -1.0, -0.2)
    position: tuple[float, float, float] = (0.0, 3.0, 0.0)
    angle: float = 0.0

    def color_intensity(self) -> tuple[tuple[float, float, float], float]:
        if isinstance(self.intensity, tuple):
            rgb = tuple(float(c) for c in self.intensity)
            scalar = max(max(rgb), 1e-6)
            return (
                (
                    self.color[0] * rgb[0] / scalar,
                    self.color[1] * rgb[1] / scalar,
                    self.color[2] * rgb[2] / scalar,
                ),
                scalar,
            )
        return self.color, float(self.intensity)

    def get_this(self) -> dict[str, Any]:
        color, intensity = self.color_intensity()
        kind = 0 if self.kind in {"directional", "distant"} else 1
        return {
            "_type": "Light",
            "kind": kind,
            "color": color,
            "intensity": intensity,
            "direction": self.direction,
            "position": self.position,
        }


@dataclass
class Material:
    name: str = "DefaultMaterial"
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_factor: float = 0.0
    specular_transmission: float = 0.0
    index_of_refraction: float = 1.0
    double_sided: bool = False
    texture_paths: dict[str, str] = field(default_factory=dict)

    @property
    def baseColor(self) -> tuple[float, float, float, float]:
        return self.base_color

    @baseColor.setter
    def baseColor(self, value: Any) -> None:
        self.base_color = _vec4(value, self.base_color)

    @property
    def emissiveColor(self) -> tuple[float, float, float]:
        return self.emissive_color

    @emissiveColor.setter
    def emissiveColor(self, value: Any) -> None:
        self.emissive_color = _vec3(value, self.emissive_color)

    @property
    def emissiveFactor(self) -> float:
        return self.emissive_factor

    @emissiveFactor.setter
    def emissiveFactor(self, value: float) -> None:
        self.emissive_factor = float(value)

    @property
    def specularTransmission(self) -> float:
        return self.specular_transmission

    @specularTransmission.setter
    def specularTransmission(self, value: float) -> None:
        self.specular_transmission = float(value)

    @property
    def indexOfRefraction(self) -> float:
        return self.index_of_refraction

    @indexOfRefraction.setter
    def indexOfRefraction(self, value: float) -> None:
        self.index_of_refraction = float(value)

    @property
    def doubleSided(self) -> bool:
        return self.double_sided

    @doubleSided.setter
    def doubleSided(self, value: bool) -> None:
        self.double_sided = bool(value)

    def loadTexture(self, slot: str, path: str, useSrgb: bool = True) -> None:
        self.texture_paths[str(slot)] = str(path)

    def clearTexture(self, slot: str) -> None:
        self.texture_paths.pop(str(slot), None)

    def get_this(self) -> dict[str, Any]:
        return {
            "_type": "StandardMaterial",
            "baseColor": self.base_color,
            "roughness": self.roughness,
            "metallic": self.metallic,
        }


@dataclass
class StandardMaterial(Material):
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


PBRTDiffuseMaterial = Material


@dataclass
class EnvMap:
    path: str
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    intensity: float = 1.0

    @classmethod
    def createFromFile(cls, path: str) -> "EnvMap":
        return cls(path)


class MaterialTextureSlot:
    BaseColor = "BaseColor"
    Specular = "Specular"
    Emissive = "Emissive"
    Normal = "Normal"
    Transmission = "Transmission"
    Displacement = "Displacement"


class Animation:
    class Behavior:
        Cycle = "Cycle"
        Linear = "Linear"
        Oscillate = "Oscillate"
        Constant = "Constant"


@dataclass
class _AnimationSettings:
    preInfinityBehavior: str = Animation.Behavior.Constant
    postInfinityBehavior: str = Animation.Behavior.Constant


@dataclass
class Mesh:
    name: str
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    material_index: int = 0

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        return len(self.index_array()) // 3

    def copy(self, material_offset: int = 0) -> "Mesh":
        return Mesh(
            name=self.name,
            positions=list(self.positions),
            normals=list(self.normals),
            uvs=list(self.uvs),
            indices=list(self.indices),
            material_index=self.material_index + material_offset,
        )

    def position_array(self) -> np.ndarray:
        if self.positions:
            return np.asarray(self.positions, dtype=np.float32).reshape(-1, 3)
        return np.asarray(
            [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)], dtype=np.float32
        )

    def normal_array(self) -> np.ndarray:
        if self.normals and len(self.normals) == len(self.positions):
            return np.asarray(self.normals, dtype=np.float32).reshape(-1, 3)
        positions = self.position_array()
        normals = np.zeros_like(positions)
        indices = self.index_array()
        for i0, i1, i2 in indices.reshape(-1, 3):
            p0, p1, p2 = positions[i0], positions[i1], positions[i2]
            n = np.cross(p1 - p0, p2 - p0)
            normals[i0] += n
            normals[i1] += n
            normals[i2] += n
        for i, n in enumerate(normals):
            normals[i] = normalize(n, (0.0, 1.0, 0.0))
        return normals

    def uv_array(self) -> np.ndarray:
        if self.uvs and len(self.uvs) == len(self.positions):
            return np.asarray(self.uvs, dtype=np.float32).reshape(-1, 2)
        return np.zeros((self.position_array().shape[0], 2), dtype=np.float32)

    def index_array(self) -> np.ndarray:
        if self.indices:
            return np.asarray(self.indices, dtype=np.uint32).reshape(-1)
        return np.arange(self.position_array().shape[0], dtype=np.uint32)


@dataclass
class TriangleMesh:
    mesh: Mesh
    _base_dir: Path = Path.cwd()

    @classmethod
    def createQuad(cls, *args: Any, **kwargs: Any) -> "TriangleMesh":
        mesh = Mesh(
            name="Quad",
            positions=[
                (-0.5, 0.0, -0.5),
                (0.5, 0.0, -0.5),
                (0.5, 0.0, 0.5),
                (-0.5, 0.0, 0.5),
            ],
            normals=[(0.0, 1.0, 0.0)] * 4,
            uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            indices=[0, 2, 1, 0, 3, 2],
        )
        return cls(mesh)

    @classmethod
    def createCube(cls, *args: Any, **kwargs: Any) -> "TriangleMesh":
        positions: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        indices: list[int] = []
        faces = [
            ((0, 0, 1), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
            ((0, 0, -1), [(0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)]),
            ((1, 0, 0), [(0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]),
            ((-1, 0, 0), [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)]),
            ((0, 1, 0), [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)]),
            ((0, -1, 0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
        ]
        for normal, verts in faces:
            base = len(positions)
            positions.extend(verts)
            normals.extend([normal] * 4)
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        return cls(Mesh(name="Cube", positions=positions, normals=normals, indices=indices))

    @classmethod
    def createSphere(cls, *args: Any, **kwargs: Any) -> "TriangleMesh":
        radius = float(kwargs.pop("radius", 0.5))
        segments = int(kwargs.pop("segments", args[0] if len(args) >= 1 else 24))
        rings = int(kwargs.pop("rings", args[1] if len(args) >= 2 else 12))
        positions: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        indices: list[int] = []
        for ring in range(rings + 1):
            theta = math.pi * ring / rings
            y = math.cos(theta) * radius
            ring_radius = math.sin(theta) * radius
            for segment in range(segments):
                phi = 2.0 * math.pi * segment / segments
                pos = (math.cos(phi) * ring_radius, y, math.sin(phi) * ring_radius)
                positions.append(pos)
                normals.append(_vec3(normalize(np.asarray(pos, dtype=np.float32), (0.0, 1.0, 0.0))))
        for ring in range(rings):
            for segment in range(segments):
                a = ring * segments + segment
                b = ring * segments + (segment + 1) % segments
                c = (ring + 1) * segments + segment
                d = (ring + 1) * segments + (segment + 1) % segments
                indices.extend([a, c, b, b, c, d])
        return cls(Mesh(name="Sphere", positions=positions, normals=normals, indices=indices))

    @classmethod
    def createFromFile(cls, path: str, *args: Any, **kwargs: Any) -> "TriangleMesh":
        source = resolve_scene_relative_path(cls._base_dir, path)
        scene = SceneLoader().load(source)
        mesh = scene.meshes[0].copy() if scene.meshes else Scene.default().meshes[0].copy()
        mesh.name = source.stem
        return cls(mesh)


@dataclass
class MeshInstance:
    name: str
    mesh_index: int
    transform: Matrix4 = IDENTITY_4X4

    def copy(self, mesh_offset: int = 0) -> "MeshInstance":
        return MeshInstance(self.name, self.mesh_index + mesh_offset, self.transform)


@dataclass
class Scene:
    source_path: Path | None = None
    meshes: list[Mesh] = field(default_factory=list)
    materials: list[Material] = field(default_factory=lambda: [Material()])
    instances: list[MeshInstance] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=lambda: [Camera()])
    lights: list[Light] = field(default_factory=lambda: [Light()])
    metadata: dict[str, Any] = field(default_factory=dict)
    env_map: EnvMap | None = None
    selected_camera: int = 0
    camera_speed: float = 1.0

    @classmethod
    def default(cls) -> "Scene":
        mesh = Mesh(
            name="DefaultTriangle",
            positions=[(-0.8, -0.7, 0.0), (0.8, -0.7, 0.0), (0.0, 0.8, 0.0)],
            normals=[(0.0, 0.0, 1.0)] * 3,
            indices=[0, 1, 2],
        )
        return cls(meshes=[mesh], instances=[MeshInstance(name="DefaultTriangle", mesh_index=0)])

    @property
    def active_camera(self) -> Camera:
        self.ensure_defaults()
        index = min(max(self.selected_camera, 0), len(self.cameras) - 1)
        return self.cameras[index]

    def ensure_defaults(self) -> "Scene":
        if not self.materials:
            self.materials.append(Material())
        if not self.cameras:
            self.cameras.append(Camera())
        if not self.lights:
            self.lights.append(Light())
        if self.meshes and not self.instances:
            self.instances.extend(MeshInstance(name=mesh.name, mesh_index=i) for i, mesh in enumerate(self.meshes))
        return self

    def merge(self, other: "Scene", transform: Matrix4 | None = None) -> list[int]:
        material_offset = len(self.materials)
        mesh_offset = len(self.meshes)
        self.materials.extend(other.materials or [Material()])
        self.meshes.extend(mesh.copy(material_offset) for mesh in other.meshes)
        imported_instance_indices: list[int] = []
        root_transform = matrix_to_numpy(transform or IDENTITY_4X4)
        for instance in other.instances or [
            MeshInstance(mesh.name, idx) for idx, mesh in enumerate(other.meshes)
        ]:
            merged = instance.copy(mesh_offset)
            merged.transform = matrix_from_numpy(root_transform @ matrix_to_numpy(instance.transform))
            imported_instance_indices.append(len(self.instances))
            self.instances.append(merged)
        self.cameras.extend(other.cameras)
        self.lights.extend(other.lights)
        if other.env_map is not None:
            self.env_map = other.env_map
        return imported_instance_indices

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        for instance in self.instances:
            if not (0 <= instance.mesh_index < len(self.meshes)):
                continue
            mesh = self.meshes[instance.mesh_index]
            points.append(transform_points(instance.transform, mesh.position_array()))
        if not points:
            return np.asarray([-1.0, -1.0, -1.0], dtype=np.float32), np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        all_points = np.concatenate(points, axis=0)
        return all_points.min(axis=0), all_points.max(axis=0)

    def to_view(self) -> "SceneView":
        return SceneView(self.ensure_defaults())

    def to_manifest(self) -> dict[str, Any]:
        self.ensure_defaults()
        return {
            "schema": "dev.elrslang.scene.v1",
            "source": str(self.source_path) if self.source_path else None,
            "meshCount": len(self.meshes),
            "instanceCount": len(self.instances),
            "materialCount": len(self.materials),
            "lightCount": len(self.lights),
            "cameraCount": len(self.cameras),
            "meshes": [
                {
                    "name": mesh.name,
                    "vertexCount": mesh.vertex_count,
                    "triangleCount": mesh.triangle_count,
                    "materialIndex": mesh.material_index,
                }
                for mesh in self.meshes
            ],
            "meshInstances": [
                {
                    "name": instance.name,
                    "meshIndex": instance.mesh_index,
                    "transform": instance.transform,
                }
                for instance in self.instances
            ],
            "materials": [{"name": material.name, "baseColor": material.base_color} for material in self.materials],
            "lights": [{"name": light.name, "kind": light.kind} for light in self.lights],
            "cameras": [{"name": camera.name, "position": camera.position, "target": camera.target} for camera in self.cameras],
        }


@dataclass
class SceneView:
    scene: Scene

    def get_this(self) -> dict[str, Any]:
        camera = self.scene.active_camera
        bounds_min, bounds_max = self.scene.bounds()
        return {
            "_type": "SceneView",
            "meshCount": len(self.scene.meshes),
            "instanceCount": len(self.scene.instances),
            "materialCount": len(self.scene.materials),
            "lightCount": len(self.scene.lights),
            "camera": camera.get_this(),
            "boundsMin": tuple(float(v) for v in bounds_min),
            "boundsMax": tuple(float(v) for v in bounds_max),
        }


class SceneImportError(RuntimeError):
    pass


class MissingSceneDependency(SceneImportError):
    pass


class PysceneExecutionError(SceneImportError):
    pass


class SceneLoader:
    supported_extensions = {".obj", ".gltf", ".glb", ".fbx", ".usd", ".usda", ".usdc", ".pyscene"}

    def load(self, path: str | Path | None) -> Scene:
        if path is None:
            return Scene.default()
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        if suffix == ".obj":
            return self._load_obj(source).ensure_defaults()
        if suffix == ".gltf":
            return self._load_gltf_json(source).ensure_defaults()
        if suffix == ".glb":
            return self._load_glb(source).ensure_defaults()
        if suffix == ".fbx":
            return self._load_fbx(source).ensure_defaults()
        if suffix in {".usd", ".usda", ".usdc"}:
            return self._load_usd(source).ensure_defaults()
        if suffix == ".pyscene":
            return self._load_pyscene(source).ensure_defaults()
        raise SceneImportError(f"Unsupported scene extension `{suffix}`.")

    def _load_obj(self, path: Path) -> Scene:
        positions: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        uvs: list[tuple[float, float]] = []
        indices: list[int] = []

        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag, values = parts[0], parts[1:]
            if tag == "v" and len(values) >= 3:
                positions.append(tuple(float(v) for v in values[:3]))  # type: ignore[arg-type]
            elif tag == "vn" and len(values) >= 3:
                normals.append(tuple(float(v) for v in values[:3]))  # type: ignore[arg-type]
            elif tag == "vt" and len(values) >= 2:
                uvs.append(tuple(float(v) for v in values[:2]))  # type: ignore[arg-type]
            elif tag == "f":
                face = [self._obj_vertex_index(token, len(positions)) for token in values]
                indices.extend(self._triangulate(face))

        mesh = Mesh(name=path.stem, positions=positions, normals=normals, uvs=uvs, indices=indices)
        return Scene(source_path=path, meshes=[mesh], instances=[MeshInstance(path.stem, 0)])

    def _load_gltf_json(self, path: Path) -> Scene:
        data = json.loads(path.read_text(encoding="utf-8"))
        return self._load_gltf_data(data, path)

    def _load_glb(self, path: Path) -> Scene:
        raw = path.read_bytes()
        if len(raw) < 20 or raw[:4] != b"glTF":
            raise SceneImportError(f"Invalid GLB file `{path}`.")
        offset = 12
        json_chunk: bytes | None = None
        bin_chunk: bytes | None = None
        while offset + 8 <= len(raw):
            chunk_length = int.from_bytes(raw[offset : offset + 4], "little")
            chunk_type = raw[offset + 4 : offset + 8]
            offset += 8
            chunk = raw[offset : offset + chunk_length]
            offset += chunk_length
            if chunk_type == b"JSON":
                json_chunk = chunk
            elif chunk_type == b"BIN\x00":
                bin_chunk = chunk
        if json_chunk is None:
            raise SceneImportError(f"GLB file `{path}` does not contain a JSON chunk.")
        data = json.loads(json_chunk.decode("utf-8"))
        return self._load_gltf_data(data, path, bin_chunk)

    def _load_gltf_data(self, data: dict[str, Any], path: Path, binary_blob: bytes | None = None) -> Scene:
        buffers = [self._read_gltf_buffer(item, path, binary_blob) for item in data.get("buffers", [])]
        buffer_views = data.get("bufferViews", [])
        accessors = data.get("accessors", [])

        def read_accessor(index: int | None) -> np.ndarray:
            if index is None:
                return np.asarray([], dtype=np.float32)
            return self._read_gltf_accessor(accessors[index], buffer_views, buffers)

        materials = [
            Material(
                name=item.get("name", f"Material{idx}"),
                base_color=_vec4(
                    item.get("pbrMetallicRoughness", {}).get("baseColorFactor", (0.8, 0.8, 0.8, 1.0))
                ),
                roughness=float(item.get("pbrMetallicRoughness", {}).get("roughnessFactor", 0.5)),
                metallic=float(item.get("pbrMetallicRoughness", {}).get("metallicFactor", 0.0)),
            )
            for idx, item in enumerate(data.get("materials", []))
        ]
        meshes: list[Mesh] = []
        mesh_primitive_indices: dict[int, list[int]] = {}
        for mesh_index, item in enumerate(data.get("meshes", [])):
            mesh_primitive_indices[mesh_index] = []
            for primitive_index, primitive in enumerate(item.get("primitives", [{}])):
                attrs = primitive.get("attributes", {})
                positions = read_accessor(attrs.get("POSITION")).astype(np.float32).reshape(-1, 3)
                normals = read_accessor(attrs.get("NORMAL")).astype(np.float32).reshape(-1, 3)
                uvs = read_accessor(attrs.get("TEXCOORD_0")).astype(np.float32).reshape(-1, 2)
                raw_indices = read_accessor(primitive.get("indices")).astype(np.uint32).reshape(-1)
                if raw_indices.size == 0 and positions.size:
                    raw_indices = np.arange(positions.shape[0], dtype=np.uint32)
                mesh = Mesh(
                    name=item.get("name") or f"Mesh{mesh_index}_{primitive_index}",
                    positions=[tuple(float(v) for v in row) for row in positions],
                    normals=[tuple(float(v) for v in row) for row in normals] if normals.size else [],
                    uvs=[tuple(float(v) for v in row) for row in uvs] if uvs.size else [],
                    indices=[int(v) for v in raw_indices],
                    material_index=int(primitive.get("material", 0)),
                )
                mesh_primitive_indices[mesh_index].append(len(meshes))
                meshes.append(mesh)

        nodes = data.get("nodes", [])
        node_world = self._gltf_node_world_matrices(data)
        instances: list[MeshInstance] = []
        cameras: list[Camera] = []
        lights: list[Light] = []
        gltf_lights = data.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
        for idx, node in enumerate(nodes):
            transform = matrix_from_numpy(node_world.get(idx, np.eye(4, dtype=np.float32)))
            mesh_index = node.get("mesh")
            if isinstance(mesh_index, int):
                for primitive_mesh_index in mesh_primitive_indices.get(mesh_index, []):
                    instances.append(MeshInstance(node.get("name", f"Node{idx}"), primitive_mesh_index, transform))
            camera_index = node.get("camera")
            if isinstance(camera_index, int) and camera_index < len(data.get("cameras", [])):
                cameras.append(self._gltf_camera(data["cameras"][camera_index], node, node_world.get(idx)))
            light_ref = node.get("extensions", {}).get("KHR_lights_punctual", {}).get("light")
            if isinstance(light_ref, int) and light_ref < len(gltf_lights):
                lights.append(self._gltf_light(gltf_lights[light_ref], node, node_world.get(idx)))
        if not cameras:
            cameras = [self._gltf_camera(item, {"name": item.get("name", f"Camera{idx}")}, None) for idx, item in enumerate(data.get("cameras", []))]
        if not lights and gltf_lights:
            lights = [self._gltf_light(item, {"name": item.get("name", f"Light{idx}")}, None) for idx, item in enumerate(gltf_lights)]
        if not instances and meshes:
            instances = [MeshInstance(mesh.name, idx) for idx, mesh in enumerate(meshes)]
        return Scene(
            source_path=path,
            meshes=meshes,
            materials=materials or [Material()],
            instances=instances,
            cameras=cameras or [Camera()],
            lights=lights or [Light()],
            metadata={"asset": data.get("asset", {})},
        )

    def _load_assimp_like(self, path: Path) -> Scene:
        try:
            import trimesh  # type: ignore
        except Exception as exc:
            raise MissingSceneDependency(
                "FBX import requires an Assimp-compatible optional dependency such as `trimesh`."
            ) from exc

        loaded = trimesh.load(str(path), force="scene")
        geometries = loaded.geometry.items() if hasattr(loaded, "geometry") else [(path.stem, loaded)]
        meshes: list[Mesh] = []
        for name, mesh in geometries:
            positions = [tuple(float(c) for c in vertex) for vertex in mesh.vertices]
            normals = [tuple(float(c) for c in vertex) for vertex in getattr(mesh, "vertex_normals", [])]
            indices = [int(i) for face in mesh.faces for i in face[:3]]
            meshes.append(Mesh(name=name, positions=positions, normals=normals, indices=indices))
        instances = [MeshInstance(name=mesh.name, mesh_index=idx) for idx, mesh in enumerate(meshes)]
        return Scene(source_path=path, meshes=meshes, instances=instances, metadata={"generator": "trimesh"})

    def _load_fbx(self, path: Path) -> Scene:
        src_root = Path(__file__).resolve().parents[1]
        helper_code = (
            "import json, sys; "
            "from pathlib import Path; "
            f"sys.path.insert(0, {str(src_root)!r}); "
            "from elrslang.tools.fbx_import import load; "
            "print(json.dumps(load(Path(sys.argv[1])), separators=(',', ':')))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", helper_code, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout).strip()
            raise MissingSceneDependency(
                "FBX import requires optional dependency `ufbx` and a working helper process. "
                f"{message}"
            )
        data = json.loads(completed.stdout)
        return Scene(
            source_path=path,
            meshes=[
                Mesh(
                    name=item["name"],
                    positions=[tuple(row) for row in item["positions"]],
                    indices=[int(index) for index in item["indices"]],
                    material_index=int(item.get("materialIndex", 0)),
                )
                for item in data.get("meshes", [])
            ],
            materials=[
                Material(
                    name=item.get("name", "Material"),
                    base_color=tuple(item.get("baseColor", (0.78, 0.78, 0.78, 1.0))),
                    roughness=float(item.get("roughness", 0.5)),
                    metallic=float(item.get("metallic", 0.0)),
                )
                for item in data.get("materials", [])
            ],
            instances=[
                MeshInstance(
                    name=item["name"],
                    mesh_index=int(item["meshIndex"]),
                    transform=_matrix(item["transform"]),
                )
                for item in data.get("instances", [])
            ],
            lights=[
                Light(
                    name=item.get("name", "Light"),
                    kind=item.get("kind", "point"),
                    color=tuple(item.get("color", (1.0, 1.0, 1.0))),
                    intensity=float(item.get("intensity", 1.0)),
                    position=tuple(item.get("position", (0.0, 3.0, 0.0))),
                    direction=tuple(item.get("direction", (0.0, -1.0, 0.0))),
                )
                for item in data.get("lights", [])
            ],
            metadata=data.get("metadata", {"generator": "ufbx"}),
        )

    def _load_usd(self, path: Path) -> Scene:
        try:
            from pxr import Usd, UsdGeom, UsdLux  # type: ignore
        except Exception as exc:
            raise MissingSceneDependency("USD import requires optional dependency `usd-core`.") from exc

        stage = Usd.Stage.Open(str(path))
        meshes: list[Mesh] = []
        cameras: list[Camera] = []
        lights: list[Light] = []
        usd_light_types = [
            cls
            for cls in (
                getattr(UsdLux, "Light", None),
                getattr(UsdLux, "BoundableLightBase", None),
                getattr(UsdLux, "NonboundableLightBase", None),
                getattr(UsdLux, "DistantLight", None),
                getattr(UsdLux, "DomeLight", None),
                getattr(UsdLux, "SphereLight", None),
                getattr(UsdLux, "RectLight", None),
                getattr(UsdLux, "DiskLight", None),
                getattr(UsdLux, "CylinderLight", None),
            )
            if cls is not None
        ]
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                usd_mesh = UsdGeom.Mesh(prim)
                positions = [tuple(float(c) for c in p) for p in usd_mesh.GetPointsAttr().Get()]
                counts = list(usd_mesh.GetFaceVertexCountsAttr().Get())
                face_indices = list(usd_mesh.GetFaceVertexIndicesAttr().Get())
                indices: list[int] = []
                offset = 0
                for count in counts:
                    indices.extend(self._triangulate(face_indices[offset : offset + count]))
                    offset += count
                meshes.append(Mesh(name=prim.GetName(), positions=positions, indices=indices))
            elif prim.IsA(UsdGeom.Camera):
                cameras.append(Camera(name=prim.GetName()))
            elif any(prim.IsA(light_type) for light_type in usd_light_types):
                kind = "directional" if prim.IsA(getattr(UsdLux, "DistantLight", object)) else "point"
                lights.append(Light(name=prim.GetName(), kind=kind))
        instances = [MeshInstance(name=mesh.name, mesh_index=idx) for idx, mesh in enumerate(meshes)]
        return Scene(source_path=path, meshes=meshes, instances=instances, cameras=cameras, lights=lights)

    def _load_pyscene(self, path: Path) -> Scene:
        scene = Scene(source_path=path, materials=[], cameras=[], lights=[])
        builder = _PysceneSceneBuilder(scene, path.parent)
        previous_base_dir = TriangleMesh._base_dir
        TriangleMesh._base_dir = path.parent
        env: dict[str, Any] = {
            "__builtins__": MappingProxyType(
                {
                    "range": range,
                    "len": len,
                    "min": min,
                    "max": max,
                    "float": float,
                    "int": int,
                    "bool": bool,
                    "abs": abs,
                    "print": print,
                }
            ),
            "sceneBuilder": builder,
            "Camera": Camera,
            "Transform": Transform,
            "TriangleMesh": TriangleMesh,
            "Material": Material,
            "StandardMaterial": StandardMaterial,
            "PBRTDiffuseMaterial": PBRTDiffuseMaterial,
            "EnvMap": EnvMap,
            "PointLight": PointLight,
            "DistantLight": DistantLight,
            "DirectionalLight": DirectionalLight,
            "Animation": Animation,
            "MaterialTextureSlot": MaterialTextureSlot,
            "AABB": AABB,
            "float2": float2,
            "float3": float3,
            "float4": float4,
            "kInvalidNodeID": kInvalidNodeID,
        }
        try:
            code = compile(path.read_text(encoding="utf-8", errors="ignore"), str(path), "exec")
            exec(code, env, env)
        except Exception as exc:
            raise PysceneExecutionError(f"Failed to execute Falcor `.pyscene` `{path}`: {exc}") from exc
        finally:
            TriangleMesh._base_dir = previous_base_dir
        scene.ensure_defaults()
        return scene

    def _read_gltf_buffer(self, item: dict[str, Any], path: Path, binary_blob: bytes | None) -> bytes:
        uri = item.get("uri")
        if not uri:
            return binary_blob or b""
        if uri.startswith("data:"):
            _, encoded = uri.split(",", 1)
            return base64.b64decode(encoded)
        return (path.parent / uri).read_bytes()

    def _read_gltf_accessor(
        self,
        accessor: dict[str, Any],
        buffer_views: list[dict[str, Any]],
        buffers: list[bytes],
    ) -> np.ndarray:
        component_type = int(accessor.get("componentType", 5126))
        dtype = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }[component_type]
        components = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }[accessor.get("type", "SCALAR")]
        count = int(accessor.get("count", 0))
        if "bufferView" not in accessor or count == 0:
            return np.asarray([], dtype=dtype).reshape(0, components)
        view = buffer_views[int(accessor["bufferView"])]
        buffer = buffers[int(view.get("buffer", 0))]
        byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        stride = int(view.get("byteStride", np.dtype(dtype).itemsize * components))
        item_size = np.dtype(dtype).itemsize * components
        if stride == item_size:
            array = np.frombuffer(buffer, dtype=dtype, count=count * components, offset=byte_offset).copy()
            return array.reshape(count, components)
        rows = []
        for idx in range(count):
            start = byte_offset + idx * stride
            rows.append(np.frombuffer(buffer[start : start + item_size], dtype=dtype, count=components))
        return np.asarray(rows, dtype=dtype).reshape(count, components)

    def _gltf_node_world_matrices(self, data: dict[str, Any]) -> dict[int, np.ndarray]:
        nodes = data.get("nodes", [])
        local = [matrix_to_numpy(self._gltf_node_matrix(node)) for node in nodes]
        children = {idx: node.get("children", []) for idx, node in enumerate(nodes)}
        referenced = {child for child_list in children.values() for child in child_list}
        roots = [
            idx
            for scene in data.get("scenes", [])
            for idx in scene.get("nodes", [])
            if isinstance(idx, int)
        ] or [idx for idx in range(len(nodes)) if idx not in referenced]
        result: dict[int, np.ndarray] = {}

        def visit(node_index: int, parent: np.ndarray) -> None:
            world = parent @ local[node_index]
            result[node_index] = world
            for child_index in children.get(node_index, []):
                visit(child_index, world)

        for root in roots:
            if 0 <= root < len(nodes):
                visit(root, np.eye(4, dtype=np.float32))
        return result

    def _gltf_node_matrix(self, node: dict[str, Any]) -> Matrix4:
        if "matrix" in node:
            return matrix_from_numpy(np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4).T)
        return compose_trs(
            node.get("translation", (0.0, 0.0, 0.0)),
            node.get("rotation", (0.0, 0.0, 0.0, 1.0)),
            node.get("scale", (1.0, 1.0, 1.0)),
        )

    def _gltf_camera(self, item: dict[str, Any], node: dict[str, Any], world: np.ndarray | None) -> Camera:
        perspective = item.get("perspective", {})
        position = (0.0, 0.0, 4.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        if world is not None:
            position = tuple(float(v) for v in world[:3, 3])  # type: ignore[assignment]
            forward = normalize(-(world[:3, 2]), (0.0, 0.0, -1.0))
            target = tuple(float(v) for v in (world[:3, 3] + forward))  # type: ignore[assignment]
            up = tuple(float(v) for v in world[:3, 1])  # type: ignore[assignment]
        return Camera(
            name=node.get("name") or item.get("name", "Camera"),
            position=position,
            target=target,
            up=up,
            vfov_degrees=math.degrees(float(perspective.get("yfov", math.radians(45.0)))),
        )

    def _gltf_light(self, item: dict[str, Any], node: dict[str, Any], world: np.ndarray | None) -> Light:
        position = (0.0, 3.0, 0.0)
        direction = (-0.4, -1.0, -0.2)
        if world is not None:
            position = tuple(float(v) for v in world[:3, 3])  # type: ignore[assignment]
            direction = tuple(float(v) for v in normalize(-(world[:3, 2]), (0.0, -1.0, 0.0)))  # type: ignore[assignment]
        kind = item.get("type", "point")
        if kind == "spot":
            kind = "point"
        return Light(
            name=node.get("name") or item.get("name", "Light"),
            kind="directional" if kind == "directional" else "point",
            color=_vec3(item.get("color", (1.0, 1.0, 1.0)), (1.0, 1.0, 1.0)),
            intensity=float(item.get("intensity", 1.0)),
            position=position,
            direction=direction,
        )

    @staticmethod
    def _obj_vertex_index(token: str, vertex_count: int) -> int:
        first = token.split("/")[0]
        index = int(first)
        if index < 0:
            return vertex_count + index
        return index - 1

    @staticmethod
    def _triangulate(face: Iterable[int]) -> list[int]:
        values = list(face)
        if len(values) < 3:
            return []
        triangles: list[int] = []
        for i in range(1, len(values) - 1):
            triangles.extend([values[0], values[i], values[i + 1]])
        return triangles


def resolve_scene_relative_path(base_dir: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    direct = base_dir / candidate
    if direct.exists():
        return direct
    current = base_dir
    for parent in [current, *current.parents]:
        found = parent / candidate
        if found.exists():
            return found
        same_name = list(parent.rglob(candidate.name))
        if same_name:
            return same_name[0]
    return direct


@dataclass
class Transform:
    translation: Any = None
    scaling: Any = None
    rotationEuler: Any = None
    rotationEulerDeg: Any = None

    def matrix(self) -> Matrix4:
        return compose_matrix(
            translation=self.translation,
            scaling=1.0 if self.scaling is None else self.scaling,
            rotation_euler=self.rotationEuler,
            rotation_euler_degrees=self.rotationEulerDeg,
        )


@dataclass
class AABB:
    min_point: Any
    max_point: Any

    @property
    def minimum(self) -> tuple[float, float, float]:
        return _vec3(self.min_point)

    @property
    def maximum(self) -> tuple[float, float, float]:
        return _vec3(self.max_point)


class NodeID(int):
    pass


class MeshID(int):
    pass


@dataclass
class _ScriptNode:
    name: str
    transform: Matrix4
    parent: int = -1
    instance_index: int | None = None


kInvalidNodeID = -1


class _PysceneSceneBuilder:
    def __init__(self, scene: Scene, base_dir: Path) -> None:
        self.scene = scene
        self.base_dir = base_dir
        self.nodes: list[_ScriptNode] = []
        self.animations: list[_AnimationSettings] = [_AnimationSettings() for _ in range(64)]

    @property
    def cameras(self) -> list[Camera]:
        return self.scene.cameras

    @property
    def materials(self) -> list[Material]:
        return self.scene.materials

    @property
    def lights(self) -> list[Light]:
        return self.scene.lights

    @property
    def envMap(self) -> EnvMap | None:
        return self.scene.env_map

    @envMap.setter
    def envMap(self, value: EnvMap | None) -> None:
        self.scene.env_map = value

    @property
    def selectedCamera(self) -> Camera:
        return self.scene.active_camera

    @selectedCamera.setter
    def selectedCamera(self, camera: Camera) -> None:
        if camera not in self.scene.cameras:
            self.addCamera(camera)
        self.scene.selected_camera = self.scene.cameras.index(camera)

    @property
    def cameraSpeed(self) -> float:
        return self.scene.camera_speed

    @cameraSpeed.setter
    def cameraSpeed(self, value: float) -> None:
        self.scene.camera_speed = float(value)

    def importScene(
        self,
        path: str,
        options: dict[str, Any] | None = None,
        instances: list[Transform] | None = None,
    ) -> None:
        source = resolve_scene_relative_path(self.base_dir, path)
        imported = SceneLoader().load(source)
        if len(imported.cameras) == 1 and imported.cameras[0].name == "DefaultCamera":
            imported.cameras = []
        if len(imported.lights) == 1 and imported.lights[0].name == "KeyLight":
            imported.lights = []
        transforms = instances or [Transform()]
        for transform in transforms:
            mesh_offset = len(self.scene.meshes)
            imported_indices = self.scene.merge(imported, transform.matrix())
            for idx in imported_indices:
                instance = self.scene.instances[idx]
                self.nodes.append(_ScriptNode(instance.name, instance.transform, kInvalidNodeID, idx))
            # Only duplicate imported meshes once when there is more than one transform.
            if len(transforms) > 1:
                imported = Scene(
                    source_path=source,
                    meshes=[mesh.copy(-mesh_offset) for mesh in self.scene.meshes[mesh_offset:]],
                    materials=list(imported.materials),
                    instances=[inst.copy(-mesh_offset) for inst in imported.instances],
                    cameras=[],
                    lights=[],
                )

    def addNode(self, name: str, transform: Transform | None = None, parent: int = kInvalidNodeID) -> NodeID:
        node = _ScriptNode(name, (transform or Transform()).matrix(), int(parent))
        self.nodes.append(node)
        return NodeID(len(self.nodes) - 1)

    def getNodeCount(self) -> int:
        return len(self.nodes)

    def getNodeParent(self, node_id: int) -> int:
        if 0 <= int(node_id) < len(self.nodes):
            return self.nodes[int(node_id)].parent
        return kInvalidNodeID

    def setNodeTransform(self, node_id: int, transform: Transform) -> None:
        node = self.nodes[int(node_id)]
        node.transform = transform.matrix()
        if node.instance_index is not None:
            self.scene.instances[node.instance_index].transform = node.transform

    def addTriangleMesh(self, triangle_mesh: TriangleMesh, material: Material | None = None) -> MeshID:
        mesh = triangle_mesh.mesh.copy()
        if material is not None:
            mesh.material_index = self.addMaterial(material)
        self.scene.meshes.append(mesh)
        return MeshID(len(self.scene.meshes) - 1)

    def addMeshInstance(self, a: int, b: int) -> int:
        if isinstance(a, NodeID) and isinstance(b, MeshID):
            node_id, mesh_id = int(a), int(b)
        elif isinstance(a, MeshID) and isinstance(b, NodeID):
            mesh_id, node_id = int(a), int(b)
        else:
            # Falcor samples pass either order; prefer the ID wrapper when available.
            node_id, mesh_id = int(a), int(b)
        node = self.nodes[node_id]
        instance = MeshInstance(node.name, mesh_id, node.transform)
        self.scene.instances.append(instance)
        node.instance_index = len(self.scene.instances) - 1
        return node.instance_index

    def addCustomPrimitive(self, user_id: int, bounds: AABB) -> int:
        minimum = np.asarray(bounds.minimum, dtype=np.float32)
        maximum = np.asarray(bounds.maximum, dtype=np.float32)
        center = (minimum + maximum) * 0.5
        scale = maximum - minimum
        material = Material(name=f"CustomPrimitive{user_id}", base_color=(0.8, 0.72, 0.35, 1.0))
        mesh_id = self.addTriangleMesh(TriangleMesh.createCube(), material)
        node_id = self.addNode(
            f"CustomPrimitive{user_id}",
            Transform(translation=tuple(float(v) for v in center), scaling=tuple(float(v) for v in scale)),
        )
        return self.addMeshInstance(node_id, mesh_id)

    def addMaterial(self, material: Material) -> int:
        for idx, existing in enumerate(self.scene.materials):
            if existing is material:
                return idx
        self.scene.materials.append(material)
        return len(self.scene.materials) - 1

    def getMaterial(self, name_or_index: str | int) -> Material | None:
        if isinstance(name_or_index, int):
            if 0 <= name_or_index < len(self.scene.materials):
                return self.scene.materials[name_or_index]
            return None
        for material in self.scene.materials:
            if material.name == name_or_index:
                return material
        return None

    def addLight(self, light: Light) -> int:
        self.scene.lights.append(light)
        return len(self.scene.lights) - 1

    def getLight(self, name: str) -> Light | None:
        for light in self.scene.lights:
            if light.name == name:
                return light
        return None

    def addCamera(self, camera: Camera) -> int:
        self.scene.cameras.append(camera)
        if len(self.scene.cameras) == 1:
            self.scene.selected_camera = 0
        return len(self.scene.cameras) - 1


def PointLight(name: str = "PointLight") -> Light:
    return Light(name=name, kind="point")


def DistantLight(name: str = "DistantLight") -> Light:
    return Light(name=name, kind="directional")


def DirectionalLight(name: str = "DirectionalLight") -> Light:
    return Light(name=name, kind="directional")
