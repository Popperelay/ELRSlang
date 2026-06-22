"""Scene schema and lightweight importers.

The first version intentionally keeps the CPU scene representation compact and
SlangPy-friendly. Importers normalize source data once, then passes consume a
`SceneView` object whose `get_this()` can be mapped directly to a Slang struct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


Matrix4 = tuple[tuple[float, float, float, float], ...]


IDENTITY_4X4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass
class Camera:
    name: str = "DefaultCamera"
    position: tuple[float, float, float] = (0.0, 0.0, 4.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vfov_degrees: float = 45.0

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
    intensity: float = 1.0
    direction: tuple[float, float, float] = (-0.4, -1.0, -0.2)
    position: tuple[float, float, float] = (0.0, 3.0, 0.0)

    def get_this(self) -> dict[str, Any]:
        return {
            "_type": "Light",
            "kind": 0 if self.kind == "directional" else 1,
            "color": self.color,
            "intensity": self.intensity,
            "direction": self.direction,
            "position": self.position,
        }


@dataclass
class Material:
    name: str = "DefaultMaterial"
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0

    def get_this(self) -> dict[str, Any]:
        return {
            "_type": "StandardMaterial",
            "baseColor": self.base_color,
            "roughness": self.roughness,
            "metallic": self.metallic,
        }


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
        return len(self.indices) // 3

    def position_array(self) -> np.ndarray:
        if self.positions:
            return np.asarray(self.positions, dtype=np.float32)
        return np.asarray([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)], dtype=np.float32)

    def index_array(self) -> np.ndarray:
        if self.indices:
            return np.asarray(self.indices, dtype=np.uint32)
        return np.asarray([0, 1, 2], dtype=np.uint32)


@dataclass
class MeshInstance:
    name: str
    mesh_index: int
    transform: Matrix4 = IDENTITY_4X4


@dataclass
class Scene:
    source_path: Path | None = None
    meshes: list[Mesh] = field(default_factory=list)
    materials: list[Material] = field(default_factory=lambda: [Material()])
    instances: list[MeshInstance] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=lambda: [Camera()])
    lights: list[Light] = field(default_factory=lambda: [Light()])
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Scene":
        mesh = Mesh(
            name="DefaultTriangle",
            positions=[(-0.8, -0.7, 0.0), (0.8, -0.7, 0.0), (0.0, 0.8, 0.0)],
            indices=[0, 1, 2],
        )
        return cls(meshes=[mesh], instances=[MeshInstance(name="DefaultTriangle", mesh_index=0)])

    def ensure_defaults(self) -> "Scene":
        if not self.materials:
            self.materials.append(Material())
        if not self.cameras:
            self.cameras.append(Camera())
        if not self.lights:
            self.lights.append(Light())
        if self.meshes and not self.instances:
            self.instances.append(MeshInstance(name=self.meshes[0].name, mesh_index=0))
        return self

    def to_view(self) -> "SceneView":
        return SceneView(self.ensure_defaults())


@dataclass
class SceneView:
    scene: Scene

    def get_this(self) -> dict[str, Any]:
        camera = self.scene.cameras[0] if self.scene.cameras else Camera()
        return {
            "_type": "SceneView",
            "meshCount": len(self.scene.meshes),
            "instanceCount": len(self.scene.instances),
            "materialCount": len(self.scene.materials),
            "lightCount": len(self.scene.lights),
            "camera": camera.get_this(),
        }


class SceneImportError(RuntimeError):
    pass


class MissingSceneDependency(SceneImportError):
    pass


class SceneLoader:
    supported_extensions = {".obj", ".gltf", ".glb", ".fbx", ".usd", ".usda", ".usdc"}

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
            return self._load_assimp_like(source).ensure_defaults()
        if suffix in {".usd", ".usda", ".usdc"}:
            return self._load_usd(source).ensure_defaults()
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
        materials = [
            Material(
                name=item.get("name", f"Material{idx}"),
                base_color=tuple(
                    item.get("pbrMetallicRoughness", {}).get(
                        "baseColorFactor", (0.8, 0.8, 0.8, 1.0)
                    )
                ),  # type: ignore[arg-type]
                roughness=float(item.get("pbrMetallicRoughness", {}).get("roughnessFactor", 0.5)),
                metallic=float(item.get("pbrMetallicRoughness", {}).get("metallicFactor", 0.0)),
            )
            for idx, item in enumerate(data.get("materials", []))
        ]
        meshes = []
        for mesh_index, item in enumerate(data.get("meshes", [])):
            primitives = item.get("primitives", [{}])
            for primitive_index, primitive in enumerate(primitives):
                meshes.append(
                    Mesh(
                        name=item.get("name", f"Mesh{mesh_index}_{primitive_index}"),
                        material_index=int(primitive.get("material", 0)),
                    )
                )
        cameras = [Camera(name=item.get("name", f"Camera{idx}")) for idx, item in enumerate(data.get("cameras", []))]
        instances = self._gltf_instances(data, len(meshes))
        lights = self._gltf_lights(data)
        return Scene(
            source_path=path,
            meshes=meshes,
            materials=materials or [Material()],
            instances=instances,
            cameras=cameras or [Camera()],
            lights=lights or [Light()],
            metadata={"asset": data.get("asset", {})},
        )

    def _load_glb(self, path: Path) -> Scene:
        try:
            from pygltflib import GLTF2  # type: ignore
        except Exception as exc:
            raise MissingSceneDependency("GLB import requires optional dependency `pygltflib`.") from exc
        gltf = GLTF2().load(str(path))
        meshes = [Mesh(name=mesh.name or f"Mesh{idx}") for idx, mesh in enumerate(gltf.meshes or [])]
        return Scene(source_path=path, meshes=meshes, metadata={"generator": "pygltflib"}).ensure_defaults()

    def _load_assimp_like(self, path: Path) -> Scene:
        try:
            import trimesh  # type: ignore
        except Exception as exc:
            raise MissingSceneDependency(
                "FBX import requires an Assimp-compatible optional dependency such as `trimesh`."
            ) from exc

        loaded = trimesh.load(str(path), force="scene")
        meshes: list[Mesh] = []
        for name, mesh in loaded.geometry.items():
            positions = [tuple(float(c) for c in vertex) for vertex in mesh.vertices]
            indices = [int(i) for face in mesh.faces for i in face[:3]]
            meshes.append(Mesh(name=name, positions=positions, indices=indices))
        instances = [MeshInstance(name=mesh.name, mesh_index=idx) for idx, mesh in enumerate(meshes)]
        return Scene(source_path=path, meshes=meshes, instances=instances, metadata={"generator": "trimesh"})

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
                meshes.append(Mesh(name=prim.GetName()))
            elif prim.IsA(UsdGeom.Camera):
                cameras.append(Camera(name=prim.GetName()))
            elif any(prim.IsA(light_type) for light_type in usd_light_types):
                lights.append(Light(name=prim.GetName(), kind="point"))
        instances = [MeshInstance(name=mesh.name, mesh_index=idx) for idx, mesh in enumerate(meshes)]
        return Scene(source_path=path, meshes=meshes, instances=instances, cameras=cameras, lights=lights)

    def _gltf_instances(self, data: dict[str, Any], mesh_count: int) -> list[MeshInstance]:
        instances: list[MeshInstance] = []
        for idx, node in enumerate(data.get("nodes", [])):
            mesh_index = node.get("mesh")
            if isinstance(mesh_index, int) and 0 <= mesh_index < max(mesh_count, 1):
                instances.append(MeshInstance(name=node.get("name", f"Node{idx}"), mesh_index=mesh_index))
        return instances

    def _gltf_lights(self, data: dict[str, Any]) -> list[Light]:
        ext = data.get("extensions", {}).get("KHR_lights_punctual", {})
        result: list[Light] = []
        for idx, item in enumerate(ext.get("lights", [])):
            result.append(
                Light(
                    name=item.get("name", f"Light{idx}"),
                    kind=item.get("type", "point"),
                    color=tuple(item.get("color", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
                    intensity=float(item.get("intensity", 1.0)),
                )
            )
        return result

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
