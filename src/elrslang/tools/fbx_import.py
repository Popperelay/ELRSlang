from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _vec3(value: Any) -> tuple[float, float, float]:
    return (float(value.x), float(value.y), float(value.z))


def _vec4(value: Any) -> tuple[float, float, float, float]:
    return (float(value.x), float(value.y), float(value.z), float(value.w))


def _matrix(value: Any) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0], matrix[1, 0], matrix[2, 0] = _vec3(value.c0)
    matrix[0, 1], matrix[1, 1], matrix[2, 1] = _vec3(value.c1)
    matrix[0, 2], matrix[1, 2], matrix[2, 2] = _vec3(value.c2)
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = _vec3(value.c3)
    return [[float(c) for c in row] for row in matrix]


def _normalize(value: np.ndarray, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    length = float(np.linalg.norm(value))
    if length <= 1e-8:
        return fallback
    return tuple(float(c) for c in (value / length))


def _triangulate(face: list[int]) -> list[int]:
    if len(face) < 3:
        return []
    indices: list[int] = []
    for i in range(1, len(face) - 1):
        indices.extend([face[0], face[i], face[i + 1]])
    return indices


def _map_vec4(value: Any, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if value is not None and getattr(value, "has_value", False):
        return _vec4(value.value_vec4)
    return fallback


def _material(value: Any, index: int) -> dict[str, Any]:
    base_color = _map_vec4(
        getattr(value.pbr, "base_color", None),
        _map_vec4(getattr(value.fbx, "diffuse_color", None), (0.78, 0.78, 0.78, 1.0)),
    )
    roughness_map = getattr(value.pbr, "roughness", None)
    metallic_map = getattr(value.pbr, "metalness", None)
    return {
        "name": value.name or f"FBXMaterial{index}",
        "baseColor": base_color,
        "roughness": float(getattr(getattr(roughness_map, "value_vec4", None), "x", 0.5)),
        "metallic": float(getattr(getattr(metallic_map, "value_vec4", None), "x", 0.0)),
    }


def _light(value: Any, name: str, transform: Any, ufbx: Any) -> dict[str, Any]:
    matrix = np.asarray(_matrix(transform), dtype=np.float32)
    return {
        "name": name,
        "kind": "directional" if value.type == ufbx.LightType.DIRECTIONAL else "point",
        "color": _vec3(value.color),
        "intensity": float(getattr(value, "intensity", 1.0)),
        "position": tuple(float(v) for v in matrix[:3, 3]),
        "direction": _normalize(-matrix[:3, 2], (0.0, -1.0, 0.0)),
    }


def load(path: Path) -> dict[str, Any]:
    import ufbx  # type: ignore

    loaded = ufbx.load_file(str(path))
    materials = [_material(loaded.materials[idx], idx) for idx in range(len(loaded.materials))]
    meshes: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    lights: list[dict[str, Any]] = []

    for idx in range(len(loaded.nodes)):
        node = loaded.nodes[idx]
        name = node.name or f"Node{idx}"
        mesh = getattr(node, "mesh", None)
        if mesh is None:
            node_light = getattr(node, "light", None)
            if node_light is not None:
                lights.append(_light(node_light, name, node.geometry_to_world, ufbx))
            continue

        positions = [_vec3(mesh.vertices[i]) for i in range(len(mesh.vertices))]
        indices: list[int] = []
        for face_index in range(mesh.num_faces):
            face = mesh.faces[face_index]
            polygon = [
                int(mesh.vertex_indices[face.index_begin + offset])
                for offset in range(face.num_indices)
            ]
            indices.extend(_triangulate(polygon))

        mesh_typed_id = int(getattr(mesh, "typed_id", len(meshes)))
        material_index = min(mesh_typed_id, len(materials) - 1) if materials else 0
        meshes.append(
            {
                "name": mesh.name or name or f"{path.stem}_mesh_{idx}",
                "positions": positions,
                "indices": indices,
                "materialIndex": material_index,
            }
        )
        instances.append(
            {
                "name": name,
                "meshIndex": len(meshes) - 1,
                "transform": _matrix(node.geometry_to_world),
            }
        )

    return {
        "materials": materials,
        "meshes": meshes,
        "instances": instances,
        "lights": lights,
        "metadata": {"generator": "ufbx", "animationCount": len(loaded.anim_stacks)},
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: fbx_import.py <scene.fbx>", file=sys.stderr)
        return 2
    try:
        result = load(Path(args[0]))
        print(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from elrslang.tools.fbx_import import main as package_main

    exit_code = package_main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
