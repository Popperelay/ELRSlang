"""Falcor-inspired render graph primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .resources import ResourceRegistry


class GraphCompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceRef:
    pass_name: str
    resource_name: str

    @classmethod
    def parse(cls, value: str) -> "ResourceRef":
        if "." not in value:
            raise GraphCompileError(f"Resource edge endpoint `{value}` must use `Pass.resource`.")
        pass_name, resource_name = value.split(".", 1)
        if not pass_name or not resource_name:
            raise GraphCompileError(f"Invalid resource endpoint `{value}`.")
        return cls(pass_name, resource_name)

    def key(self) -> str:
        return f"{self.pass_name}.{self.resource_name}"


@dataclass(frozen=True)
class PassReflection:
    inputs: frozenset[str] = field(default_factory=frozenset)
    outputs: frozenset[str] = field(default_factory=frozenset)
    optional_inputs: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_iterables(
        cls,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        optional_inputs: Iterable[str] = (),
    ) -> "PassReflection":
        return cls(frozenset(inputs), frozenset(outputs), frozenset(optional_inputs))


@dataclass
class FrameState:
    frame_index: int = 0
    time_seconds: float = 0.0


@dataclass
class RenderContext:
    device: Any = None
    width: int = 1280
    height: int = 720
    scene: Any = None
    app: Any = None
    settings: dict[str, Any] = field(default_factory=dict)
    shader_paths: list[Path] = field(default_factory=list)
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)
    frame: FrameState = field(default_factory=FrameState)
    output: Any = None

    def frame_constants(self) -> dict[str, Any]:
        return {
            "_type": "FrameConstants",
            "frameIndex": self.frame.frame_index,
            "resolution": (float(self.width), float(self.height)),
            "timeSeconds": float(self.frame.time_seconds),
        }


class RenderPass:
    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("RenderPass name cannot be empty.")
        self.name = name

    def reflect(self) -> PassReflection:
        raise NotImplementedError

    def execute(self, context: RenderContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


class RenderGraph:
    def __init__(self, name: str = "RenderGraph") -> None:
        self.name = name
        self._passes: dict[str, RenderPass] = {}
        self._edges: list[tuple[ResourceRef, ResourceRef]] = []
        self._outputs: list[ResourceRef] = []
        self._compiled_order: list[str] = []

    @property
    def passes(self) -> Mapping[str, RenderPass]:
        return self._passes

    @property
    def edges(self) -> tuple[tuple[ResourceRef, ResourceRef], ...]:
        return tuple(self._edges)

    def add_pass(self, render_pass: RenderPass) -> None:
        if render_pass.name in self._passes:
            raise GraphCompileError(f"Pass `{render_pass.name}` already exists.")
        self._passes[render_pass.name] = render_pass

    def add_edge(self, source: str, destination: str) -> None:
        self._edges.append((ResourceRef.parse(source), ResourceRef.parse(destination)))

    def mark_output(self, endpoint: str) -> None:
        self._outputs.append(ResourceRef.parse(endpoint))

    def compile(self) -> list[str]:
        self._validate_edges()
        self._validate_required_inputs()
        order = self._topological_sort()
        self._validate_outputs()
        self._compiled_order = order
        return order

    def execute(self, context: RenderContext) -> ResourceRegistry:
        if not self._compiled_order:
            self.compile()

        edge_inputs: dict[str, dict[str, Any]] = {name: {} for name in self._passes}
        for pass_name in self._compiled_order:
            render_pass = self._passes[pass_name]
            produced = render_pass.execute(context, edge_inputs[pass_name])
            for output_name, value in produced.items():
                context.resources.set(f"{pass_name}.{output_name}", value)
            for source, destination in self._edges:
                if source.pass_name == pass_name:
                    value = context.resources.require(source.key())
                    edge_inputs[destination.pass_name][destination.resource_name] = value

        if self._outputs:
            last = self._outputs[-1]
            context.output = context.resources.require(last.key())
        context.frame.frame_index += 1
        return context.resources

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        pass_factory: Callable[[dict[str, Any]], RenderPass],
    ) -> "RenderGraph":
        graph_path = Path(path)
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        graph = cls(data.get("name", graph_path.stem))
        for item in data.get("passes", []):
            graph.add_pass(pass_factory(item))
        for edge in data.get("edges", []):
            if isinstance(edge, str):
                source, destination = edge.split("->", 1)
                graph.add_edge(source.strip(), destination.strip())
            else:
                graph.add_edge(edge["from"], edge["to"])
        for output in data.get("outputs", []):
            graph.mark_output(output)
        return graph

    def _validate_edges(self) -> None:
        for source, destination in self._edges:
            if source.pass_name not in self._passes:
                raise GraphCompileError(f"Edge source pass `{source.pass_name}` does not exist.")
            if destination.pass_name not in self._passes:
                raise GraphCompileError(
                    f"Edge destination pass `{destination.pass_name}` does not exist."
                )
            source_reflection = self._passes[source.pass_name].reflect()
            destination_reflection = self._passes[destination.pass_name].reflect()
            if source.resource_name not in source_reflection.outputs:
                raise GraphCompileError(
                    f"Pass `{source.pass_name}` does not output `{source.resource_name}`."
                )
            if destination.resource_name not in destination_reflection.inputs:
                raise GraphCompileError(
                    f"Pass `{destination.pass_name}` does not input `{destination.resource_name}`."
                )

    def _validate_required_inputs(self) -> None:
        connected = {(dst.pass_name, dst.resource_name) for _, dst in self._edges}
        for name, render_pass in self._passes.items():
            reflection = render_pass.reflect()
            for input_name in reflection.inputs - reflection.optional_inputs:
                if (name, input_name) not in connected:
                    raise GraphCompileError(f"Required input `{name}.{input_name}` has no edge.")

    def _validate_outputs(self) -> None:
        for output in self._outputs:
            if output.pass_name not in self._passes:
                raise GraphCompileError(f"Graph output pass `{output.pass_name}` does not exist.")
            if output.resource_name not in self._passes[output.pass_name].reflect().outputs:
                raise GraphCompileError(
                    f"Graph output `{output.key()}` is not declared by its pass reflection."
                )

    def _topological_sort(self) -> list[str]:
        adjacency: dict[str, set[str]] = {name: set() for name in self._passes}
        indegree: dict[str, int] = {name: 0 for name in self._passes}
        for source, destination in self._edges:
            if destination.pass_name not in adjacency[source.pass_name]:
                adjacency[source.pass_name].add(destination.pass_name)
                indegree[destination.pass_name] += 1

        ready = sorted(name for name, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for child in sorted(adjacency[name]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()

        if len(order) != len(self._passes):
            raise GraphCompileError("Render graph contains a cycle.")
        return order
