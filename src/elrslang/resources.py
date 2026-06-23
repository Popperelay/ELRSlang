"""Resource descriptors and runtime registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextureDesc:
    width: int
    height: int
    format: str = "rgba32_float"
    label: str = ""


@dataclass
class ResourceHandle:
    name: str
    value: Any
    desc: TextureDesc | None = None


class ResourceRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ResourceHandle] = {}

    def set(self, name: str, value: Any, desc: TextureDesc | None = None) -> Any:
        self._items[name] = ResourceHandle(name=name, value=value, desc=desc)
        return value

    def get(self, name: str, default: Any = None) -> Any:
        handle = self._items.get(name)
        return default if handle is None else handle.value

    def describe(self, name: str) -> TextureDesc | None:
        handle = self._items.get(name)
        return None if handle is None else handle.desc

    def require(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"Resource `{name}` was not produced by the render graph.")
        return self._items[name].value

    def has(self, name: str) -> bool:
        return name in self._items

    def items(self):
        return self._items.items()

    def clear_transient(self) -> None:
        self._items.clear()
