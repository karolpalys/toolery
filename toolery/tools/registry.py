from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict  # full OpenAI tool schema
    impl: Callable[..., Any] | None = None


class ToolRegistry:
    _global: ToolRegistry | None = None

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def openai_schemas(self, names: list[str]) -> list[dict]:
        return [self.get(n).json_schema for n in names]

    @classmethod
    def default(cls) -> ToolRegistry:
        if cls._global is None:
            cls._global = cls()
        return cls._global


def register(spec: ToolSpec) -> None:
    ToolRegistry.default().register(spec)
