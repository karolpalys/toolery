from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from toolery.core.models import Scenario


class DuplicateIdError(ValueError):
    pass


def load_scenario(path: Path) -> Scenario:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return Scenario.model_validate(data)


def scenario_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_all_scenarios(root: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}
    for p in sorted(root.rglob("*.yaml")):
        s = load_scenario(p)
        if s.id in seen:
            raise DuplicateIdError(f"duplicate id {s.id!r} in {p} and {seen[s.id]}")
        seen[s.id] = p
        scenarios.append(s)
    return scenarios
