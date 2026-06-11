from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from toolery.core.models import Scenario


class DuplicateIdError(ValueError):
    pass


_TIER_PREFIX_RE = re.compile(r"^(?:easy|medium|hard|very-hard)-")


def display_name(scenario_id: str) -> str:
    """Scenario id with its historical tier prefix stripped, for display next
    to the (now empirical) tier column.

    Scenario ids carry a leading ``easy-``/``hard-``/etc. prefix from when tier
    was hand-assigned. After empirical re-tiering the prefix no longer matches
    the real tier, so showing the raw id beside the tier column reads as a
    contradiction (``easy-39`` tagged ``hard``). The id stays unchanged as a
    stable key; only presentation drops the prefix so the tier column is the
    single source of truth for difficulty.
    """
    return _TIER_PREFIX_RE.sub("", scenario_id)


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
