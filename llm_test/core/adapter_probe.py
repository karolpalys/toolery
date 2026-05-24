from __future__ import annotations

import os
import shutil
from collections.abc import Callable

from pydantic import BaseModel


class AdapterStatus(BaseModel):
    available: bool
    reason: str | None = None


def available_adapters(
    path_lookup: Callable[[str], str | None] = shutil.which,
    env: dict[str, str] | None = None,
) -> dict[str, AdapterStatus]:
    env = env if env is not None else dict(os.environ)
    out: dict[str, AdapterStatus] = {
        "raw": AdapterStatus(available=True),
    }
    out["hermes"] = (
        AdapterStatus(available=True)
        if path_lookup("hermes")
        else AdapterStatus(available=False, reason="hermes CLI not in PATH")
    )
    if env.get("CLAUDE_CLI_PATH") or path_lookup("claude"):
        out["claude_code"] = AdapterStatus(available=True)
    else:
        out["claude_code"] = AdapterStatus(
            available=False, reason="set CLAUDE_CLI_PATH or install claude"
        )
    if env.get("CODEX_CLI_PATH") or path_lookup("codex"):
        out["codex"] = AdapterStatus(available=True)
    else:
        out["codex"] = AdapterStatus(
            available=False, reason="set CODEX_CLI_PATH or install codex"
        )
    return out
