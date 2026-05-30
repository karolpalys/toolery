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
    # Cloud needs an API key — without it, a cloud-API call will 401.
    if env.get("OPENAI_API_KEY") or env.get("ANTHROPIC_API_KEY"):
        out["cloud"] = AdapterStatus(available=True)
    else:
        out["cloud"] = AdapterStatus(
            available=False, reason="set OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )
    out["hermes"] = (
        AdapterStatus(available=True)
        if path_lookup("hermes")
        else AdapterStatus(available=False, reason="hermes CLI not in PATH")
    )
    return out
