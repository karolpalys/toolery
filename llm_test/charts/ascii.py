from __future__ import annotations


def _bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    return "▇" * filled + "░" * (width - filled)


def bar_per_tier(data: dict[str, dict[str, float]],
                 tiers: tuple[str, ...] = ("easy", "medium", "hard", "very_hard")) -> str:
    lines = ["", "        " + "  ".join(f"{t:<10}" for t in tiers)]
    for adapter, vals in data.items():
        parts = [f"{_bar(vals.get(t, 0.0))} " for t in tiers]
        lines.append(f"{adapter:<8}  " + "  ".join(parts))
    lines.append("        " + "  ".join(f"{vals.get(t, 0.0)*100:>5.0f}%      "
                                         for t in tiers))
    return "\n".join(lines)


def heatmap(data: dict[str, dict[str, float]]) -> str:
    categories = sorted({c for vals in data.values() for c in vals.keys()})
    col_w = max((len(c) for c in categories), default=8)
    col_w = max(col_w, 8)
    header = "            " + "  ".join(f"{c:<{col_w}}" for c in categories)
    lines = [header]
    for adapter, vals in data.items():
        cells = [f"{_bar(vals.get(c, 0.0), width=col_w):<{col_w}}" for c in categories]
        lines.append(f"{adapter:<12}" + "  ".join(cells))
    return "\n".join(lines)


def failure_taxonomy(counts: dict[str, int], width: int = 24) -> str:
    if not counts:
        return "(no failures)"
    total = sum(counts.values())
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    lines = []
    for kind, n in items:
        ratio = n / total
        filled = int(round(ratio * width))
        bar = "█" * filled + "░" * (width - filled)
        lines.append(f"{kind:<22} {bar} {ratio*100:>5.1f}%  ({n})")
    return "\n".join(lines)
