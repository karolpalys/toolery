from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def overall_bar(model_to_score_ci: dict[str, tuple[float, float]], out: Path) -> None:
    names = list(model_to_score_ci.keys())
    scores = [v[0] for v in model_to_score_ci.values()]
    ci = [v[1] for v in model_to_score_ci.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x, scores, yerr=ci, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Overall pass-rate")
    ax.set_title("Overall pass-rate per model (95% CI)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def tier_breakdown(data: dict[str, dict[str, float]], out: Path) -> None:
    tiers = ["easy", "medium", "hard", "very_hard"]
    adapters = list(data.keys())
    x = np.arange(len(tiers))
    width = 0.8 / max(len(adapters), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, a in enumerate(adapters):
        vals = [data[a].get(t, 0.0) for t in tiers]
        ax.bar(x + i * width, vals, width, label=a)
    ax.set_xticks(x + width * (len(adapters) - 1) / 2)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pass-rate")
    ax.set_title("Pass-rate per tier × adapter")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def category_heatmap(data: dict[str, dict[str, float]], out: Path) -> None:
    adapters = list(data.keys())
    cats = sorted({c for v in data.values() for c in v.keys()})
    mat = np.array([[data[a].get(c, 0.0) for c in cats] for a in adapters])
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cats)), 0.6 * len(adapters) + 1))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticks(range(len(adapters)))
    ax.set_yticklabels(adapters)
    for i in range(len(adapters)):
        for j in range(len(cats)):
            ax.text(
                j,
                i,
                f"{mat[i, j]*100:.0f}%",
                ha="center",
                va="center",
                color="black" if mat[i, j] > 0.5 else "white",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="Pass-rate")
    ax.set_title("Category × adapter heatmap")
    _save(fig, out)


def radar(model_to_scores: dict[str, list[float]], categories: list[str], out: Path) -> None:
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for model, scores in model_to_scores.items():
        vals = list(scores) + [scores[0]]
        ax.plot(angles, vals, label=model)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Per-category radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(fig, out)


def failure_taxonomy_png(per_model: dict[str, dict[str, int]], out: Path) -> None:
    models = list(per_model.keys())
    kinds = sorted({k for v in per_model.values() for k in v.keys()})
    bottoms = np.zeros(len(models))
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in kinds:
        vals = np.array([per_model[m].get(k, 0) for m in models])
        ax.bar(models, vals, bottom=bottoms, label=k)
        bottoms += vals
    ax.set_ylabel("Failure count")
    ax.set_title("Failure taxonomy per model")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out)


def perf_vs_quality(model_to_perf_quality: dict[str, tuple[float, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, (tps, q) in model_to_perf_quality.items():
        ax.scatter(tps, q, s=80)
        ax.annotate(m, (tps, q), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Generation tokens/s (median)")
    ax.set_ylabel("Overall pass-rate")
    ax.set_ylim(0, 1)
    ax.set_title("Perf vs Quality — Pareto frontier")
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def pass_rate_vs_budget(scenarios_to_curves: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sid, curve in scenarios_to_curves.items():
        xs = [b for b, _ in curve]
        ys = [p for _, p in curve]
        ax.plot(xs, ys, marker="o", label=sid)
    ax.set_xlabel("max_tool_calls budget")
    ax.set_ylabel("Pass-rate")
    ax.set_ylim(0, 1)
    ax.set_title("Pass-rate sensitivity to budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out)
