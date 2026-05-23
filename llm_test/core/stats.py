from __future__ import annotations

import math
import random
from collections.abc import Iterable


def bootstrap_ci(samples: list[float], iterations: int = 1000,
                 ci: float = 0.95, seed: int | None = None) -> tuple[float, float, float]:
    if not samples:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(samples)
    mean = sum(samples) / n
    means = []
    for _ in range(iterations):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2
    lo = means[int(math.floor(alpha * iterations))]
    hi = means[int(math.ceil((1 - alpha) * iterations)) - 1]
    return (mean, lo, hi)


def mcnemar_p(a: list[float], b: list[float]) -> float:
    """McNemar's test (continuity-corrected) for paired binary outcomes.
    Values > 0.5 treated as 'pass'. Returns two-sided p-value via chi-square(1).
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    b01 = sum(1 for x, y in zip(a, b, strict=True) if x <= 0.5 and y > 0.5)
    b10 = sum(1 for x, y in zip(a, b, strict=True) if x > 0.5 and y <= 0.5)
    n = b01 + b10
    if n == 0:
        return 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0.0
    return math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0


def decay_weighted_mean(values_with_ages_days: Iterable[tuple[float, float]],
                        half_life_days: float) -> float:
    num, den = 0.0, 0.0
    lam = math.log(2) / half_life_days
    for v, age in values_with_ages_days:
        w = math.exp(-lam * age)
        num += w * v
        den += w
    return num / den if den > 0 else 0.0
