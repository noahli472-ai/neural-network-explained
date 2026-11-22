"""Synthetic 2-D datasets for binary classification.

All generators return ``(x, y)`` with ``x`` of shape (n, 2), roughly inside [-1, 1]^2,
and integer labels ``y`` in {0, 1}. They are deterministic for a given seed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def make_linear(n: int = 40, margin: float = 0.12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two classes separated by a straight line, with a gap of ``margin`` around it.

    The line is ``0.8 x1 - 0.6 x2 + 0.1 = 0`` (unit normal), so a single neuron can
    classify every point correctly.
    """
    rng = np.random.default_rng(seed)
    normal, offset = np.array([0.8, -0.6]), 0.1
    points: list[np.ndarray] = []
    while len(points) < n:
        p = rng.uniform(-0.95, 0.95, size=2)
        if abs(p @ normal + offset) > margin:
            points.append(p)
    x = np.array(points)
    y = (x @ normal + offset > 0).astype(int)
    return x, y


def make_xor(n: int = 200, gap: float = 0.08, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Four quadrants; label 1 where x1 and x2 have the same sign (not linearly separable)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(gap, 0.95, size=(n, 2)) * rng.choice([-1.0, 1.0], size=(n, 2))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    return x, y


def make_circles(
    n: int = 200, noise: float = 0.06, factor: float = 0.45, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A disc (label 1) inside a ring (label 0); ``factor`` is the inner/outer radius ratio."""
    rng = np.random.default_rng(seed)
    n_inner = n // 2
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = np.where(np.arange(n) < n_inner, 0.85 * factor, 0.85)
    x = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    x += rng.normal(0.0, noise, size=x.shape)
    y = (np.arange(n) < n_inner).astype(int)
    return x, y


def make_moons(n: int = 200, noise: float = 0.07, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half circles."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    t0 = rng.uniform(0.0, np.pi, size=n0)
    t1 = rng.uniform(0.0, np.pi, size=n - n0)
    upper = np.column_stack([np.cos(t0), np.sin(t0)])
    lower = np.column_stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)])
    x = np.vstack([upper, lower])
    x = (x - np.array([0.5, 0.25])) / 1.6
    x += rng.normal(0.0, noise, size=x.shape)
    y = np.concatenate([np.zeros(n0, int), np.ones(n - n0, int)])
    return x, y


def make_spiral(
    n: int = 200, turns: float = 1.5, noise: float = 0.03, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved spiral arms, one per class."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    xs, ys = [], []
    for label, count in ((0, n0), (1, n - n0)):
        r = rng.uniform(0.12, 1.0, size=count)
        theta = 2.0 * np.pi * turns * r + label * np.pi
        arm = 0.92 * np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        xs.append(arm + rng.normal(0.0, noise, size=arm.shape))
        ys.append(np.full(count, label))
    return np.vstack(xs), np.concatenate(ys)


TOY_DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "circles": make_circles,
    "xor": make_xor,
    "moons": make_moons,
    "spiral": make_spiral,
}
