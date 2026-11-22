"""Numerical gradient checking: compare backprop with central finite differences."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r"""Central-difference gradient of the scalar function ``f`` with respect to ``x``.

    .. math::
        \frac{\partial f}{\partial x_i} \approx
        \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}

    The truncation error is :math:`O(\epsilon^2)`. ``x`` is modified in place during the
    evaluation and restored afterwards, so ``f`` must read ``x`` (e.g. a network weight).
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readwrite"]])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    r"""Maximum element-wise relative error :math:`|a - b| / \max(|a| + |b|, 10^{-12})`."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12)))
