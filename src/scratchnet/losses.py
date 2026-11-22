"""Loss functions that take raw network outputs (logits) and return their gradient.

Combining the output non-linearity with the loss (softmax + cross-entropy,
sigmoid + binary cross-entropy) gives simple, numerically stable gradients.
"""

from __future__ import annotations

import numpy as np

from .layers import sigmoid


def log_softmax(z: np.ndarray) -> np.ndarray:
    r"""Row-wise :math:`\log \mathrm{softmax}(z)`, stable for large logits.

    Subtracting the row maximum does not change the result but prevents overflow:
    :math:`\log p_k = z_k - m - \log \sum_j e^{z_j - m}`.
    """
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax: turns logits of shape (N, K) into probabilities."""
    return np.exp(log_softmax(z))


class SoftmaxCrossEntropy:
    r"""Mean cross-entropy of a softmax classifier.

    .. math::
        L = -\frac{1}{N} \sum_n \log p_{n, y_n}, \qquad
        \frac{\partial L}{\partial z_n} = \frac{1}{N} (p_n - e_{y_n})

    where :math:`p_n = \mathrm{softmax}(z_n)` and :math:`e_{y}` is the one-hot vector of label y.
    """

    def __init__(self) -> None:
        """Create the loss; it caches probabilities between forward and backward."""
        self._p: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Return the mean loss for logits (N, K) and integer labels (N,)."""
        logp = log_softmax(logits)
        self._p = np.exp(logp)
        self._y = y
        return float(-logp[np.arange(len(y)), y].mean())

    def backward(self) -> np.ndarray:
        """Return dL/dlogits of shape (N, K)."""
        if self._p is None or self._y is None:
            raise RuntimeError("backward called before forward")
        n = len(self._y)
        grad = self._p.copy()
        grad[np.arange(n), self._y] -= 1.0
        return grad / n


class BinaryCrossEntropy:
    r"""Mean binary cross-entropy of a single sigmoid output, computed from the logit z.

    .. math::
        L = \frac{1}{N} \sum_n \big[\log(1 + e^{z_n}) - y_n z_n\big], \qquad
        \frac{\partial L}{\partial z_n} = \frac{1}{N} (\sigma(z_n) - y_n)

    This equals :math:`-y\log\sigma(z) - (1-y)\log(1-\sigma(z))`, written so that it
    cannot overflow (``np.logaddexp(0, z)`` is a stable softplus).
    """

    def __init__(self) -> None:
        """Create the loss; it caches the logits between forward and backward."""
        self._z: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Return the mean loss for logits (N, 1) and labels (N,) in {0, 1}."""
        z = logits.reshape(-1)
        self._z = z
        self._y = y.astype(float)
        return float(np.mean(np.logaddexp(0.0, z) - self._y * z))

    def backward(self) -> np.ndarray:
        """Return dL/dlogits of shape (N, 1)."""
        if self._z is None or self._y is None:
            raise RuntimeError("backward called before forward")
        return ((sigmoid(self._z) - self._y) / len(self._y)).reshape(-1, 1)
