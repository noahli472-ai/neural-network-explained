"""Layers of a fully connected network, each with a hand-written backward pass.

Every layer caches what it needs during ``forward`` and uses it in ``backward``
to turn the upstream gradient dL/d(output) into dL/d(input). Layers with
trainable parameters also store dL/d(parameter) in ``self.grads``.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class for all layers.

    Attributes:
        params: Trainable parameters by name (empty for activation layers).
        grads: Gradients of the loss with respect to ``params``, same keys and shapes.
        kind: Short identifier used when the network is exported for the website.
    """

    kind = "layer"

    def __init__(self) -> None:
        """Create a layer without parameters."""
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a batch ``x`` of shape (N, ...)."""
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backpropagate ``grad_out`` = dL/d(output) and return dL/d(input)."""
        raise NotImplementedError


class Dense(Layer):
    r"""Affine layer :math:`y = xW + b`.

    Shapes: ``x`` (N, n_in), ``W`` (n_in, n_out), ``b`` (n_out,), ``y`` (N, n_out).

    Backward pass, with :math:`G = \partial L / \partial y`:

    .. math::
        \frac{\partial L}{\partial W} = x^\top G, \quad
        \frac{\partial L}{\partial b} = \sum_n G_n, \quad
        \frac{\partial L}{\partial x} = G W^\top
    """

    kind = "dense"

    def __init__(
        self,
        n_in: int,
        n_out: int,
        rng: np.random.Generator | None = None,
        init: str = "he",
    ) -> None:
        """Create a dense layer with random weights and zero biases.

        Args:
            n_in: Number of inputs.
            n_out: Number of outputs (neurons).
            rng: Random generator for the weight initialisation.
            init: ``"he"`` (std = sqrt(2 / n_in), suited to ReLU) or
                ``"glorot"`` (std = sqrt(2 / (n_in + n_out)), suited to tanh/sigmoid).
        """
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        if init == "he":
            std = np.sqrt(2.0 / n_in)
        elif init == "glorot":
            std = np.sqrt(2.0 / (n_in + n_out))
        else:
            raise ValueError(f"unknown init {init!r}")
        self.params["W"] = rng.normal(0.0, std, size=(n_in, n_out))
        self.params["b"] = np.zeros(n_out)
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._x: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        """Number of inputs."""
        return self.params["W"].shape[0]

    @property
    def n_out(self) -> int:
        """Number of outputs."""
        return self.params["W"].shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``x @ W + b``."""
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Store parameter gradients and return dL/dx."""
        if self._x is None:
            raise RuntimeError("backward called before forward")
        self.grads["W"] = self._x.T @ grad_out
        self.grads["b"] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T


class ReLU(Layer):
    r"""Rectified linear unit :math:`y = \max(0, x)`; derivative is 1 for x > 0, else 0."""

    kind = "relu"

    def __init__(self) -> None:
        """Create a ReLU layer."""
        super().__init__()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``max(0, x)`` element-wise."""
        self._mask = x > 0
        return np.where(self._mask, x, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Pass the gradient through where the input was positive."""
        if self._mask is None:
            raise RuntimeError("backward called before forward")
        return grad_out * self._mask


class Tanh(Layer):
    r"""Hyperbolic tangent; derivative :math:`1 - \tanh^2(x)`."""

    kind = "tanh"

    def __init__(self) -> None:
        """Create a tanh layer."""
        super().__init__()
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``tanh(x)`` element-wise."""
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Multiply the gradient by ``1 - y**2``."""
        if self._y is None:
            raise RuntimeError("backward called before forward")
        return grad_out * (1.0 - self._y**2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""Numerically stable logistic function :math:`\sigma(x) = 1 / (1 + e^{-x})`.

    Uses the identity :math:`\sigma(x) = (1 + \tanh(x/2)) / 2`, which never overflows.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Sigmoid(Layer):
    r"""Logistic sigmoid; derivative :math:`\sigma(x)(1 - \sigma(x))`."""

    kind = "sigmoid"

    def __init__(self) -> None:
        """Create a sigmoid layer."""
        super().__init__()
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``sigmoid(x)`` element-wise."""
        self._y = sigmoid(x)
        return self._y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Multiply the gradient by ``y * (1 - y)``."""
        if self._y is None:
            raise RuntimeError("backward called before forward")
        return grad_out * self._y * (1.0 - self._y)


ACTIVATIONS: dict[str, type[Layer]] = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}
