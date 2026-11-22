"""Gradient-based optimisers that update a network's parameters in place."""

from __future__ import annotations

import numpy as np

from .network import Sequential


class Optimizer:
    """Base class: holds the network and the learning rate."""

    def __init__(self, network: Sequential, lr: float) -> None:
        """Attach the optimiser to ``network`` with learning rate ``lr``."""
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.network = network
        self.lr = lr

    def step(self) -> None:
        """Apply one update using the gradients stored in the layers."""
        raise NotImplementedError


class SGD(Optimizer):
    r"""Stochastic gradient descent with optional momentum.

    .. math::
        v \leftarrow \mu v - \eta \nabla L, \qquad \theta \leftarrow \theta + v

    With :math:`\mu = 0` this is plain gradient descent
    :math:`\theta \leftarrow \theta - \eta \nabla L`.
    """

    def __init__(self, network: Sequential, lr: float = 0.1, momentum: float = 0.0) -> None:
        """Create the optimiser; ``momentum`` is the decay factor mu in [0, 1)."""
        super().__init__(network, lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self._velocity = [np.zeros_like(layer.params[k]) for layer, k in network.parameters()]

    def step(self) -> None:
        """Update every parameter."""
        for v, (layer, key) in zip(self._velocity, self.network.parameters(), strict=True):
            v *= self.momentum
            v -= self.lr * layer.grads[key]
            layer.params[key] += v


class Adam(Optimizer):
    r"""Adam (Kingma & Ba, 2015) with optional decoupled weight decay (AdamW).

    .. math::
        m \leftarrow \beta_1 m + (1-\beta_1) g, \quad
        v \leftarrow \beta_2 v + (1-\beta_2) g^2, \quad
        \theta \leftarrow \theta - \eta \Big(\frac{\hat m}{\sqrt{\hat v} + \epsilon}
        + \lambda \theta\Big)

    with bias-corrected :math:`\hat m = m / (1-\beta_1^t)`, :math:`\hat v = v / (1-\beta_2^t)`.
    Weight decay :math:`\lambda` is applied to weight matrices only, not to biases.
    """

    def __init__(
        self,
        network: Sequential,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Create the optimiser with the usual default hyper-parameters."""
        super().__init__(network, lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        params = network.parameters()
        self._m = [np.zeros_like(layer.params[k]) for layer, k in params]
        self._v = [np.zeros_like(layer.params[k]) for layer, k in params]
        self.t = 0

    def step(self) -> None:
        """Update every parameter."""
        self.t += 1
        c1 = 1.0 - self.beta1**self.t
        c2 = 1.0 - self.beta2**self.t
        for m, v, (layer, key) in zip(self._m, self._v, self.network.parameters(), strict=True):
            g = layer.grads[key]
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            update = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if self.weight_decay and key == "W":
                update = update + self.weight_decay * layer.params[key]
            layer.params[key] -= self.lr * update
