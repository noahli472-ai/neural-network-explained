"""A sequential network: layers applied one after another, backpropagated in reverse."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .layers import ACTIVATIONS, Dense, Layer


class Sequential:
    """Chain of layers. ``forward`` runs them in order, ``backward`` in reverse (chain rule)."""

    def __init__(self, layers: Iterable[Layer]) -> None:
        """Create a network from a list of layers."""
        self.layers: list[Layer] = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the network output (logits) for a batch ``x``."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_all(self, x: np.ndarray) -> list[np.ndarray]:
        """Return the input followed by the output of every layer (for visualisation)."""
        outputs = [x]
        for layer in self.layers:
            outputs.append(layer.forward(outputs[-1]))
        return outputs

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate dL/d(output) through all layers and return dL/d(input)."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[tuple[Layer, str]]:
        """Return ``(layer, name)`` for every trainable parameter."""
        return [(layer, name) for layer in self.layers for name in layer.params]

    def n_params(self) -> int:
        """Total number of trainable scalars."""
        return sum(layer.params[name].size for layer, name in self.parameters())

    @property
    def n_outputs(self) -> int:
        """Size of the output layer."""
        dense = [layer for layer in self.layers if isinstance(layer, Dense)]
        return dense[-1].n_out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class labels: argmax for K > 1 outputs, ``logit > 0`` for one output."""
        logits = self.forward(x)
        if logits.shape[1] == 1:
            return (logits[:, 0] > 0).astype(int)
        return logits.argmax(axis=1)


def mlp(
    sizes: Sequence[int],
    activation: str = "relu",
    seed: int | None = 0,
) -> Sequential:
    """Build a multilayer perceptron.

    Args:
        sizes: Layer widths including input and output, e.g. ``[784, 64, 32, 10]``.
        activation: Hidden-layer non-linearity: ``"relu"``, ``"tanh"`` or ``"sigmoid"``.
            The output layer has no activation; the loss applies softmax/sigmoid.
        seed: Seed for the weight initialisation.

    Returns:
        The network. ``mlp([2, 1])`` is a single neuron (logistic regression).
    """
    if len(sizes) < 2:
        raise ValueError("need at least an input and an output size")
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}")
    rng = np.random.default_rng(seed)
    init = "he" if activation == "relu" else "glorot"
    layers: list[Layer] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(Dense(n_in, n_out, rng=rng, init=init))
        if i < len(sizes) - 2:
            layers.append(ACTIVATIONS[activation]())
    return Sequential(layers)
