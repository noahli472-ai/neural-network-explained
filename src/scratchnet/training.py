"""Mini-batch training loop and evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np

from .network import Sequential
from .optim import Optimizer


class Loss(Protocol):
    """Interface shared by the loss classes in :mod:`scratchnet.losses`."""

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Return the mean loss."""
        ...

    def backward(self) -> np.ndarray:
        """Return dL/dlogits."""
        ...


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Yield index arrays that cover ``range(n)`` once in random order."""
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def accuracy(network: Sequential, x: np.ndarray, y: np.ndarray, batch_size: int = 4096) -> float:
    """Fraction of samples classified correctly."""
    correct = 0
    for start in range(0, len(x), batch_size):
        correct += int(
            (network.predict(x[start : start + batch_size]) == y[start : start + batch_size]).sum()
        )
    return correct / len(x)


def train_step(
    network: Sequential, loss_fn: Loss, optimizer: Optimizer, x: np.ndarray, y: np.ndarray
) -> float:
    """Forward pass, loss, backward pass and one optimiser update on a single batch."""
    loss = loss_fn.forward(network.forward(x), y)
    network.backward(loss_fn.backward())
    optimizer.step()
    return loss


def fit(
    network: Sequential,
    loss_fn: Loss,
    optimizer: Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int = 64,
    seed: int = 0,
    transform: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Train for a number of epochs and return one stats dict per epoch.

    Args:
        network: Network to train (updated in place).
        loss_fn: Loss on the logits, e.g. :class:`~scratchnet.losses.SoftmaxCrossEntropy`.
        optimizer: Optimiser attached to ``network``.
        x: Training inputs (N, D).
        y: Training labels (N,).
        epochs: Number of passes over the training set.
        batch_size: Mini-batch size.
        seed: Seed for shuffling (and for ``transform``).
        transform: Optional data augmentation ``f(x_batch, rng) -> x_batch``.
        x_val: Optional validation inputs.
        y_val: Optional validation labels.
        verbose: Print one line per epoch.

    Returns:
        History with keys ``epoch``, ``loss`` (mean training loss) and, if validation
        data is given, ``val_accuracy``.
    """
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        losses = []
        for idx in iterate_minibatches(len(x), batch_size, rng):
            xb = x[idx] if transform is None else transform(x[idx], rng)
            losses.append(train_step(network, loss_fn, optimizer, xb, y[idx]))
        stats: dict[str, float] = {"epoch": epoch, "loss": float(np.mean(losses))}
        if x_val is not None and y_val is not None:
            stats["val_accuracy"] = accuracy(network, x_val, y_val)
        history.append(stats)
        if verbose:
            print(
                ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()
                )
            )
    return history
