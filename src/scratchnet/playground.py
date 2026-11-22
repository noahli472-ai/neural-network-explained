"""Record training runs of small 2-D networks for the website.

The website does not train anything itself. It replays snapshots recorded here: for
every dataset/architecture pair the network is trained with full-batch Adam and its
parameters are saved at roughly logarithmically spaced steps (many early, when the
decision boundary changes fast, fewer later).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .datasets import TOY_DATASETS, make_linear
from .export import flat_parameters
from .losses import BinaryCrossEntropy
from .network import Sequential, mlp
from .optim import SGD, Adam, Optimizer
from .training import accuracy

ARCHITECTURES: dict[str, dict[str, Any]] = {
    "neuron": {"sizes": [2, 1], "activation": "tanh", "label": "1 neuron"},
    "small": {"sizes": [2, 4, 1], "activation": "tanh", "label": "4 neurons"},
    "large": {"sizes": [2, 8, 8, 1], "activation": "tanh", "label": "8 + 8 neurons"},
}
DATASET_SIZE = 200
TRAIN_STEPS = 3000
LEARNING_RATE = 0.03
N_FRAMES = 60


def snapshot_steps(total_steps: int, n_frames: int) -> list[int]:
    """Step 0 plus ``n_frames - 1`` roughly geometric steps up to ``total_steps``."""
    geometric = np.unique(np.round(np.geomspace(1, total_steps, n_frames - 1)).astype(int))
    return [0, *geometric.tolist()]


def record_training(
    network: Sequential,
    optimizer: Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    steps: list[int],
) -> list[dict[str, Any]]:
    """Train with full-batch updates and record a frame at each step in ``steps``.

    A frame holds the step number, the loss and accuracy *before* that step's update
    (so frame 0 is the untrained network) and the flattened parameters.
    """
    loss_fn = BinaryCrossEntropy()
    frames: list[dict[str, Any]] = []
    wanted = set(steps)
    for step in range(max(steps) + 1):
        loss = loss_fn.forward(network.forward(x), y)
        if step in wanted:
            frames.append(
                {
                    "step": step,
                    "loss": round(loss, 5),
                    "accuracy": round(accuracy(network, x, y), 4),
                    "params": flat_parameters(network),
                }
            )
        network.backward(loss_fn.backward())
        optimizer.step()
    return frames


def build_neuron_station(seed: int = 0) -> dict[str, Any]:
    """Data and a recorded gradient-descent run for the single-neuron station.

    Starts from deliberately wrong weights so the visitor sees the line move into place.
    """
    x, y = make_linear(n=40, seed=seed)
    net = mlp([2, 1], seed=seed)
    dense = net.layers[0]
    dense.params["W"] = np.array([[-2.0], [-1.5]])
    dense.params["b"] = np.array([1.0])
    frames = record_training(net, SGD(net, lr=3.0), x, y, snapshot_steps(150, 60))
    return {
        "x": np.round(x, 3).tolist(),
        "y": y.tolist(),
        "frames": frames,
    }


def build_playground(seed: int = 0) -> dict[str, Any]:
    """Generate all datasets and record every dataset/architecture run."""
    steps = snapshot_steps(TRAIN_STEPS, N_FRAMES)
    datasets: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    for d_name, make in TOY_DATASETS.items():
        x, y = make(n=DATASET_SIZE, seed=seed)
        datasets[d_name] = {"x": np.round(x, 3).tolist(), "y": y.tolist()}
        for a_name, arch in ARCHITECTURES.items():
            net = mlp(arch["sizes"], activation=arch["activation"], seed=seed)
            frames = record_training(net, Adam(net, lr=LEARNING_RATE), x, y, steps)
            runs[f"{d_name}/{a_name}"] = {"frames": frames}
    return {
        "version": 1,
        "neuron": build_neuron_station(seed),
        "architectures": ARCHITECTURES,
        "datasets": datasets,
        "runs": runs,
    }
