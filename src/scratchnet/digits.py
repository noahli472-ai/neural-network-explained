"""Train the handwritten-digit classifier shown on the website."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .losses import SoftmaxCrossEntropy
from .mnist import load_mnist
from .network import Sequential, mlp
from .optim import Adam
from .preprocessing import preprocess_drawing
from .training import accuracy, fit

ARCHITECTURE = [784, 64, 32, 10]


def to_inputs(images: np.ndarray) -> np.ndarray:
    """uint8 images (N, 28, 28) -> float32 inputs (N, 784) in [0, 1]."""
    return (images.reshape(len(images), -1) / 255.0).astype(np.float32)


def renormalize(images: np.ndarray) -> np.ndarray:
    """Run every image through the website's preprocessing (flattened output)."""
    flat = to_inputs(images)
    return np.stack([preprocess_drawing(f.reshape(28, 28)).reshape(-1) for f in flat])


def train_digit_model(
    epochs: int = 16,
    seed: int = 0,
    augment: bool = True,
    lr: float = 2e-3,
    weight_decay: float = 0.05,
    batch_size: int = 128,
    data_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[Sequential, dict[str, Any]]:
    """Train the 784-64-32-10 ReLU network on MNIST.

    The learning rate is halved after 50 % and 75 % of the epochs. With ``augment=True``
    every training batch is randomly rotated, sheared, stretched and thickened, then
    normalised exactly like a drawing on the website.

    Returns:
        The trained network and a dict with the training settings and test accuracies.
    """
    x_train, y_train, x_test, y_test = load_mnist(data_dir)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    xt = to_inputs(x_train)
    xv = to_inputs(x_test)
    transform = None
    if augment:
        from .augment import augment_digits

        transform = augment_digits

    net = mlp(ARCHITECTURE, activation="relu", seed=seed)
    opt = Adam(net, lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        opt.lr = lr * (0.5 ** (int(epoch >= epochs // 2) + int(epoch >= 3 * epochs // 4)))
        stats = fit(
            net,
            SoftmaxCrossEntropy(),
            opt,
            xt,
            y_train,
            epochs=1,
            batch_size=batch_size,
            seed=seed + epoch,
            transform=transform,
            x_val=xv,
            y_val=y_test,
        )[0]
        stats["epoch"] = epoch + 1
        stats["lr"] = opt.lr
        history.append(stats)
        if verbose:
            print(
                f"epoch {epoch + 1:2d}  lr={opt.lr:.5f}  loss={stats['loss']:.4f}  "
                f"test_acc={stats['val_accuracy']:.4f}",
                flush=True,
            )

    info = {
        "architecture": ARCHITECTURE,
        "activation": "relu",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "augmentation": augment,
        "n_params": net.n_params(),
        "test_accuracy": accuracy(net, xv, y_test),
        "test_accuracy_renormalized": accuracy(net, renormalize(x_test), y_test),
        "history": history,
    }
    return net, info
