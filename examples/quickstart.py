"""Train a small network on the spiral and verify its gradients numerically.

Run from the repository root:
    python examples/quickstart.py
"""

from __future__ import annotations

from scratchnet import SGD, BinaryCrossEntropy, accuracy, fit, mlp
from scratchnet.datasets import make_spiral
from scratchnet.gradcheck import numerical_gradient, relative_error


def main() -> None:
    x, y = make_spiral(n=200, seed=0)

    # 1) Backpropagation agrees with finite differences
    net = mlp([2, 8, 8, 1], activation="tanh", seed=0)
    loss = BinaryCrossEntropy()

    def f() -> float:
        return loss.forward(net.forward(x), y)

    f()
    net.backward(loss.backward())
    first = net.layers[0]
    analytic = first.grads["W"].copy()
    err = relative_error(analytic, numerical_gradient(f, first.params["W"]))
    print(f"gradient check, first layer: max relative error {err:.1e}")

    # 2) A single neuron versus a network with two hidden layers
    for sizes in ([2, 1], [2, 16, 16, 1]):
        net = mlp(sizes, activation="tanh", seed=0)
        history = fit(net, loss, SGD(net, lr=0.1, momentum=0.9), x, y, epochs=300, batch_size=32)
        print(
            f"{str(sizes):16s} final loss {history[-1]['loss']:.3f}, "
            f"accuracy {accuracy(net, x, y):.0%}"
        )


if __name__ == "__main__":
    main()
