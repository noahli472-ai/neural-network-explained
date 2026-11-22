"""Whole-network backprop, optimisers and the training loop."""

import numpy as np
import pytest

from scratchnet import SGD, Adam, BinaryCrossEntropy, SoftmaxCrossEntropy, accuracy, fit, mlp
from scratchnet.datasets import make_spiral
from scratchnet.gradcheck import numerical_gradient, relative_error
from scratchnet.layers import Dense, ReLU
from scratchnet.network import Sequential
from scratchnet.training import iterate_minibatches

XOR_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = np.array([0, 1, 1, 0])


@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid"])
def test_network_gradients_match_finite_differences(activation):
    rng = np.random.default_rng(0)
    net = mlp([3, 5, 4, 3], activation=activation, seed=1)
    for layer, key in net.parameters():
        layer.params[key] = layer.params[key] + rng.normal(0, 0.1, layer.params[key].shape)
    x = rng.normal(size=(7, 3))
    y = rng.integers(0, 3, size=7)
    loss = SoftmaxCrossEntropy()

    def f():
        return loss.forward(net.forward(x), y)

    f()
    net.backward(loss.backward())
    for layer, key in net.parameters():
        analytic = layer.grads[key].copy()
        assert relative_error(analytic, numerical_gradient(f, layer.params[key])) < 1e-6, key


def test_mlp_structure():
    net = mlp([4, 6, 3], activation="relu")
    assert [type(layer).__name__ for layer in net.layers] == ["Dense", "ReLU", "Dense"]
    assert net.n_params() == 4 * 6 + 6 + 6 * 3 + 3
    assert net.n_outputs == 3
    with pytest.raises(ValueError):
        mlp([4])
    with pytest.raises(ValueError):
        mlp([4, 2], activation="swish")


def test_forward_all_returns_every_layer_output():
    net = mlp([2, 3, 1], activation="tanh")
    outputs = net.forward_all(np.ones((5, 2)))
    assert [o.shape for o in outputs] == [(5, 2), (5, 3), (5, 3), (5, 1)]
    np.testing.assert_allclose(outputs[-1], net.forward(np.ones((5, 2))))


def train_xor(sizes, steps=2000, seed=0):
    net = mlp(sizes, activation="tanh", seed=seed)
    opt = Adam(net, lr=0.05)
    loss = BinaryCrossEntropy()
    for _ in range(steps):
        loss.forward(net.forward(XOR_X), XOR_Y)
        net.backward(loss.backward())
        opt.step()
    return accuracy(net, XOR_X, XOR_Y)


def test_single_neuron_cannot_learn_xor():
    # XOR is not linearly separable: any line gets at most 3 of the 4 points right
    assert train_xor([2, 1]) <= 0.75


def test_hidden_layer_learns_xor():
    assert train_xor([2, 4, 1]) == 1.0


def test_sgd_matches_update_rule():
    net = Sequential([Dense(2, 1)])
    layer = net.layers[0]
    layer.params["W"] = np.array([[1.0], [2.0]])
    layer.params["b"] = np.array([0.5])
    opt = SGD(net, lr=0.1, momentum=0.9)
    layer.grads = {"W": np.array([[1.0], [-1.0]]), "b": np.array([2.0])}
    opt.step()
    np.testing.assert_allclose(layer.params["W"], [[0.9], [2.1]])
    opt.step()  # velocity = 0.9 * (-0.1 g) - 0.1 g = -0.19 g
    np.testing.assert_allclose(layer.params["W"], [[0.71], [2.29]])
    np.testing.assert_allclose(layer.params["b"], [0.5 - 0.2 - 0.38])


@pytest.mark.parametrize(
    "make_opt", [lambda n: SGD(n, lr=0.1), lambda n: SGD(n, 0.05, 0.9), lambda n: Adam(n, 0.05)]
)
def test_optimisers_minimise_a_quadratic(make_opt):
    # L(w) = 1/2 (w - w*)^T A (w - w*), gradient A (w - w*)
    a = np.array([[3.0, 0.5], [0.5, 1.0]])
    target = np.array([[1.0], [-2.0]])
    net = Sequential([Dense(2, 1, np.random.default_rng(0))])
    layer = net.layers[0]
    opt = make_opt(net)
    for _ in range(3000):
        layer.grads["W"] = a @ (layer.params["W"] - target)
        layer.grads["b"] = layer.params["b"].copy()
        opt.step()
    np.testing.assert_allclose(layer.params["W"], target, atol=1e-3)
    np.testing.assert_allclose(layer.params["b"], 0.0, atol=1e-3)


def test_first_adam_step_has_size_lr():
    # after bias correction the first step is lr * g / |g| for every parameter
    net = Sequential([Dense(3, 2)])
    layer = net.layers[0]
    before = layer.params["W"].copy()
    layer.grads = {"W": np.array([[5.0, -0.01], [2.0, 3.0], [-7.0, 1e-3]]), "b": np.ones(2)}
    Adam(net, lr=0.01).step()
    np.testing.assert_allclose(
        before - layer.params["W"], 0.01 * np.sign(layer.grads["W"]), rtol=1e-3
    )


def test_weight_decay_only_touches_weights():
    net = Sequential([Dense(2, 2)])
    layer = net.layers[0]
    layer.params["W"] = np.ones((2, 2))
    layer.params["b"] = np.ones(2)
    layer.grads = {"W": np.zeros((2, 2)), "b": np.zeros(2)}
    Adam(net, lr=0.1, weight_decay=0.5).step()
    np.testing.assert_allclose(layer.params["W"], 0.95)
    np.testing.assert_allclose(layer.params["b"], 1.0)


def test_minibatches_cover_every_sample_once():
    batches = list(iterate_minibatches(103, 10, np.random.default_rng(0)))
    assert len(batches) == 11
    assert sorted(np.concatenate(batches).tolist()) == list(range(103))


def test_fit_learns_the_spiral():
    x, y = make_spiral(n=200, seed=0)
    net = mlp([2, 16, 16, 1], activation="tanh", seed=0)
    opt = SGD(net, lr=0.1, momentum=0.9)
    history = fit(net, BinaryCrossEntropy(), opt, x, y, epochs=300, batch_size=32)
    assert history[-1]["loss"] < 0.2 * history[0]["loss"]
    assert accuracy(net, x, y) > 0.95


def test_fit_reports_validation_accuracy():
    x, y = make_spiral(n=60, seed=1)
    net = mlp([2, 4, 1], activation="tanh")
    history = fit(net, BinaryCrossEntropy(), SGD(net, lr=0.1), x, y, epochs=2, x_val=x, y_val=y)
    assert [h["epoch"] for h in history] == [1, 2]
    assert 0.0 <= history[-1]["val_accuracy"] <= 1.0


def test_optimiser_rejects_bad_settings():
    net = Sequential([Dense(1, 1), ReLU()])
    with pytest.raises(ValueError):
        SGD(net, lr=0.0)
    with pytest.raises(ValueError):
        SGD(net, lr=0.1, momentum=1.0)
