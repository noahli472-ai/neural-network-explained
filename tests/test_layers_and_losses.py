"""Backpropagation of every layer and loss, checked against finite differences."""

import numpy as np
import pytest

from scratchnet.gradcheck import numerical_gradient, relative_error
from scratchnet.layers import Dense, ReLU, Sigmoid, Tanh, sigmoid
from scratchnet.losses import BinaryCrossEntropy, SoftmaxCrossEntropy, log_softmax, softmax

TOL = 1e-6


def scalar_head(out: np.ndarray, rng: np.random.Generator):
    """Random linear functional L = sum(c * out) so that dL/dout = c."""
    return rng.normal(size=out.shape)


def test_dense_gradients():
    rng = np.random.default_rng(0)
    layer = Dense(4, 3, rng)
    layer.params["b"] = rng.normal(size=3)
    x = rng.normal(size=(5, 4))
    c = scalar_head(layer.forward(x), rng)

    def loss():
        return float(np.sum(c * layer.forward(x)))

    loss()
    dx = layer.backward(c)
    assert relative_error(dx, numerical_gradient(loss, x)) < TOL
    for key in ("W", "b"):
        assert relative_error(layer.grads[key], numerical_gradient(loss, layer.params[key])) < TOL


@pytest.mark.parametrize("act", [ReLU, Tanh, Sigmoid])
def test_activation_gradients(act):
    rng = np.random.default_rng(1)
    layer = act()
    # keep inputs away from the ReLU kink at 0, where the derivative is undefined
    x = rng.uniform(0.1, 2.0, size=(6, 5)) * rng.choice([-1.0, 1.0], size=(6, 5))
    c = scalar_head(layer.forward(x), rng)

    def loss():
        return float(np.sum(c * layer.forward(x)))

    loss()
    assert relative_error(layer.backward(c), numerical_gradient(loss, x)) < TOL


def test_activation_values():
    x = np.array([[-2.0, 0.0, 3.0]])
    np.testing.assert_allclose(ReLU().forward(x), [[0.0, 0.0, 3.0]])
    np.testing.assert_allclose(Tanh().forward(x), np.tanh(x))
    np.testing.assert_allclose(Sigmoid().forward(x), 1 / (1 + np.exp(-x)))


def test_sigmoid_is_stable_for_extreme_inputs():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        y = sigmoid(np.array([-1e4, 1e4]))
    np.testing.assert_allclose(y, [0.0, 1.0])


def test_he_initialisation_has_expected_spread():
    layer = Dense(500, 400, np.random.default_rng(0), init="he")
    assert np.std(layer.params["W"]) == pytest.approx(np.sqrt(2 / 500), rel=0.02)
    assert np.all(layer.params["b"] == 0)


def test_softmax_properties():
    z = np.random.default_rng(2).normal(size=(4, 7))
    p = softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(softmax(z + 100.0), p)  # invariant to a constant shift
    big = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.all(np.isfinite(big)) and big[0, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(log_softmax(z), np.log(p))


def test_cross_entropy_known_values():
    loss = SoftmaxCrossEntropy()
    # uniform prediction over K classes costs log K
    assert loss.forward(np.zeros((3, 4)), np.array([0, 1, 3])) == pytest.approx(np.log(4))
    # gradient is (p - onehot) / N
    np.testing.assert_allclose(loss.backward()[0], np.array([-0.75, 0.25, 0.25, 0.25]) / 3)


def test_cross_entropy_gradient():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(5, 4))
    y = rng.integers(0, 4, size=5)
    loss = SoftmaxCrossEntropy()

    def f():
        return loss.forward(z, y)

    f()
    assert relative_error(loss.backward(), numerical_gradient(f, z)) < TOL


def test_binary_cross_entropy_matches_textbook_formula():
    z = np.array([[-2.0], [0.3], [1.5]])
    y = np.array([0, 1, 1])
    p = 1 / (1 + np.exp(-z[:, 0]))
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert BinaryCrossEntropy().forward(z, y) == pytest.approx(expected)


def test_binary_cross_entropy_gradient_and_stability():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(6, 1))
    y = rng.integers(0, 2, size=6)
    loss = BinaryCrossEntropy()

    def f():
        return loss.forward(z, y)

    f()
    assert relative_error(loss.backward(), numerical_gradient(f, z)) < TOL
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = BinaryCrossEntropy().forward(np.array([[800.0], [-800.0]]), np.array([0, 1]))
    assert value == pytest.approx(800.0)
