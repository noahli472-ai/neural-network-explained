"""Export format, and the data files the website ships with."""

import json
from pathlib import Path

import numpy as np
import pytest

from scratchnet.digits import ARCHITECTURE, to_inputs
from scratchnet.export import (
    decode_array,
    encode_array,
    export_network,
    flat_parameters,
    network_from_export,
    network_from_flat,
)
from scratchnet.mnist import load_mnist, mnist_available
from scratchnet.network import mlp
from scratchnet.playground import ARCHITECTURES, snapshot_steps
from scratchnet.training import accuracy

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "data"


def test_encode_decode_roundtrip():
    a = np.random.default_rng(0).normal(size=(3, 5))
    b = decode_array(encode_array(a))
    assert b.shape == (3, 5) and b.dtype == np.float32
    np.testing.assert_allclose(b, a, rtol=1e-7, atol=1e-7)


def test_exported_network_gives_same_outputs():
    net = mlp([5, 4, 3], activation="relu", seed=2)
    x = np.random.default_rng(1).normal(size=(10, 5))
    again = network_from_export(json.loads(json.dumps(export_network(net))))
    np.testing.assert_allclose(again.forward(x), net.forward(x), rtol=1e-5, atol=1e-5)


def test_flat_parameters_roundtrip_and_order():
    net = mlp([2, 3, 1], activation="tanh", seed=0)
    flat = flat_parameters(net, decimals=12)
    w1 = net.layers[0].params["W"]
    assert flat[:6] == pytest.approx(w1.reshape(-1).tolist())  # W1 row-major comes first
    again = network_from_flat([2, 3, 1], "tanh", flat)
    x = np.random.default_rng(0).normal(size=(4, 2))
    np.testing.assert_allclose(again.forward(x), net.forward(x))
    with pytest.raises(ValueError):
        network_from_flat([2, 3, 1], "tanh", flat[:-1])


def test_snapshot_steps():
    steps = snapshot_steps(3000, 60)
    assert steps[0] == 0 and steps[-1] == 3000
    assert steps == sorted(set(steps))
    assert steps[1:6] == [1, 2, 3, 4, 5]  # dense at the start, where learning is fastest


@pytest.fixture(scope="module")
def playground():
    return json.loads((WEB_DATA / "playground.json").read_text())


def test_playground_frames_are_consistent(playground):
    for key, run in playground["runs"].items():
        dataset, arch = key.split("/")
        frames = run["frames"]
        steps = [f["step"] for f in frames]
        assert steps == sorted(steps) and steps[0] == 0
        assert frames[-1]["loss"] < frames[0]["loss"], key
        # the rounded parameters stored for the website reproduce the recorded accuracy
        data = playground["datasets"][dataset]
        net = network_from_flat(ARCHITECTURES[arch]["sizes"], "tanh", frames[-1]["params"])
        acc = accuracy(net, np.array(data["x"]), np.array(data["y"]))
        assert acc == pytest.approx(frames[-1]["accuracy"], abs=0.02), key


def test_playground_tells_the_story_the_website_tells(playground):
    final = {k: r["frames"][-1]["accuracy"] for k, r in playground["runs"].items()}
    for dataset in playground["datasets"]:
        assert final[f"{dataset}/neuron"] < 0.95, "one neuron should fail on every curved pattern"
        assert final[f"{dataset}/large"] == 1.0, "8 + 8 neurons should solve every pattern"
    assert final["spiral/small"] < 0.9, "4 neurons should not be enough for the spiral"
    for dataset in ("circles", "xor", "moons"):
        assert final[f"{dataset}/small"] == 1.0


def test_neuron_station_learns_to_separate(playground):
    frames = playground["neuron"]["frames"]
    assert frames[0]["accuracy"] < 0.7 and frames[-1]["accuracy"] == 1.0
    # final weights stay inside the slider range on the page (|w| <= 10, |b| <= 5)
    w1, w2, b = frames[-1]["params"]
    assert max(abs(w1), abs(w2)) <= 10 and abs(b) <= 5


@pytest.fixture(scope="module")
def digits_model():
    return json.loads((WEB_DATA / "digits.json").read_text())


def test_digit_model_file(digits_model):
    net = network_from_export(digits_model)
    sizes = [layer.n_in for layer in net.layers if hasattr(layer, "n_in")] + [net.n_outputs]
    assert sizes == ARCHITECTURE
    assert digits_model["info"]["n_params"] == net.n_params()
    assert digits_model["info"]["test_accuracy"] > 0.95


@pytest.mark.slow
@pytest.mark.skipif(not mnist_available(), reason="MNIST not downloaded")
def test_digit_model_accuracy_on_mnist_test_set(digits_model):
    _, _, x_test, y_test = load_mnist(download=False)
    net = network_from_export(digits_model)
    acc = accuracy(net, to_inputs(x_test), y_test.astype(int))
    assert acc == pytest.approx(digits_model["info"]["test_accuracy"], abs=1e-3)
    assert acc > 0.96
