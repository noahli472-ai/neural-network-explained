"""Toy datasets, MNIST reading and the drawing preprocessing."""

import gzip

import numpy as np
import pytest

from scratchnet.datasets import TOY_DATASETS, make_linear, make_xor
from scratchnet.mnist import load_mnist, mnist_available, read_idx
from scratchnet.preprocessing import (
    BOX_SIZE,
    area_weights,
    bounding_box,
    center_of_mass,
    preprocess_drawing,
    resize_area,
    round_half_up,
)


@pytest.mark.parametrize("name", list(TOY_DATASETS))
def test_toy_datasets_shape_balance_and_determinism(name):
    make = TOY_DATASETS[name]
    x, y = make(n=200, seed=3)
    assert x.shape == (200, 2) and y.shape == (200,)
    assert set(np.unique(y)) == {0, 1}
    assert 0.4 < y.mean() < 0.6
    assert np.abs(x).max() < 1.25
    x2, y2 = make(n=200, seed=3)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)


def test_linear_dataset_is_separable_with_margin():
    x, y = make_linear(n=40, margin=0.12, seed=0)
    signed = x @ np.array([0.8, -0.6]) + 0.1
    assert np.all((signed > 0) == (y == 1))
    assert np.abs(signed).min() > 0.12


def test_xor_labels_follow_sign_rule():
    x, y = make_xor(n=100, seed=0)
    np.testing.assert_array_equal(y, (np.sign(x[:, 0]) == np.sign(x[:, 1])).astype(int))


def write_idx(path, array):
    header = bytes([0, 0, 0x08, array.ndim]) + b"".join(d.to_bytes(4, "big") for d in array.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + array.astype(np.uint8).tobytes())


def test_read_idx_roundtrip(tmp_path):
    array = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    write_idx(tmp_path / "a.gz", array)
    np.testing.assert_array_equal(read_idx(tmp_path / "a.gz"), array)


def test_read_idx_rejects_wrong_type(tmp_path):
    with gzip.open(tmp_path / "bad.gz", "wb") as f:
        f.write(bytes([0, 0, 0x0D, 1]) + (4).to_bytes(4, "big") + b"\x00" * 16)
    with pytest.raises(ValueError):
        read_idx(tmp_path / "bad.gz")


@pytest.mark.slow
@pytest.mark.skipif(not mnist_available(), reason="MNIST not downloaded")
def test_mnist_has_official_split():
    x_train, y_train, x_test, y_test = load_mnist(download=False)
    assert x_train.shape == (60000, 28, 28) and x_test.shape == (10000, 28, 28)
    assert np.bincount(y_test).tolist() == [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]


def test_round_half_up_matches_javascript():
    assert [round_half_up(v) for v in (0.5, 1.5, 2.5, -0.5, -1.5, 2.4999)] == [1, 2, 3, 0, -1, 2]


@pytest.mark.parametrize(("n_in", "n_out"), [(100, 20), (37, 20), (20, 20), (3, 20), (7, 3)])
def test_area_weights_are_a_proper_average(n_in, n_out):
    w = area_weights(n_in, n_out)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)  # each output is a weighted mean
    np.testing.assert_allclose(w.sum(axis=0), n_out / n_in)  # each input counts equally


def test_resize_area_preserves_mean_intensity():
    img = np.random.default_rng(0).uniform(size=(57, 31))
    small = resize_area(img, 20, 11)
    assert small.mean() == pytest.approx(img.mean())


def stroke_image(shift=(0, 0), scale=1.0, size=200):
    """A synthetic '4'-like drawing used to test invariances."""
    img = np.zeros((size, size))
    s = scale
    oy, ox = shift
    img[int(oy + 20 * s) : int(oy + 100 * s), int(ox + 30 * s) : int(ox + 40 * s)] = 1.0
    img[int(oy + 90 * s) : int(oy + 100 * s), int(ox + 30 * s) : int(ox + 90 * s)] = 1.0
    img[int(oy + 20 * s) : int(oy + 150 * s), int(ox + 70 * s) : int(ox + 80 * s)] = 1.0
    return img


def test_preprocessing_mimics_mnist_normalisation():
    out = preprocess_drawing(stroke_image())
    assert out.shape == (28, 28) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    top, bottom, left, right = bounding_box(out)
    assert max(bottom - top, right - left) == BOX_SIZE
    cy, cx = center_of_mass(out)
    assert abs(cy - 13.5) <= 0.5 and abs(cx - 13.5) <= 0.5


def test_preprocessing_ignores_position_and_size():
    reference = preprocess_drawing(stroke_image())
    np.testing.assert_array_equal(preprocess_drawing(stroke_image(shift=(30, -20))), reference)
    larger = preprocess_drawing(stroke_image(scale=1.25, size=260))
    assert np.corrcoef(larger.ravel(), reference.ravel())[0, 1] > 0.95


def test_preprocessing_empty_and_tiny_inputs():
    assert not preprocess_drawing(np.zeros((50, 50))).any()
    dot = np.zeros((30, 30))
    dot[5, 7] = 1.0
    out = preprocess_drawing(dot)
    assert out.max() == 1.0 and bounding_box(out) is not None
