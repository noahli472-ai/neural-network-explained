"""Python side of the Python/JavaScript parity check.

tests/fixtures/parity.json holds inputs and expected outputs. This test makes sure the
Python code still produces them; tests/js/parity.test.mjs checks the website's JavaScript
against the same file. If this fails after an intended change, run
``python scripts/make_fixtures.py``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scratchnet.export import decode_array, network_from_export, network_from_flat
from scratchnet.preprocessing import preprocess_drawing

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixtures():
    return json.loads((ROOT / "tests" / "fixtures" / "parity.json").read_text())


def test_mixed_network(fixtures):
    case = fixtures["mixed_network"]
    out = network_from_export(case["model"]).forward(np.array(case["inputs"]))
    np.testing.assert_allclose(out, case["expected"], rtol=1e-10, atol=1e-12)


def test_flat_network(fixtures):
    case = fixtures["flat_network"]
    net = network_from_flat(case["sizes"], case["activation"], case["params"])
    np.testing.assert_allclose(net.forward(np.array(case["inputs"])), case["expected"], rtol=1e-10)


def test_digits_model_matches_shipped_file(fixtures):
    model = json.loads((ROOT / "web" / "data" / "digits.json").read_text())
    x = np.stack([decode_array(e) for e in fixtures["digits_model"]["inputs"]]).astype(float)
    out = network_from_export(model).forward(x)
    np.testing.assert_allclose(
        out,
        fixtures["digits_model"]["expected"],
        rtol=1e-6,
        atol=1e-6,
        err_msg="web/data/digits.json changed: run python scripts/make_fixtures.py",
    )


def test_preprocessing_cases(fixtures):
    names = []
    for case in fixtures["preprocessing"]:
        out = preprocess_drawing(decode_array(case["image"]))
        np.testing.assert_allclose(
            out, decode_array(case["expected"]), atol=1e-6, err_msg=case["name"]
        )
        names.append(case["name"])
    assert {"seven", "dot", "bar", "heavy_end", "empty"} <= set(names)
