"""Write tests/fixtures/parity.json: inputs and Python outputs for the JavaScript tests.

``tests/test_parity_fixtures.py`` checks that Python still produces these outputs and
``tests/js/parity.test.mjs`` checks that the website's JavaScript produces them too.
Re-run this script after retraining the digit model.

Usage:
    python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scratchnet.export import (
    decode_array,
    encode_array,
    export_network,
    flat_parameters,
    network_from_export,
    network_from_flat,
    write_json,
)
from scratchnet.layers import Dense, ReLU, Sigmoid, Tanh
from scratchnet.network import Sequential, mlp
from scratchnet.preprocessing import preprocess_drawing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "parity.json"


def draw_line(
    img: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], width: float
) -> None:
    """Rasterise a thick line with soft edges (max-combined into ``img``)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    p0a, p1a = np.array(p0, float), np.array(p1, float)
    d = p1a - p0a
    t = np.clip(((yy - p0a[0]) * d[0] + (xx - p0a[1]) * d[1]) / max(d @ d, 1e-12), 0, 1)
    dist = np.hypot(yy - (p0a[0] + t * d[0]), xx - (p0a[1] + t * d[1]))
    np.maximum(img, np.clip(width / 2 + 0.5 - dist, 0, 1), out=img)


def preprocessing_cases() -> list[dict[str, Any]]:
    seven = np.zeros((120, 90))
    draw_line(seven, (20, 15), (20, 75), 9)
    draw_line(seven, (20, 75), (105, 35), 9)

    dot = np.zeros((40, 40))
    dot[10:13, 20:22] = 1.0

    bar = np.zeros((60, 140))
    draw_line(bar, (30, 15), (32, 120), 6)

    heavy_end = np.zeros((100, 100))  # mass at one end pushes the box past the border
    draw_line(heavy_end, (10, 50), (90, 50), 3)
    heavy_end[75:95, 35:65] = 1.0

    cases = {
        "seven": seven,
        "dot": dot,
        "bar": bar,
        "heavy_end": heavy_end,
        "empty": np.zeros((20, 20)),
    }
    return [
        {
            "name": name,
            "image": encode_array(img.astype(np.float32)),
            "expected": encode_array(preprocess_drawing(img.astype(np.float32))),
        }
        for name, img in cases.items()
    ]


def main() -> None:
    rng = np.random.default_rng(42)

    mixed = Sequential(
        [
            Dense(6, 5, rng),
            ReLU(),
            Dense(5, 4, rng),
            Tanh(),
            Dense(4, 3, rng),
            Sigmoid(),
            Dense(3, 2, rng),
        ]
    )
    for layer in mixed.layers:
        if isinstance(layer, Dense):
            layer.params["b"] = rng.normal(0, 0.5, layer.params["b"].shape)
    exported = export_network(mixed)
    reloaded = network_from_export(exported)  # float32-rounded weights, as in the browser
    x_mixed = rng.normal(0, 1, (4, 6)).astype(np.float32)

    small = mlp([2, 8, 8, 1], activation="tanh", seed=7)
    flat = flat_parameters(small)
    x_flat = rng.uniform(-1.1, 1.1, (6, 2)).round(3)

    digits = json.loads((ROOT / "web" / "data" / "digits.json").read_text())
    digit_net = network_from_export(digits)
    x_digits = (rng.uniform(0, 1, (3, 784)) * (rng.uniform(0, 1, (3, 784)) > 0.8)).astype(
        np.float32
    )

    fixtures = {
        "mixed_network": {
            "model": exported,
            "inputs": x_mixed.tolist(),
            "expected": reloaded.forward(x_mixed.astype(float)).tolist(),
        },
        "flat_network": {
            "sizes": [2, 8, 8, 1],
            "activation": "tanh",
            "params": flat,
            "inputs": x_flat.tolist(),
            "expected": network_from_flat([2, 8, 8, 1], "tanh", flat).forward(x_flat).tolist(),
        },
        "digits_model": {
            "inputs": [encode_array(x) for x in x_digits],
            "expected": digit_net.forward(x_digits.astype(float)).tolist(),
        },
        "preprocessing": preprocessing_cases(),
    }
    # sanity check before writing
    assert decode_array(fixtures["preprocessing"][0]["expected"]).shape == (28, 28)
    size = write_json(fixtures, OUT)
    print(f"wrote {OUT} ({size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
