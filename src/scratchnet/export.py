"""Export trained networks to JSON for the website, and load them back.

Weight matrices are stored as base64-encoded little-endian float32, which is compact and
can be decoded in the browser with ``new Float32Array(...)``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from .layers import ACTIVATIONS, Dense, Layer
from .network import Sequential, mlp


def encode_array(array: np.ndarray) -> dict[str, Any]:
    """Encode an array as ``{"shape": [...], "data": <base64 float32, row-major>}``."""
    data = np.ascontiguousarray(array, dtype="<f4").tobytes()
    return {"shape": list(array.shape), "data": base64.b64encode(data).decode("ascii")}


def decode_array(encoded: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_array`; returns a float32 array."""
    raw = base64.b64decode(encoded["data"])
    return np.frombuffer(raw, dtype="<f4").reshape(encoded["shape"]).astype(np.float32)


def export_network(network: Sequential) -> dict[str, Any]:
    """Describe a network as a list of layers with encoded weights."""
    layers: list[dict[str, Any]] = []
    for layer in network.layers:
        entry: dict[str, Any] = {"type": layer.kind}
        if isinstance(layer, Dense):
            entry["W"] = encode_array(layer.params["W"])
            entry["b"] = encode_array(layer.params["b"])
        layers.append(entry)
    return {"layers": layers}


def network_from_export(exported: dict[str, Any]) -> Sequential:
    """Rebuild a :class:`Sequential` from :func:`export_network` output (float64 weights)."""
    layers: list[Layer] = []
    for entry in exported["layers"]:
        if entry["type"] == "dense":
            w = decode_array(entry["W"]).astype(float)
            dense = Dense(w.shape[0], w.shape[1])
            dense.params["W"] = w
            dense.params["b"] = decode_array(entry["b"]).astype(float)
            layers.append(dense)
        elif entry["type"] in ACTIVATIONS:
            layers.append(ACTIVATIONS[entry["type"]]())
        else:
            raise ValueError(f"unknown layer type {entry['type']!r}")
    return Sequential(layers)


def flat_parameters(network: Sequential, decimals: int = 4) -> list[float]:
    """All parameters as one rounded list: W1 (row-major), b1, W2, b2, ...

    Used for the training snapshots of the small 2-D networks, where JSON numbers are
    more compact than base64 blocks.
    """
    values: list[float] = []
    for layer, key in network.parameters():
        values.extend(np.round(layer.params[key].reshape(-1), decimals).tolist())
    return values


def network_from_flat(sizes: list[int], activation: str, params: list[float]) -> Sequential:
    """Inverse of :func:`flat_parameters` for an MLP with the given layer sizes."""
    net = mlp(sizes, activation=activation, seed=0)
    offset = 0
    for layer, key in net.parameters():
        size = layer.params[key].size
        if offset + size > len(params):
            raise ValueError("too few parameters for these layer sizes")
        layer.params[key] = np.asarray(params[offset : offset + size], dtype=float).reshape(
            layer.params[key].shape
        )
        offset += size
    if offset != len(params):
        raise ValueError(f"expected {offset} parameters, got {len(params)}")
    return net


def write_json(obj: dict[str, Any], path: Path) -> int:
    """Write compact JSON and return the file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))
