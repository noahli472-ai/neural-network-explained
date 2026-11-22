"""Classify a synthetic hand-drawn 7 with the digit model the website uses.

Draws two thick strokes into a 200x200 array, runs the same preprocessing as the website,
and prints what the network sees and how sure it is. Run from the repository root:
    python examples/classify_drawing.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scratchnet.export import network_from_export
from scratchnet.losses import softmax
from scratchnet.preprocessing import preprocess_drawing

ROOT = Path(__file__).resolve().parents[1]


def stroke(img: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], width: float) -> None:
    """Draw a thick line from p0 to p1 (row, column) into img."""
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    a, b = np.array(p0), np.array(p1)
    d = b - a
    t = np.clip(((yy - a[0]) * d[0] + (xx - a[1]) * d[1]) / (d @ d), 0, 1)
    dist = np.hypot(yy - (a[0] + t * d[0]), xx - (a[1] + t * d[1]))
    np.maximum(img, np.clip(width / 2 - dist, 0, 1), out=img)


def main() -> None:
    drawing = np.zeros((200, 200))
    stroke(drawing, (40, 50), (40, 150), 16)  # top bar
    stroke(drawing, (40, 150), (170, 85), 16)  # diagonal

    image = preprocess_drawing(drawing)
    print("What the network sees (28 x 28):")
    for row in image:
        print("".join(" .:-=+*#%@"[min(9, int(v * 10))] for v in row))

    model = json.loads((ROOT / "web" / "data" / "digits.json").read_text())
    net = network_from_export(model)
    probs = softmax(net.forward(image.reshape(1, -1).astype(float)))[0]
    for digit in np.argsort(probs)[::-1][:3]:
        print(f"{digit}: {probs[digit]:6.1%}")


if __name__ == "__main__":
    main()
