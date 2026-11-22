"""MNIST-style normalisation of a hand-drawn digit.

MNIST digits were size-normalised to fit a 20x20 box (keeping the aspect ratio) and
then placed in a 28x28 image so that their centre of mass is in the middle. A digit
drawn in the browser must go through the same steps, otherwise the network sees
something it was never trained on.

The website implements the identical algorithm in ``web/js/preprocess.js``;
``tests/js/parity.test.mjs`` checks that both give the same result.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

IMAGE_SIZE = 28
BOX_SIZE = 20
INK_THRESHOLD = 0.05


def round_half_up(x: float) -> int:
    """Round to the nearest integer, halves upwards (same as JavaScript ``Math.round``)."""
    return int(np.floor(x + 0.5))


@lru_cache(maxsize=4096)
def area_weights(n_in: int, n_out: int) -> np.ndarray:
    """Weights of a 1-D area-averaging resampler, shape (n_out, n_in).

    Output pixel ``o`` covers the source interval ``[o*s, (o+1)*s)`` with ``s = n_in / n_out``;
    its value is the average of the source pixels, weighted by their overlap. Every row sums
    to one. Works for down- and up-sampling.
    """
    scale = n_in / n_out
    w = np.zeros((n_out, n_in))
    for o in range(n_out):
        start, end = o * scale, (o + 1) * scale
        for k in range(int(np.floor(start)), min(int(np.ceil(end)), n_in)):
            overlap = min(end, k + 1) - max(start, k)
            if overlap > 0:
                w[o, k] = overlap / scale
    w.flags.writeable = False
    return w


def resize_area(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2-D image by area averaging (separable: rows, then columns)."""
    h, w = image.shape
    return area_weights(h, out_h) @ image @ area_weights(w, out_w).T


def bounding_box(
    image: np.ndarray, threshold: float = INK_THRESHOLD
) -> tuple[int, int, int, int] | None:
    """Return ``(top, bottom, left, right)`` (bottom/right exclusive) of pixels above threshold."""
    rows = np.flatnonzero((image > threshold).any(axis=1))
    cols = np.flatnonzero((image > threshold).any(axis=0))
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def center_of_mass(image: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted mean row and column index."""
    total = image.sum()
    cy = float(image.sum(axis=1) @ np.arange(image.shape[0]) / total)
    cx = float(image.sum(axis=0) @ np.arange(image.shape[1]) / total)
    return cy, cx


def preprocess_drawing(image: np.ndarray, threshold: float = INK_THRESHOLD) -> np.ndarray:
    """Turn a grey-scale drawing into a 28x28 MNIST-like image.

    Steps:
        1. Crop to the bounding box of all pixels brighter than ``threshold``.
        2. Scale (area averaging) so the longer side is 20 px, keeping the aspect ratio.
        3. Paste into a 28x28 image so that the centre of mass lands at (13.5, 13.5).

    Args:
        image: 2-D array with values in [0, 1], 0 = background, 1 = ink. Any size.
        threshold: Minimum intensity that counts as ink for the bounding box.

    Returns:
        float32 array of shape (28, 28) in [0, 1]. All zeros if the image is empty.
    """
    image = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    box = bounding_box(image, threshold)
    if box is None:
        return out.astype(np.float32)
    top, bottom, left, right = box
    crop = image[top:bottom, left:right]
    h, w = crop.shape
    scale = BOX_SIZE / max(h, w)
    new_h = max(1, round_half_up(h * scale))
    new_w = max(1, round_half_up(w * scale))
    small = resize_area(crop, new_h, new_w)
    cy, cx = center_of_mass(small)
    oy = round_half_up((IMAGE_SIZE - 1) / 2 - cy)
    ox = round_half_up((IMAGE_SIZE - 1) / 2 - cx)
    # paste with clipping at the borders
    y0, y1 = max(0, oy), min(IMAGE_SIZE, oy + new_h)
    x0, x1 = max(0, ox), min(IMAGE_SIZE, ox + new_w)
    out[y0:y1, x0:x1] = small[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
