"""Data augmentation for MNIST so the model copes with digits drawn on a phone.

Random rotation, shear, aspect change and stroke thickness, followed by the same
normalisation the website applies (:func:`~scratchnet.preprocessing.preprocess_drawing`).
Requires SciPy (``pip install -e ".[train]"``).
"""

from __future__ import annotations

import numpy as np

from .preprocessing import preprocess_drawing


def random_affine(
    image: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 15.0,
    max_shear: float = 0.25,
    aspect_range: tuple[float, float] = (0.75, 1.25),
) -> np.ndarray:
    """Rotate, shear and stretch a 2-D image about its centre (bilinear interpolation)."""
    from scipy.ndimage import affine_transform

    theta = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    shear = rng.uniform(-max_shear, max_shear)
    aspect = rng.uniform(*aspect_range)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shear_m = np.array([[1.0, 0.0], [shear, 1.0]])
    stretch = np.diag([aspect, 1.0 / aspect])
    forward = rotation @ shear_m @ stretch
    inverse = np.linalg.inv(forward)
    center = (np.array(image.shape) - 1) / 2
    offset = center - inverse @ center
    return affine_transform(image, inverse, offset=offset, order=1, mode="constant", cval=0.0)


def random_thickness(
    image: np.ndarray, rng: np.random.Generator, p_thicker: float = 0.35, p_thinner: float = 0.1
) -> np.ndarray:
    """With some probability make strokes thicker (dilation) or thinner (erosion)."""
    from scipy.ndimage import grey_dilation, grey_erosion

    u = rng.uniform()
    if u < p_thicker:
        return grey_dilation(image, size=(2, 2))
    if u < p_thicker + p_thinner:
        eroded = grey_erosion(image, size=(2, 2))
        return eroded if eroded.sum() > 0.3 * image.sum() else image
    return image


def augment_digits(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augment a batch of flattened 28x28 images in [0, 1]; returns shape (N, 784) float32."""
    out = np.empty((len(images), 28 * 28), dtype=np.float32)
    for i, flat in enumerate(images):
        img = flat.reshape(28, 28).astype(float)
        img = random_thickness(random_affine(img, rng), rng)
        peak = img.max()
        if peak > 0:
            img = img / peak  # interpolation softens strokes; drawings always reach full ink
        out[i] = preprocess_drawing(img).reshape(-1)
    return out
