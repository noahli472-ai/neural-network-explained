"""Download and read the MNIST handwritten-digit dataset (IDX format).

MNIST is not part of this repository. :func:`download_mnist` fetches the four
original ``.gz`` files from a public mirror and verifies their SHA-256 checksums.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

FILES: dict[str, tuple[str, str]] = {
    "train_images": (
        "train-images-idx3-ubyte.gz",
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    ),
    "train_labels": (
        "train-labels-idx1-ubyte.gz",
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    ),
    "test_images": (
        "t10k-images-idx3-ubyte.gz",
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    ),
    "test_labels": (
        "t10k-labels-idx1-ubyte.gz",
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    ),
}

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
)


def default_data_dir() -> Path:
    """Cache directory: ``$SCRATCHNET_DATA_DIR`` or ``~/.cache/scratchnet/mnist``."""
    env = os.environ.get("SCRATCHNET_DATA_DIR")
    return Path(env) if env else Path.home() / ".cache" / "scratchnet" / "mnist"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mnist_available(root: Path | None = None) -> bool:
    """Return True if all four files are present (checksums are not verified here)."""
    root = root or default_data_dir()
    return all((root / name).is_file() for name, _ in FILES.values())


def download_mnist(root: Path | None = None, timeout: float = 60.0) -> Path:
    """Download missing MNIST files into ``root`` and verify their checksums.

    Returns:
        The data directory.

    Raises:
        RuntimeError: If a file cannot be downloaded from any mirror.
    """
    root = root or default_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.values():
        target = root / name
        if target.is_file() and _sha256(target) == sha:
            continue
        errors = []
        for mirror in MIRRORS:
            tmp = target.with_suffix(".part")
            try:
                with urllib.request.urlopen(mirror + name, timeout=timeout) as response:
                    tmp.write_bytes(response.read())
                if _sha256(tmp) != sha:
                    raise ValueError("checksum mismatch")
                tmp.replace(target)
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                errors.append(f"{mirror}: {exc}")
                tmp.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"could not download {name}:\n" + "\n".join(errors))
    return root


def read_idx(path: Path) -> np.ndarray:
    """Read a gzipped IDX file (the MNIST container format) into a uint8 array."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    zero, dtype_code, ndim = data[0:2], data[2], data[3]
    if zero != b"\x00\x00" or dtype_code != 0x08:
        raise ValueError(f"{path} is not an unsigned-byte IDX file")
    dims = [int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    array = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    if array.size != int(np.prod(dims)):
        raise ValueError(f"{path}: expected {np.prod(dims)} values, found {array.size}")
    return array.reshape(dims)


def load_mnist(
    root: Path | None = None, download: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, y_train, x_test, y_test)``.

    Images are uint8 arrays of shape (N, 28, 28) with 0 = background, 255 = ink;
    labels are uint8 arrays of shape (N,). The split is the official one
    (60 000 training, 10 000 test images).
    """
    root = root or default_data_dir()
    if download:
        download_mnist(root)
    arrays = {key: read_idx(root / name) for key, (name, _) in FILES.items()}
    return (
        arrays["train_images"],
        arrays["train_labels"],
        arrays["test_images"],
        arrays["test_labels"],
    )
