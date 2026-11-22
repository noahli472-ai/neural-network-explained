"""Train the digit classifier on MNIST and export it for the website.

Usage:
    python scripts/train_digits.py               # downloads MNIST on first run
    python scripts/train_digits.py --epochs 4    # quicker, slightly less accurate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from scratchnet.digits import train_digit_model
from scratchnet.export import export_network, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true", help="train on plain MNIST")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "digits.json")
    args = parser.parse_args()

    start = time.time()
    net, info = train_digit_model(epochs=args.epochs, seed=args.seed, augment=not args.no_augment)
    info["training_seconds"] = round(time.time() - start)
    model = {"version": 1, "info": info, **export_network(net)}
    size = write_json(model, args.out)
    print(
        f"test accuracy {info['test_accuracy']:.4f} "
        f"(renormalized {info['test_accuracy_renormalized']:.4f}); "
        f"wrote {args.out} ({size / 1024:.0f} KiB)"
    )


if __name__ == "__main__":
    main()
