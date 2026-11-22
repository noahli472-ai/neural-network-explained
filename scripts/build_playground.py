"""Record the training runs replayed in stations 1 and 2 of the website.

Usage:
    python scripts/build_playground.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scratchnet.export import write_json
from scratchnet.playground import build_playground

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "data" / "playground.json")
    args = parser.parse_args()

    data = build_playground(seed=args.seed)
    size = write_json(data, args.out)
    for key, run in data["runs"].items():
        print(f"{key:16s} final accuracy {run['frames'][-1]['accuracy']:.3f}")
    print(f"wrote {args.out} ({size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
