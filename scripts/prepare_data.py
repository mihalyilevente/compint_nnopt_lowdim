"""Download checksum-verified IDX data and export the original notebook's CSV layout."""
import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from torchvision.datasets import MNIST, FashionMNIST


class OfficialFashionMNIST(FashionMNIST):
    # HTTPS copy in the dataset authors' repository; inherited MD5 checks apply.
    mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashionmnist", "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    specs = [("mnist", MNIST, "mnist"), ("fashionmnist", OfficialFashionMNIST, "fashion-mnist")]
    for name, cls, prefix in specs:
        if args.dataset not in (name, "all"):
            continue
        for train, split in [(True, "train"), (False, "test")]:
            dest = args.root / name / f"{prefix}_{split}.csv"
            if dest.exists():
                print(f"Preserving existing {dest}", flush=True)
                continue
            dataset = cls(root=str(args.root / "raw"), train=train, download=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
                    for start in range(0, len(dataset), 1000):
                        pixels = dataset.data[start:start + 1000].numpy().reshape(-1, 784)
                        labels = dataset.targets[start:start + 1000].numpy()[:, None]
                        pd.DataFrame(np.concatenate([labels, pixels], axis=1),
                                     columns=["label"] + [f"pixel{i}" for i in range(784)]).to_csv(
                                         stream, index=False, header=start == 0)
                # Atomically publish a complete file; fail if destination exists.
                # Temporary file is on the same filesystem (NTFS supports hard links).
                os.link(tmp, dest)
                print(f"Created {dest}: {len(dataset)} rows", flush=True)
            finally:
                Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
