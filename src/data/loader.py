"""Memory-mapped token loader over packed .bin shards produced by prepare.py."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

DTYPE = np.uint32


class PackedDataset:
    """Yields random contiguous (x, y) windows from mmapped uint32 shards."""

    def __init__(self, data_dirs: list[str], seq_len: int, device: str = "cpu") -> None:
        self.seq_len = seq_len
        self.device = device
        self.shards: list[np.memmap] = []
        for d in data_dirs:
            for path in sorted(glob.glob(os.path.join(d, "*.bin"))):
                self.shards.append(np.memmap(path, dtype=DTYPE, mode="r"))
        if not self.shards:
            raise FileNotFoundError(f"no .bin shards found under {data_dirs}")
        self.lengths = np.array([len(s) for s in self.shards])
        self.total = int(self.lengths.sum())
        print(f"loaded {len(self.shards)} shard(s), {self.total:,} tokens")

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        xs, ys = [], []
        for _ in range(batch_size):
            si = int(torch.randint(len(self.shards), (1,), generator=generator).item())
            shard = self.shards[si]
            hi = len(shard) - self.seq_len - 1
            start = int(torch.randint(max(hi, 1), (1,), generator=generator).item())
            chunk = torch.from_numpy(
                shard[start : start + self.seq_len + 1].astype(np.int64)
            )
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.stack(xs)
        y = torch.stack(ys)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
