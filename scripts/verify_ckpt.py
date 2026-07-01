"""Validate that an FSDP checkpoint restores correctly (loss sane, not random).

    .venv/bin/python -m scripts.verify_ckpt checkpoints/gaon_1.7b_4gpu/latest.pt

Loads the saved weights into a fresh model on CPU and computes loss on a few
batches. A correctly-saved full-state-dict checkpoint gives loss ~ the training
loss (~3-4). A corrupt sharded save gives ~11-12 (random-level).
"""
import sys
import torch
import yaml

from src.model import ModelConfig, Gaon
from src.data.loader import PackedDataset

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/gaon_1.7b_4gpu/latest.pt"
cfg = yaml.safe_load(open("configs/gaon_1.7b_4gpu.yaml"))
mcfg = ModelConfig(**cfg["model"])

device = "cpu"
model = Gaon(mcfg).to(device)
ck = torch.load(ckpt_path, map_location="cpu")
sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
model.load_state_dict(sd, strict=True)
model.eval()
print(f"loaded {ckpt_path} @ step {ck['step']}", flush=True)

data = PackedDataset(cfg["data_dirs"], mcfg.max_seq_len, device=device)
losses = []
with torch.no_grad():
    for _ in range(5):
        x, y = data.get_batch(2)
        _, loss = model(x, y)
        losses.append(loss.item())

mean = sum(losses) / len(losses)
print("val losses:", [round(l, 3) for l in losses], flush=True)
print(f"MEAN LOSS: {mean:.3f}", flush=True)
print("RESULT:", "FIX OK (checkpoint restores correctly)" if mean < 6
      else "STILL BROKEN (random-level loss)", flush=True)
import os
os._exit(0)
