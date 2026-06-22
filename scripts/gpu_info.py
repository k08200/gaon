"""Detect the GPUs on this box and recommend a micro_batch_size.

    python scripts/gpu_info.py [--model 0.6b|1.7b]

Removes the "is it A100 or H100?" guesswork: prints the actual hardware and a
safe per-GPU micro_batch_size for the chosen model, so you can paste it straight
into the config. Uses torch if available, else falls back to nvidia-smi.
"""

import argparse
import subprocess

# Rough safe per-GPU micro_batch at seq_len=4096, bf16, with activation ckpt.
# Keyed by (model, gpu_mem_GB_bucket). Conservative; bump if no OOM.
RECO = {
    "0.6b": {40: 8, 80: 16, 94: 24},
    "1.7b": {40: 4, 80: 8, 94: 12},
}


def via_torch():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append((p.name, round(p.total_memory / 1e9)))
    return out


def via_smi():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except Exception:
        return None
    out = []
    for line in r.stdout.strip().splitlines():
        name, mem = [s.strip() for s in line.split(",")]
        out.append((name, round(int(mem) / 1024)))
    return out


def bucket(mem_gb: int) -> int:
    for b in (94, 80, 40):
        if mem_gb >= b - 4:
            return b
    return 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=RECO, default="0.6b")
    args = ap.parse_args()

    gpus = via_torch() or via_smi()
    if not gpus:
        print("No GPU detected (no torch CUDA, no nvidia-smi). Run this on the NHN box.")
        return

    print(f"detected {len(gpus)} GPU(s):")
    mems = []
    for i, (name, mem) in enumerate(gpus):
        gen = "H100" if "H100" in name else "A100" if "A100" in name else "?"
        print(f"  [{i}] {name}  {mem}GB  ({gen})")
        mems.append(mem)

    b = bucket(min(mems))
    mb = RECO[args.model][b]
    fp8 = all("H100" in n for n, _ in gpus)
    print()
    print(f"recommended for {args.model} @ {b}GB-class:")
    print(f"  micro_batch_size: {mb}")
    print(f"  NGPU={len(gpus)}  -> set CUDA_VISIBLE_DEVICES accordingly")
    if fp8:
        print("  note: all H100 — FP8 available later for ~2x speedup (bf16 first).")


if __name__ == "__main__":
    main()
