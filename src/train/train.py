"""Pretraining loop for the Qwen3-style model.

Single-GPU and multi-GPU (FSDP via torchrun) capable. Start small to validate the
full pipeline, then scale `target_tokens` and GPU count without code changes.

Single GPU:
    python -m src.train.train --config configs/qwen3_0.6b.yaml

Multi-GPU (e.g. 5x A100):
    torchrun --standalone --nproc_per_node=5 -m src.train.train \
        --config configs/qwen3_0.6b.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import yaml

from ..data.loader import PackedDataset
from ..model import ModelConfig, Gaon


def is_dist() -> bool:
    return int(os.environ.get("RANK", -1)) != -1


def setup_dist():
    if not is_dist():
        return 0, 0, 1
    import torch.distributed as dist
    from datetime import timedelta

    # Moderate timeout: long enough to ride out a brief co-located inference burst,
    # short enough that a SUSTAINED grab (a colleague's vLLM serving for an hour) makes
    # us crash-and-resume fast instead of stalling — holding GPU memory — the whole time.
    # Resume is correct now (full-state-dict checkpoints), so a crash is cheap.
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=20))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def get_lr(step: int, cfg: dict) -> float:
    warmup, total = cfg["warmup_steps"], cfg["max_steps"]
    lr, min_lr = cfg["lr"], cfg["min_lr"]
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def build_optimizer(model, cfg, device, master):
    """AdamW, or 8-bit AdamW (bitsandbytes) to cut optimizer memory ~4x on small GPUs."""
    lr, wd = cfg["lr"], cfg["weight_decay"]
    if cfg.get("optimizer", "adamw") == "adamw8bit":
        try:
            import bitsandbytes as bnb
            if master:
                print("optimizer: 8-bit AdamW (bitsandbytes)")
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
        except ImportError:
            if master:
                print("bitsandbytes not installed -> falling back to fp32 AdamW")
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd,
        fused=device.startswith("cuda"),
    )


def wrap_fsdp(model, local_rank):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools
    from ..model.gaon import Block

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, local_rank, world = setup_dist()
    master = rank == 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337 + rank)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        # cap our share of a SHARED GPU so we don't starve other users' resident jobs
        mf = cfg.get("max_mem_fraction")
        if mf:
            torch.cuda.set_per_process_memory_fraction(float(mf), local_rank)
            if master:
                print(f"GPU memory cap: {float(mf)*100:.0f}% of device")

    mcfg = ModelConfig(**cfg.get("model", {}))
    model = Gaon(mcfg).to(device)
    model.grad_checkpoint = cfg.get("grad_checkpoint", False)
    model.loss_chunk_size = cfg.get("loss_chunk_size", 4096)
    if master:
        print(f"model params: {model.num_params() / 1e6:.1f}M | "
              f"grad_checkpoint={model.grad_checkpoint} chunk={model.loss_chunk_size}")

    # --- resume: load weights into the plain model BEFORE FSDP wrap ---
    start_step = 0
    resume_path = cfg.get("resume_from")
    resume_ck = None
    if resume_path and os.path.exists(resume_path):
        resume_ck = torch.load(resume_path, map_location=device)
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
              for k, v in resume_ck["model"].items()}
        model.load_state_dict(sd, strict=True)
        start_step = int(resume_ck.get("step", 0))
        if master:
            print(f"resumed model from {resume_path} @ step {start_step}")

    if is_dist():
        model = wrap_fsdp(model, local_rank)

    opt = build_optimizer(model, cfg, device, master)

    # optimizer state only resumes in the single-process case (Colab/1-GPU);
    # under FSDP it is sharded and re-initialized (warmup smooths the restart).
    if resume_ck is not None and "optimizer" in resume_ck and not is_dist():
        opt.load_state_dict(resume_ck["optimizer"])
        if master:
            print("resumed optimizer state")
    resume_ck = None

    data = PackedDataset(cfg["data_dirs"], mcfg.max_seq_len, device=device)
    micro_bs = cfg["micro_batch_size"]
    grad_accum = cfg["grad_accum_steps"]
    tokens_per_step = micro_bs * grad_accum * world * mcfg.max_seq_len

    # precision: bf16 (A100/H100/B200), fp16 (T4/older, needs GradScaler), or fp32.
    dtype_name = cfg.get("dtype", "bf16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    use_amp = device.startswith("cuda") and amp_dtype != torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else _null()
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype_name == "fp16"))
    if master:
        print(f"precision: {dtype_name}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    t0 = time.time()
    for step in range(start_step, cfg["max_steps"]):
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = data.get_batch(micro_bs)
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        scaler.unscale_(opt)
        if hasattr(model, "clip_grad_norm_"):
            model.clip_grad_norm_(cfg["grad_clip"])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()

        if master and step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            tps = tokens_per_step * (step + 1) / dt
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"{tps/1e3:.1f}k tok/s | {step * tokens_per_step / 1e9:.2f}B tok")

        # save() runs on ALL ranks under FSDP (the full-state-dict gather is a
        # collective — every rank must participate); only master writes the file.
        if step > 0 and step % cfg["save_interval"] == 0:
            save(model, opt, mcfg, cfg, step, master)

    save(model, opt, mcfg, cfg, cfg["max_steps"], master)
    if is_dist():
        import torch.distributed as dist
        dist.destroy_process_group()


def save(model, opt, mcfg, cfg, step, master=True):
    # Under FSDP each rank holds only a 1/N shard of every parameter. A plain
    # model.state_dict() would save just the local shard -> a corrupt checkpoint
    # (this exact bug gave loss 11.9 on resume). FULL_STATE_DICT gathers the full,
    # unsharded weights onto rank 0 (offloaded to CPU). It is a COLLECTIVE: all
    # ranks must call it, but only rank 0 receives the populated dict.
    if is_dist():
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            sd = model.state_dict()
    else:
        sd = model.state_dict()
    if not master:                          # non-master ranks only join the gather
        return
    ckpt = {"model": sd, "config": mcfg.__dict__, "step": step}
    if not is_dist():                       # FSDP optimizer state is sharded; skip
        ckpt["optimizer"] = opt.state_dict()
    path = os.path.join(cfg["out_dir"], f"ckpt_{step}.pt")
    torch.save(ckpt, path)
    # stable 'latest.pt' pointer so resume can auto-find the newest checkpoint
    latest = os.path.join(cfg["out_dir"], "latest.pt")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except OSError:
        pass
    print(f"saved checkpoint -> {path}")

    # Disk safety: keep only the most recent N checkpoints. Intermediate ckpts
    # exist for crash-resume; on a shared box, unbounded accumulation fills the
    # disk (60 * 8GB filled 1.6TB once). 0/unset = keep all.
    import glob
    import re
    keep = int(cfg.get("keep_last_checkpoints", 0))
    if keep > 0:
        cks = glob.glob(os.path.join(cfg["out_dir"], "ckpt_*.pt"))
        cks.sort(key=lambda p: int(re.search(r"ckpt_(\d+)\.pt", os.path.basename(p)).group(1)))
        for old in cks[:-keep]:
            try:
                os.remove(old)
            except OSError:
                pass


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
