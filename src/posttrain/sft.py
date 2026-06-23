"""Supervised fine-tuning (instruction tuning) of the pretrained base model.

Converts our from-scratch checkpoint into a HuggingFace Qwen3 model so we can use
TRL's mature SFT/DPO trainers, then teaches it to follow instructions and emit the
chat format. Run this AFTER pretraining converges.

    python -m src.posttrain.sft --ckpt checkpoints/qwen3_0.6b/ckpt_40000.pt \
        --dataset HuggingFaceH4/ultrachat_200k --out checkpoints/sft

Note: weight-name mapping from our module to HF Qwen3 lives in `to_hf()`. Keep it
in sync if you change the architecture. For Korean instruction following, mix in a
Korean instruct dataset alongside ultrachat.
"""

from __future__ import annotations

import argparse

import torch


def to_hf(ckpt_path: str, tokenizer_id: str):
    """Map our state_dict onto a HF Qwen3ForCausalLM with matching config."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from ..model import ModelConfig

    ck = torch.load(ckpt_path, map_location="cpu")
    c = ModelConfig(**ck["config"])
    hf_cfg = Qwen3Config(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_layers,
        num_attention_heads=c.num_attn_heads,
        num_key_value_heads=c.num_kv_heads,
        head_dim=c.head_dim,
        max_position_embeddings=c.max_seq_len,
        rope_theta=c.rope_theta,
        rms_norm_eps=c.rms_norm_eps,
        tie_word_embeddings=c.tie_embeddings,
        attention_bias=False,
    )
    hf = Qwen3ForCausalLM(hf_cfg)
    src = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}

    new = {}
    new["model.embed_tokens.weight"] = src["embed.weight"]
    new["model.norm.weight"] = src["norm.weight"]
    if not c.tie_embeddings:
        new["lm_head.weight"] = src["lm_head.weight"]
    for i in range(c.num_layers):
        p, q = f"blocks.{i}", f"model.layers.{i}"
        new[f"{q}.input_layernorm.weight"] = src[f"{p}.attn_norm.weight"]
        new[f"{q}.post_attention_layernorm.weight"] = src[f"{p}.mlp_norm.weight"]
        new[f"{q}.self_attn.q_proj.weight"] = src[f"{p}.attn.q_proj.weight"]
        new[f"{q}.self_attn.k_proj.weight"] = src[f"{p}.attn.k_proj.weight"]
        new[f"{q}.self_attn.v_proj.weight"] = src[f"{p}.attn.v_proj.weight"]
        new[f"{q}.self_attn.o_proj.weight"] = src[f"{p}.attn.o_proj.weight"]
        new[f"{q}.self_attn.q_norm.weight"] = src[f"{p}.attn.q_norm.weight"]
        new[f"{q}.self_attn.k_norm.weight"] = src[f"{p}.attn.k_norm.weight"]
        new[f"{q}.mlp.gate_proj.weight"] = src[f"{p}.mlp.gate_proj.weight"]
        new[f"{q}.mlp.up_proj.weight"] = src[f"{p}.mlp.up_proj.weight"]
        new[f"{q}.mlp.down_proj.weight"] = src[f"{p}.mlp.down_proj.weight"]
    missing, unexpected = hf.load_state_dict(new, strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys mapping to HF: {unexpected[:5]}")
    return hf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = to_hf(args.ckpt, args.tokenizer)
    ds = load_dataset(args.dataset, split="train_sft")

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        processing_class=tok,
        args=SFTConfig(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            bf16=True,
            logging_steps=20,
            save_strategy="epoch",
            max_length=2048,        # trl>=1.x renamed from max_seq_length
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"SFT model saved -> {args.out}")


if __name__ == "__main__":
    main()
