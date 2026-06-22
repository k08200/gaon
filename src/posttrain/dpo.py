"""Direct Preference Optimization on top of the SFT model.

Run AFTER sft.py. DPO is simpler and more stable than RLHF and is the current
default for preference alignment. Uses TRL's DPOTrainer on a chosen/rejected
preference dataset.

    python -m src.posttrain.dpo --sft checkpoints/sft \
        --dataset HuggingFaceH4/ultrafeedback_binarized --out checkpoints/dpo
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="path to SFT model dir")
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(args.sft)
    model = AutoModelForCausalLM.from_pretrained(args.sft)
    ds = load_dataset(args.dataset, split="train_prefs")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # TRL builds the frozen ref internally
        train_dataset=ds,
        processing_class=tok,
        args=DPOConfig(
            output_dir=args.out,
            beta=args.beta,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            bf16=True,
            logging_steps=20,
            save_strategy="epoch",
            max_length=2048,
            max_prompt_length=1024,
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"DPO model saved -> {args.out}")


if __name__ == "__main__":
    main()
