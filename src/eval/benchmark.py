"""Benchmark a checkpoint against the Qwen3-0.6B baseline.

Wraps EleutherAI's lm-evaluation-harness. Evaluates English reasoning + Korean
benchmarks so we can put a hard number on "how close are we to Qwen3-0.6B".

Install first:
    pip install "lm-eval[ifeval] @ git+https://github.com/EleutherAI/lm-evaluation-harness"

Our from-scratch checkpoints are saved as raw state_dicts, so convert to a HF dir
once via src.posttrain.sft.to_hf (or after SFT the model is already HF format).

    # evaluate an HF-format model dir (SFT/DPO output, or a converted base)
    python -m src.eval.benchmark --model checkpoints/sft
    # compare against the official baseline
    python -m src.eval.benchmark --model Qwen/Qwen3-0.6B --tag baseline
"""

from __future__ import annotations

import argparse
import subprocess

# English general + reasoning, then Korean.
DEFAULT_TASKS = "mmlu,hellaswag,arc_challenge,winogrande,gsm8k,kmmlu,haerae"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local HF dir")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model},dtype=bfloat16",
        "--tasks", args.tasks,
        "--batch_size", str(args.batch_size),
        "--output_path", f"{args.out}/{args.tag}",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nresults under {args.out}/{args.tag}/ — compare 'run' vs 'baseline'.")
    print("This score table is the credibility artifact for the tech report.")


if __name__ == "__main__":
    main()
