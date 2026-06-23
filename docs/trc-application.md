# Google TPU Research Cloud (TRC) — Application

Apply at: https://sites.research.google/trc/about/  (short Google Form)

TRC grants **free** TPU v3/v4 quota (typically ~30 days, renewable) to researchers.
Use a **personal Google account** for this — Gaon is a personal project, keep it off
any company/work account or resources.

---

## Form fields

- **First / Last name:** (your name)
- **Email:** (your *personal* Google account — this account gets the TPU quota)
- **Country / region:** South Korea
- **Affiliation:** Independent researcher
- **What would you like to do with Cloud TPUs?** → paste the text below

---

## Project description (paste into the free-text box)

I'm training **Gaon**, a small open bilingual (Korean + English) language model, from
scratch as an independent research project. The model is a decoder-only transformer
(RoPE, Grouped-Query Attention, QK-Norm, SwiGLU, RMSNorm) starting at ~0.6B parameters,
with the goal of scaling the same training pipeline up to larger sizes.

My research focuses on: (1) efficient from-scratch pretraining of compact, strongly
Korean-capable models, since most small open models underperform in Korean; (2)
memory-efficient training techniques (e.g. chunked, gradient-checkpointed cross-entropy
to bound the large-vocabulary LM-head memory); and (3) a fully reproducible open
pipeline — pretraining on FineWeb-Edu and FineWeb-2 (Korean), followed by supervised
fine-tuning and DPO, evaluated on MMLU/HellaSwag and the Korean KMMLU/HAERAE benchmarks
against a same-size baseline.

I have a working pipeline and an early checkpoint (validation run reached a healthy loss
curve) and need sustained accelerator time to complete full-scale pretraining (tens of
billions of tokens), which free GPU tiers can't provide. I intend to release the model
weights and a technical report. TPU v3/v4 via PyTorch/XLA (or JAX) would let me run this
efficiently.

Thank you for considering my application.

---

## After approval (checklist)
- [ ] Create a GCP project tied to the personal account (TRC emails setup steps)
- [ ] Keep peripheral cost at ~$0: prefer TPU VM local disk; if using Cloud Storage,
      stay within free tier and delete buckets when done
- [ ] Port training to PyTorch/XLA (or JAX) — see code prep
- [ ] Add checkpoint resume (TRC TPUs can be preemptible)
- [ ] Continue Gaon from `checkpoints/gaon-0.6b-step1500-loss2.94.pt`
