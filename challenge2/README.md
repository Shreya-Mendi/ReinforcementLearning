# Challenge 2: RLHF on GPT-2 — When Should a Voice Assistant Stay Silent?

End-to-end implementation of the RLHF pipeline (Reward Model → PPO → Alignment Evaluation) on GPT-2 (124M), trained on human preference data for voice assistant conversations.

---

## Task

Given a voice assistant conversation, train a model to prefer `(silent)` responses over verbose interruptions — i.e., "knowing when to shut up."

---

## Repository Structure

```
challenge2/
├── notebooks/
│   ├── challenge2.ipynb   # Main implementation: RM training, PPO alignment, evaluation
│   └── main.ipynb         # Scratch / exploratory runs
├── docs/
│   ├── RLHF_Analysis.md   # Full experiment report (failures, fixes, results)
│   └── rl-challenge2.pdf  # Original challenge specification
└── data/                  # Preference pairs (generated at runtime, not committed)
```

---

## Pipeline

```
256 preference pairs (prompt, chosen, rejected)
        │
        ▼
[Stage 1] SFT warm-start
  Fine-tune GPT-2 on chosen responses (causal LM loss)
  → teaches the model (silent) as a valid output
        │
        ▼
[Stage 2] Reward Model
  GPT-2 + classification head, trained with Bradley-Terry loss
  → learns to score chosen > rejected
        │
        ▼
[Stage 3] PPO Alignment
  Policy (SFT checkpoint) optimized against frozen RM
  KL penalty anchors policy to SFT reference
        │
        ▼
[Eval] Misalignment signals: KL divergence, repetition score, silent rate, RM pairwise accuracy
```

---

## Key Results

| Version | RM Pairwise Acc | PPO KL | Repetition | Notes |
|---------|----------------|--------|------------|-------|
| v1 | 0.35 ❌ | −483 (collapsed) | — | Truncation bug + KL explosion |
| v2 | 0.692 ✓ | 23.66 (8× target) | +60% (reward hack) | RM fixed, PPO unstable |
| v3 | 0.750 ✓ | 0.16 ✓ | −45% ✓ | SFT warm-start added |
| v4 | TBD | TBD | TBD | PPOv2Trainer (planned) |

---

## Key Findings

1. **Truncation direction matters.** Right-truncation silently destroys the preference signal when the distinguishing token is at the end of a conversation. Fix: `truncation_side="left"`.

2. **`ppo_epochs > 1` is dangerous on small datasets.** Stale rollout data causes importance weight explosion. Use `ppo_epochs=1` with more outer steps.

3. **A broken reward model makes alignment worse than none.** With pairwise acc = 0.35, PPO actively pushed the policy toward rejected responses.

4. **Dataset imbalance explained the 0% silent rate.** 65.6% of pairs had `chosen=verbose`, so the RM penalized silence — PPO never had a valid signal to reinforce `(silent)`.

5. **SFT warm-start (InstructGPT Stage 1) is not optional.** PPO can only reinforce behaviors the policy already sometimes generates. Without SFT, GPT-2 never produces `(silent)` and reward signal has nothing to latch onto.

---

## Setup

```bash
pip install transformers trl datasets torch
```

Run `notebooks/challenge2.ipynb` in order — each cell is labeled by pipeline stage.

---

## Full Analysis

See [docs/RLHF_Analysis.md](docs/RLHF_Analysis.md) for the complete experiment log: failure diagnoses, hyperparameter ablations, and per-version results.
