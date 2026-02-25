# Week 6 Lab — Human Preference Labeling App

**Category:** Advice Under Uncertainty
**Goal:** Collect `{prompt, chosen, rejected}` pairs for downstream RLHF training.

---

## What it does

- Displays one advice-under-uncertainty prompt at a time
- Generates **two responses** from the same model (varied by sampling temperature)
- You pick which response you prefer — or mark a tie
- Responses are **fixed once generated** (no re-rolls)
- Exports labeled data as **CSV** (full metadata) or **JSONL** (`{prompt, chosen, rejected}`)
- Optional: saves to **Supabase** in real time

---

## Quick start

```bash
cd lab6
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY

streamlit run app.py
```

Or put your key in `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Supabase (stretch goal)

1. Create a Supabase project at https://supabase.com
2. Run `supabase_schema.sql` in the Supabase SQL editor
3. Add to `.env`:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key
   ```

Without Supabase, data persists in the browser session — use the sidebar export buttons before closing.

---

## Output formats

### CSV (full metadata)
All fields: prompt, responses, preference, model, temperature, token counts, latency, session ID, timestamp.

### JSONL (training pairs)
One JSON object per line, ties excluded:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```
Ready for reward model training with TRL, OpenRLHF, or any RLHF framework.

---

## Prompt dataset

25 prompts in the **Advice Under Uncertainty** category. Each prompt is tagged with:
- `tension` — the core tradeoff being probed
- `difficulty` — `medium` or `hard`
- `notes` — what to look for in responses

**Design rationale:** These prompts are designed to make preference labeling *genuinely difficult* — a good response balances concrete helpfulness with appropriate epistemic humility. Both sycophantic ("follow your dreams!") and excessively hedged ("it depends on many factors...") responses should score poorly.

---

## Why this matters for alignment

This app collects the raw signal that makes RLHF work:
- **Interface choices** affect label quality (showing uncertainty tags, metadata)
- **Prompt design** affects what behaviors get selected for or against
- **Tie rate** is a signal of prompt quality — too many ties means prompts aren't discriminative
- **Export format** must match your downstream training pipeline
