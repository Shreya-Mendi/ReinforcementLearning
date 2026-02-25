"""
Week 6 Lab — Human Preference Labeling App
Category: Advice Under Uncertainty

UI style: clean two-column comparison with progress tracking.
Storage: Supabase (primary, if configured) + session state + CSV export fallback.
Output format: {prompt, chosen, rejected} pairs for downstream RLHF training.
"""

import os
import io
import json
import time
import uuid
import csv
import random
from datetime import datetime, timezone

import streamlit as st
import anthropic

from prompts import PROMPTS, DATASET_METADATA

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Preference Labeler — Advice Under Uncertainty",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding: 2rem 3rem 4rem;
        max-width: 1200px;
    }

    /* ── App header ── */
    .app-header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: #6b7280;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Progress bar ── */
    .progress-section {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 2rem;
    }
    .progress-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* ── Prompt card ── */
    .prompt-card {
        background: #ffffff;
        border: 1.5px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .prompt-tag {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
    }
    .prompt-text {
        font-size: 1.1rem;
        line-height: 1.65;
        color: #111827;
        font-weight: 400;
    }
    .prompt-tension {
        margin-top: 0.75rem;
        font-size: 0.8rem;
        color: #9ca3af;
        font-style: italic;
    }

    /* ── Response cards ── */
    .response-card {
        background: #ffffff;
        border: 1.5px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        height: 100%;
        min-height: 280px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
    }
    .response-card.selected-a {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    .response-card.selected-b {
        border-color: #16a34a;
        box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.12);
    }
    .response-card.selected-tie {
        border-color: #d97706;
        box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.12);
    }
    .response-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #f3f4f6;
    }
    .response-label-a { color: #2563eb; }
    .response-label-b { color: #16a34a; }
    .response-text {
        font-size: 0.92rem;
        line-height: 1.7;
        color: #374151;
    }
    .response-meta {
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid #f3f4f6;
        font-size: 0.72rem;
        color: #9ca3af;
    }

    /* ── Preference buttons ── */
    .preference-section {
        margin: 1.5rem 0;
        text-align: center;
    }
    .preference-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    /* ── Winner badge ── */
    .winner-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .badge-a { background: #dbeafe; color: #1d4ed8; }
    .badge-b { background: #dcfce7; color: #15803d; }
    .badge-tie { background: #fef3c7; color: #92400e; }

    /* ── Sidebar stats ── */
    .stat-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* ── Generating spinner ── */
    .generating-msg {
        text-align: center;
        color: #6b7280;
        padding: 3rem 0;
        font-size: 0.95rem;
    }

    /* ── Completed state ── */
    .completed-card {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .completed-card h2 { color: #15803d; }
    .completed-card p { color: #166534; }

    /* ── Streamlit button overrides ── */
    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }

    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Supabase (optional) ─────────────────────────────────────────────────────

def get_supabase_client():
    """Return a Supabase client if credentials are configured, else None."""
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        return None


def save_to_supabase(client, record: dict) -> bool:
    """Insert a single preference record into Supabase. Returns success bool."""
    try:
        client.table("preferences").insert(record).execute()
        return True
    except Exception as e:
        st.warning(f"Supabase write failed: {e}")
        return False


# ─── LLM Calls ───────────────────────────────────────────────────────────────

def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error(
            "No ANTHROPIC_API_KEY found. Set it in `.streamlit/secrets.toml` or as an "
            "environment variable."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def generate_response(client: anthropic.Anthropic, prompt: str, temperature: float, seed_note: str) -> dict:
    """
    Generate a single response. Uses two different temperatures to produce variation.
    Returns a dict with response text and generation metadata.
    """
    system = (
        "You are a thoughtful advisor. Give genuine, practical advice that balances "
        "helpfulness with appropriate epistemic humility. Avoid empty hedging — be "
        "concrete where you can, uncertain where you should be."
    )
    t0 = time.time()
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = round(time.time() - t0, 2)
    return {
        "text": message.content[0].text,
        "model": message.model,
        "temperature": temperature,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "latency_s": elapsed,
        "stop_reason": message.stop_reason,
        "seed_note": seed_note,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_pair(client: anthropic.Anthropic, prompt: str) -> tuple[dict, dict]:
    """
    Generate two responses for the same prompt using different temperatures.
    Randomly assigns which temperature goes to A vs B to avoid ordering bias.
    """
    temps = [0.7, 1.0]
    random.shuffle(temps)
    resp_a = generate_response(client, prompt, temps[0], f"temp={temps[0]}")
    resp_b = generate_response(client, prompt, temps[1], f"temp={temps[1]}")
    return resp_a, resp_b


# ─── State Management ─────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "current_idx": 0,
        "responses": {},          # idx -> (resp_a, resp_b)
        "preferences": {},        # idx -> "A" | "B" | "Tie"
        "records": [],            # list of full preference records
        "generating": False,
        "prompts": PROMPTS.copy(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def current_prompt() -> dict:
    return st.session_state.prompts[st.session_state.current_idx]


def is_last_prompt() -> bool:
    return st.session_state.current_idx >= len(st.session_state.prompts) - 1


def all_labeled() -> bool:
    return len(st.session_state.preferences) >= len(st.session_state.prompts)


def build_record(prompt_obj: dict, resp_a: dict, resp_b: dict, preference: str) -> dict:
    """Build the full preference record including the {chosen, rejected} pair."""
    chosen = resp_a["text"] if preference == "A" else (resp_b["text"] if preference == "B" else None)
    rejected = resp_b["text"] if preference == "A" else (resp_a["text"] if preference == "B" else None)
    return {
        # Core RLHF fields
        "prompt": prompt_obj["prompt"],
        "chosen": chosen,
        "rejected": rejected,
        "preference": preference,   # "A" | "B" | "Tie"
        "is_tie": preference == "Tie",
        # Response A details
        "response_a": resp_a["text"],
        "response_a_model": resp_a["model"],
        "response_a_temp": resp_a["temperature"],
        "response_a_tokens": resp_a["output_tokens"],
        "response_a_latency": resp_a["latency_s"],
        # Response B details
        "response_b": resp_b["text"],
        "response_b_model": resp_b["model"],
        "response_b_temp": resp_b["temperature"],
        "response_b_tokens": resp_b["output_tokens"],
        "response_b_latency": resp_b["latency_s"],
        # Metadata
        "prompt_id": prompt_obj["id"],
        "prompt_tension": prompt_obj["tension"],
        "prompt_difficulty": prompt_obj["difficulty"],
        "session_id": st.session_state.session_id,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Export ───────────────────────────────────────────────────────────────────

def records_to_csv(records: list[dict]) -> str:
    if not records:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()


def records_to_jsonl(records: list[dict]) -> str:
    """JSONL format: one JSON object per line, only {prompt, chosen, rejected}."""
    lines = []
    for r in records:
        if not r["is_tie"]:
            lines.append(json.dumps({
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }))
    return "\n".join(lines)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚖ Preference Labeler")
        st.caption("Advice Under Uncertainty · RLHF Dataset")
        st.divider()

        total = len(st.session_state.prompts)
        labeled = len(st.session_state.preferences)
        a_wins = sum(1 for v in st.session_state.preferences.values() if v == "A")
        b_wins = sum(1 for v in st.session_state.preferences.values() if v == "B")
        ties   = sum(1 for v in st.session_state.preferences.values() if v == "Tie")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{labeled}</div>
                <div class="stat-label">Labeled</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{total - labeled}</div>
                <div class="stat-label">Remaining</div>
            </div>""", unsafe_allow_html=True)

        if labeled:
            st.markdown("**Preference breakdown**")
            st.markdown(f"- Response A preferred: **{a_wins}**")
            st.markdown(f"- Response B preferred: **{b_wins}**")
            st.markdown(f"- Ties: **{ties}**")
            st.divider()

        st.markdown("**Jump to prompt**")
        jump_options = {
            f"{'✓ ' if i in st.session_state.preferences else ''}{i+1}. {st.session_state.prompts[i]['prompt'][:50]}..."
            : i
            for i in range(total)
        }
        selected_label = st.selectbox(
            "Select prompt",
            list(jump_options.keys()),
            index=st.session_state.current_idx,
            label_visibility="collapsed",
        )
        if jump_options[selected_label] != st.session_state.current_idx:
            st.session_state.current_idx = jump_options[selected_label]
            st.rerun()

        st.divider()

        # Export section
        st.markdown("**Export data**")
        records = st.session_state.records
        if records:
            csv_str = records_to_csv(records)
            st.download_button(
                "⬇ Download CSV (full)",
                data=csv_str,
                file_name=f"preferences_{st.session_state.session_id[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            jsonl_str = records_to_jsonl(records)
            if jsonl_str:
                st.download_button(
                    "⬇ Download JSONL (chosen/rejected)",
                    data=jsonl_str,
                    file_name=f"rlhf_pairs_{st.session_state.session_id[:8]}.jsonl",
                    mime="application/x-ndjson",
                    use_container_width=True,
                )
                st.caption(f"{len([r for r in records if not r['is_tie']])} training pairs (ties excluded)")
        else:
            st.caption("Label some prompts to enable export.")

        st.divider()
        st.markdown("**Dataset info**")
        st.caption(f"Category: {DATASET_METADATA['category']}")
        st.caption(f"Session: `{st.session_state.session_id[:12]}...`")

        supabase_client = get_supabase_client()
        if supabase_client:
            st.success("Supabase connected")
        else:
            st.caption("No Supabase config — using local session only.")


# ─── Main UI ─────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="app-header">
        <h1>Advice Under Uncertainty</h1>
        <p>Read both AI responses and select which advice you'd rather receive — or mark them equal.</p>
    </div>
    """, unsafe_allow_html=True)


def render_progress():
    total = len(st.session_state.prompts)
    labeled = len(st.session_state.preferences)
    pct = labeled / total if total else 0
    st.markdown(f"""
    <div class="progress-section">
        <div class="progress-label">Progress — {labeled} of {total} labeled</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct)


def render_prompt(prompt_obj: dict):
    difficulty_colors = {"hard": "#fee2e2", "medium": "#fef9c3"}
    difficulty_text = {"hard": "#991b1b", "medium": "#854d0e"}
    diff = prompt_obj["difficulty"]
    st.markdown(f"""
    <div class="prompt-card">
        <span class="prompt-tag">Prompt {prompt_obj['id']} of {len(st.session_state.prompts)}</span>
        <div class="prompt-text">{prompt_obj['prompt']}</div>
        <div class="prompt-tension">Core tension: {prompt_obj['tension']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_responses(resp_a: dict, resp_b: dict, existing_pref: str | None):
    col_a, col_b = st.columns(2, gap="large")
    card_class_a = f"response-card {'selected-a' if existing_pref == 'A' else 'selected-tie' if existing_pref == 'Tie' else ''}"
    card_class_b = f"response-card {'selected-b' if existing_pref == 'B' else 'selected-tie' if existing_pref == 'Tie' else ''}"

    badge_a = '<span class="winner-badge badge-a">★ Preferred</span>' if existing_pref == "A" else ""
    badge_b = '<span class="winner-badge badge-b">★ Preferred</span>' if existing_pref == "B" else ""
    badge_tie_a = '<span class="winner-badge badge-tie">≈ Tie</span>' if existing_pref == "Tie" else ""
    badge_tie_b = '<span class="winner-badge badge-tie">≈ Tie</span>' if existing_pref == "Tie" else ""

    with col_a:
        st.markdown(f"""
        <div class="{card_class_a}">
            <div class="response-label response-label-a">Response A</div>
            <div class="response-text">{resp_a['text']}</div>
            <div class="response-meta">
                {resp_a['model']} · {resp_a['output_tokens']} tokens · {resp_a['latency_s']}s
            </div>
            {badge_a}{badge_tie_a}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="{card_class_b}">
            <div class="response-label response-label-b">Response B</div>
            <div class="response-text">{resp_b['text']}</div>
            <div class="response-meta">
                {resp_b['model']} · {resp_b['output_tokens']} tokens · {resp_b['latency_s']}s
            </div>
            {badge_b}{badge_tie_b}
        </div>
        """, unsafe_allow_html=True)


def render_preference_buttons(idx: int, existing_pref: str | None):
    st.markdown('<div class="preference-label" style="text-align:center;margin-top:1.5rem;">Which response do you prefer?</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        a_type = "primary" if existing_pref == "A" else "secondary"
        if st.button("← Prefer A", key=f"pref_a_{idx}", use_container_width=True, type=a_type):
            return "A"
    with col2:
        tie_type = "primary" if existing_pref == "Tie" else "secondary"
        if st.button("≈ Tie", key=f"pref_tie_{idx}", use_container_width=True, type=tie_type):
            return "Tie"
    with col3:
        b_type = "primary" if existing_pref == "B" else "secondary"
        if st.button("Prefer B →", key=f"pref_b_{idx}", use_container_width=True, type=b_type):
            return "B"
    return None


def render_completed():
    records = st.session_state.records
    non_tie = [r for r in records if not r["is_tie"]]
    st.markdown(f"""
    <div class="completed-card">
        <h2>All prompts labeled!</h2>
        <p>You labeled {len(records)} prompts · {len(non_tie)} usable training pairs · {len(records) - len(non_tie)} ties excluded</p>
    </div>
    """, unsafe_allow_html=True)

    if non_tie:
        st.subheader("Preview: RLHF training pairs")
        for r in non_tie[:3]:
            with st.expander(f"Prompt {r['prompt_id']}: {r['prompt'][:60]}..."):
                st.markdown(f"**Prompt:** {r['prompt']}")
                st.markdown(f"**Chosen:** {r['chosen'][:200]}...")
                st.markdown(f"**Rejected:** {r['rejected'][:200]}...")

    if st.button("Start over", type="secondary"):
        for key in ["responses", "preferences", "records", "current_idx", "generating"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    init_state()
    render_sidebar()
    render_header()
    render_progress()

    if all_labeled():
        render_completed()
        return

    idx = st.session_state.current_idx
    prompt_obj = current_prompt()
    supabase_client = get_supabase_client()

    render_prompt(prompt_obj)

    # ── Generate responses ────────────────────────────────────────────────
    if idx not in st.session_state.responses:
        if st.button("Generate responses", type="primary", use_container_width=False):
            with st.spinner("Generating two responses…"):
                client = get_anthropic_client()
                resp_a, resp_b = generate_pair(client, prompt_obj["prompt"])
                st.session_state.responses[idx] = (resp_a, resp_b)
            st.rerun()
        st.caption("Click to generate two AI responses for this prompt.")
        return

    # ── Show responses & preference buttons ──────────────────────────────
    resp_a, resp_b = st.session_state.responses[idx]
    existing_pref = st.session_state.preferences.get(idx)

    render_responses(resp_a, resp_b, existing_pref)

    chosen = render_preference_buttons(idx, existing_pref)

    if chosen:
        # Record the preference
        st.session_state.preferences[idx] = chosen
        record = build_record(prompt_obj, resp_a, resp_b, chosen)

        # Update or append record
        existing_record_idx = next(
            (i for i, r in enumerate(st.session_state.records) if r["prompt_id"] == prompt_obj["id"]),
            None,
        )
        if existing_record_idx is not None:
            st.session_state.records[existing_record_idx] = record
        else:
            st.session_state.records.append(record)

        # Supabase save
        if supabase_client:
            save_to_supabase(supabase_client, record)

        # Auto-advance to next unlabeled prompt
        if not is_last_prompt():
            next_unlabeled = next(
                (i for i in range(idx + 1, len(st.session_state.prompts))
                 if i not in st.session_state.preferences),
                None,
            )
            if next_unlabeled is not None:
                st.session_state.current_idx = next_unlabeled
        st.rerun()

    # ── Navigation ───────────────────────────────────────────────────────
    st.divider()
    col_prev, col_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        if idx > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.current_idx -= 1
                st.rerun()

    with col_info:
        status = f"✓ Labeled: {chosen or existing_pref}" if existing_pref else "Not yet labeled"
        st.markdown(f"<div style='text-align:center;color:#6b7280;font-size:0.85rem;padding-top:0.5rem;'>{status}</div>", unsafe_allow_html=True)

    with col_next:
        if not is_last_prompt():
            if st.button("Next →", use_container_width=True):
                st.session_state.current_idx += 1
                st.rerun()


if __name__ == "__main__":
    main()
