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

import html as _html

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:wght@400;600;700&display=swap');

    /* ── Base & page background ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #faf6f0;
        color: #222222;
    }
    .stApp {
        background-color: #faf6f0;
    }
    .main .block-container {
        padding: 2rem 3rem 4rem;
        max-width: 1200px;
        background-color: #faf6f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #f2ebe0;
        border-right: 2px solid #e0d4c0;
    }
    [data-testid="stSidebar"] * {
        color: #3d3530 !important;
    }

    /* ── App header ── */
    .app-header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2.5rem 2rem 2rem;
        border-bottom: 2px solid #e0d4c0;
    }
    .app-header h1 {
        font-family: 'Lora', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2e1f;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #7a6e62;
        font-size: 0.97rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Progress ── */
    .progress-section {
        background: #f2ebe0;
        border: 1.5px solid #e0d4c0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.75rem;
    }
    .progress-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9c8a78;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.4rem;
    }

    /* ── Streamlit progress bar color ── */
    [data-testid="stProgressBar"] > div > div {
        background-color: #c9963a !important;
    }

    /* ── Prompt card ── */
    .prompt-card {
        background: #ffffff;
        border: 1.5px solid #e0d4c0;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 2px 8px rgba(139,108,70,0.07);
    }
    .prompt-tag {
        display: inline-block;
        background: #f5e9d5;
        color: #8a5c1e;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.9rem;
        border: 1px solid #e0c898;
    }
    .prompt-text {
        font-family: 'Lora', serif;
        font-size: 1.08rem;
        line-height: 1.7;
        color: #1f2e1f;
        font-weight: 400;
    }
    .prompt-tension {
        margin-top: 0.85rem;
        font-size: 0.78rem;
        color: #b09880;
        font-style: italic;
        letter-spacing: 0.01em;
    }

    /* ── Response cards ── */
    .response-card {
        background: #ffffff;
        border: 1.5px solid #e0d4c0;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        height: 100%;
        min-height: 300px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 1px 4px rgba(139,108,70,0.06);
    }
    .response-card.selected-a {
        border-color: #c9963a;
        box-shadow: 0 0 0 3px rgba(201,150,58,0.15);
    }
    .response-card.selected-b {
        border-color: #5f704f;
        box-shadow: 0 0 0 3px rgba(95,112,79,0.15);
    }
    .response-card.selected-tie {
        border-color: #9c4f48;
        box-shadow: 0 0 0 3px rgba(156,79,72,0.12);
    }
    .response-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1.5px solid #f0e8d8;
    }
    .response-label-a { color: #c9963a; }
    .response-label-b { color: #5f704f; }
    .response-text {
        font-size: 0.91rem;
        line-height: 1.75;
        color: #3d3530;
        white-space: pre-wrap;
    }
    .response-meta {
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid #f0e8d8;
        font-size: 0.7rem;
        color: #b09880;
        font-family: 'Sometype Mono', monospace;
    }

    /* ── Winner badge ── */
    .winner-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }
    .badge-a { background: #fdf0d5; color: #8a5c1e; border: 1px solid #e8c87a; }
    .badge-b { background: #eaf0e4; color: #3d5c30; border: 1px solid #a8c490; }
    .badge-tie { background: #fce8e6; color: #7a2e28; border: 1px solid #e8a8a4; }

    /* ── Preference label ── */
    .preference-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #9c8a78;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 1rem;
    }

    /* ── Streamlit button overrides ── */
    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.15s ease;
        border: 1.5px solid #d4c4a8 !important;
        background-color: #ffffff !important;
        color: #3d3530 !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #f5e9d5 !important;
        border-color: #c9963a !important;
        color: #8a5c1e !important;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #c9963a !important;
        border-color: #c9963a !important;
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #b5842e !important;
        border-color: #b5842e !important;
        color: #ffffff !important;
    }

    /* ── Stat boxes ── */
    .stat-box {
        background: #ffffff;
        border: 1.5px solid #e0d4c0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(139,108,70,0.06);
    }
    .stat-number {
        font-family: 'Lora', serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #c9963a;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.68rem;
        color: #9c8a78;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 0.3rem;
    }

    /* ── Completed card ── */
    .completed-card {
        background: #eaf0e4;
        border: 1.5px solid #a8c490;
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .completed-card h2 {
        font-family: 'Lora', serif;
        color: #3d5c30;
        margin-bottom: 0.5rem;
    }
    .completed-card p { color: #4a6e3a; }

    /* ── Dividers ── */
    hr {
        border-color: #e0d4c0 !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Supabase (optional) ─────────────────────────────────────────────────────

def get_supabase_client():
    """Return a Supabase client if credentials are configured, else None."""
    try:
        url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        url, key = "", ""
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
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
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

    text_a = _html.escape(resp_a['text'])
    text_b = _html.escape(resp_b['text'])

    with col_a:
        st.markdown(f"""
        <div class="{card_class_a}">
            <div class="response-label response-label-a">Response A</div>
            <div class="response-text">{text_a}</div>
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
            <div class="response-text">{text_b}</div>
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
