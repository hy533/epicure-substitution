import json
import numpy as np
from pathlib import Path
from difflib import get_close_matches
import streamlit as st
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

st.set_page_config(page_title="Epicure", page_icon="⚗", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

:root {
  --bg:      #0c0c09;
  --bg2:     #111109;
  --border:  #2a2a20;
  --text:    #d4c9a8;
  --dim:     #5a5a40;
  --amber:   #c8892a;
  --green:   #6a8f5a;
  --red:     #a05050;
  --heading: #f0e6c8;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] { padding: 2rem 3rem !important; max-width: 1100px; }

/* Inputs */
input, textarea {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 13px !important;
}
input:focus, textarea:focus {
  border-color: var(--amber) !important;
  box-shadow: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 9px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  color: var(--amber) !important;
  font-weight: 600 !important;
}

/* Button */
[data-testid="stButton"] button {
  background: var(--amber) !important;
  color: var(--bg) !important;
  border: none !important;
  border-radius: 0 !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  padding: 14px 28px !important;
  width: 100% !important;
}
[data-testid="stButton"] button:hover {
  background: #e09a35 !important;
  color: var(--bg) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Spinner */
[data-testid="stSpinner"] { color: var(--amber) !important; }

/* Captions */
[data-testid="stCaptionContainer"] p {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important;
  color: #5a5a40 !important;
  letter-spacing: 0.5px !important;
  margin-top: 2px !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #2a2a20;padding-bottom:12px;margin-bottom:6px">
  <span style="font-family:'Cormorant Garamond',serif;font-size:38px;font-weight:600;color:#f0e6c8;letter-spacing:-0.5px">Epicure</span>
</div>
<div style="font-size:9px;letter-spacing:3px;text-transform:uppercase;color:#5a5a40;margin-bottom:36px">
  Flavor substitution engine &nbsp;·&nbsp; Chem embeddings &nbsp;·&nbsp; 1790 ingredients
</div>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
CACHE = Path("scratch/cache")
MODELS = ["Kaikaku/epicure-cooc", "Kaikaku/epicure-chem", "Kaikaku/epicure-core"]

@st.cache_resource(show_spinner="Loading flavor embeddings…")
def load_models():
    matrices, vocab, idx_to_name = {}, None, None
    for repo in MODELS:
        name = repo.split("/")[1]
        emb_path  = hf_hub_download(repo_id=repo, filename="embeddings.safetensors", cache_dir=CACHE)
        vocab_path = hf_hub_download(repo_id=repo, filename="vocab.json", cache_dir=CACHE)
        tensors = load_file(emb_path)
        mat = tensors["embeddings"].astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        matrices[name] = mat / np.where(norms == 0, 1.0, norms)
        if vocab is None:
            with open(vocab_path) as f:
                vocab = json.load(f)
            idx_to_name = {v: k for k, v in vocab.items()}
    return matrices, vocab, idx_to_name

matrices, vocab, idx_to_name = load_models()
ALL_VOCAB = set(vocab.keys())

# ── Normalization ─────────────────────────────────────────────────────────────
ALIASES = {"soy_sauce": "light_soy_sauce"}

def normalize(s: str) -> str:
    s = s.strip().lower()
    if s in ALL_VOCAB: return s                          # D-003: raw first
    s = s.replace(" ", "_").replace("-", "_")
    if s in ALIASES: return ALIASES[s]                  # D-002: alias map
    if s in ALL_VOCAB: return s
    for pre in ("fresh_", "dried_", "ground_", "whole_", "raw_", "roasted_"):
        t = s.removeprefix(pre)
        if t in ALL_VOCAB: return t
    for suf in ("_cheese", "_pepper", "_mushroom", "_sauce", "_oil"):
        if (s + suf) in ALL_VOCAB: return s + suf       # D-004: suffix expand
    if s.endswith("_chili") and (s[:-6] + "_chile") in ALL_VOCAB:
        return s[:-6] + "_chile"
    if s.endswith("_chile") and (s[:-6] + "_chili") in ALL_VOCAB:
        return s[:-6] + "_chili"
    return s

def check(raw: str):
    canon = normalize(raw)
    if canon in ALL_VOCAB:
        status = "alias" if canon != raw.strip().lower().replace(" ", "_").replace("-", "_") else "exact"
        return canon, status
    suggestions = get_close_matches(canon, ALL_VOCAB, n=2, cutoff=0.6)
    return None, f"missing — did you mean: {', '.join(suggestions)}" if suggestions else "missing"

def parse_list(text: str) -> list[str]:
    return [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]

# ── Engine ────────────────────────────────────────────────────────────────────
def get_neighbors(ingredient: str, k: int = 30) -> list[tuple[str, float]]:
    idx = vocab[ingredient]
    mat = matrices["epicure-chem"]
    sims = mat @ mat[idx]
    top = np.argsort(sims)[::-1]
    results = []
    for i in top:
        if i == idx: continue
        results.append((idx_to_name[i], float(sims[i])))
        if len(results) == k: break
    return results

def substitute(ingredient: str, available: set, forbidden: set, k: int = 50, n: int = 3):
    nb_display = get_neighbors(ingredient, k)
    results = []

    # Collect constrained candidates from top-k
    for name, sim in nb_display:
        if name in available and name not in forbidden:
            results.append({"result": name, "sim": round(sim, 3), "constrained": True, "rank": None})
            if len(results) == n:
                return results, nb_display

    # Full scan for remaining pantry slots
    if available and len(results) < n:
        idx = vocab[ingredient]
        mat = matrices["epicure-chem"]
        sims = mat @ mat[idx]
        ranked = np.argsort(sims)[::-1]
        seen = {r["result"] for r in results}
        rank = 0
        for i in ranked:
            if i == idx: continue
            rank += 1
            name = idx_to_name[i]
            if name in seen: continue
            if name in available and name not in forbidden:
                results.append({"result": name, "sim": round(float(sims[i]), 3), "constrained": False, "rank": rank})
                seen.add(name)
                if len(results) == n:
                    return results, nb_display

    # Fill remaining slots with unconstrained top neighbors
    seen = {r["result"] for r in results}
    for name, sim in nb_display:
        if name not in seen:
            results.append({"result": name, "sim": round(sim, 3), "constrained": False, "rank": None})
            seen.add(name)
            if len(results) == n:
                break

    return results, nb_display

# ── Pantry persistence ────────────────────────────────────────────────────────
PANTRY_FILE = Path("scratch/pantry.json")

def load_pantry() -> str:
    if PANTRY_FILE.exists():
        try:
            return json.loads(PANTRY_FILE.read_text()).get("pantry", "")
        except Exception:
            return ""
    return ""

def save_pantry(text: str):
    PANTRY_FILE.write_text(json.dumps({"pantry": text}))

if "pantry_input" not in st.session_state:
    st.session_state["pantry_input"] = load_pantry()
if "pantry_saved" not in st.session_state:
    st.session_state["pantry_saved"] = st.session_state["pantry_input"]

# ── Shared UI helpers ─────────────────────────────────────────────────────────
import streamlit.components.v1 as components

LABELS     = ["Best substitute", "2nd choice", "3rd choice"]
FONT_SIZES = ["28px", "22px", "18px"]

def render_result_cards(results, col=None, label_offset=0):
    ctx = col if col else st
    for i, result in enumerate(results):
        li = label_offset + i
        if result["constrained"]:
            status_html  = '<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#6a8f5a;margin-top:8px">✓ in pantry</div>'
            border_color = "#c8892a"
        elif result["rank"] is not None:
            status_html  = f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#c8892a;margin-top:8px">⚠ rank #{result["rank"]}</div>'
            border_color = "#7a5a1a"
        else:
            status_html  = '<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#a05050;margin-top:8px">✗ not in pantry</div>'
            border_color = "#3a3a2a" if li > 0 else "#5a2a2a"

        margin = "0 0 8px 0" if i < len(results) - 1 else "0"
        ctx.markdown(f"""
        <div style="background:#111109;border:1px solid #2a2a20;border-left:3px solid {border_color};padding:16px;margin:{margin}">
          <div style="font-size:9px;letter-spacing:3px;text-transform:uppercase;color:#5a5a40">{LABELS[li]}</div>
          <div style="font-family:'Cormorant Garamond',serif;font-size:{FONT_SIZES[li]};font-weight:600;color:#f0e6c8;margin:6px 0 4px;line-height:1.1">{result['result']}</div>
          <div style="font-size:11px;color:#5a5a40">sim <span style="color:#c8892a">{result['sim']}</span></div>
          {status_html}
        </div>
        """, unsafe_allow_html=True)

def render_neighbor_panel(neighbors, pantry_tokens, forbidden_tokens):
    rows_html = ""
    for name, sim in neighbors:
        in_pantry    = name in pantry_tokens
        is_forbidden = name in forbidden_tokens
        if is_forbidden:
            tag, name_class, sim_class = '<span class="tag tag-forbidden">forbidden</span>', "nb-name forbidden", "nb-sim dim"
        elif in_pantry:
            tag, name_class, sim_class = '<span class="tag tag-pantry">pantry</span>', "nb-name pantry", "nb-sim green"
        else:
            tag, name_class, sim_class = '<span class="tag-spacer"></span>', "nb-name", "nb-sim"
        rows_html += f'<div class="nb-row">{tag}<span class="{name_class}">{name}</span><span class="{sim_class}">{sim:.3f}</span></div>'

    panel_html = f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  body {{ margin:0; background:#111109; font-family:'IBM Plex Mono',monospace; color:#d4c9a8; }}
  .header {{ font-size:9px; letter-spacing:3px; text-transform:uppercase; color:#5a5a40;
             border-bottom:1px solid #1e1e14; padding-bottom:8px; margin-bottom:4px; }}
  .nb-row {{ display:flex; align-items:baseline; gap:10px; padding:5px 0;
             border-bottom:1px solid #181810; font-size:12px; }}
  .nb-name {{ flex:1; color:#d4c9a8; }}
  .nb-name.pantry {{ color:#a0c890; }}
  .nb-name.forbidden {{ color:#704040; text-decoration:line-through; }}
  .nb-sim {{ width:38px; text-align:right; color:#5a5a40; font-size:11px; }}
  .nb-sim.green {{ color:#6a8f5a; }}
  .nb-sim.dim {{ color:#3a2a2a; }}
  .tag {{ font-size:8px; letter-spacing:1.5px; text-transform:uppercase;
          padding:2px 5px; font-weight:600; white-space:nowrap; }}
  .tag-pantry {{ background:#1a2e14; color:#6a8f5a; }}
  .tag-forbidden {{ background:#2e1414; color:#a05050; }}
  .tag-spacer {{ display:inline-block; width:54px; }}
</style></head>
<body>
  <div class="header">Chem neighbors — top {len(neighbors)}</div>
  {rows_html}
</body></html>"""
    components.html(panel_html, height=min(30 * len(neighbors) + 60, 700), scrolling=True)

# ── Shared: pantry & forbidden ────────────────────────────────────────────────
col_f, col_p = st.columns([1, 2])

with col_f:
    forbidden_raw = st.text_input(
        "Forbidden / allergens",
        placeholder="e.g. soybean_paste, fermented_black_bean",
        help="Comma-separated. Excluded even if they're in your pantry.",
    )
    st.caption("Comma-separated. Leave blank if none.")

with col_p:
    pantry_raw = st.text_area(
        "Pantry",
        placeholder="e.g. miso, gochujang, fish_sauce, tamarind, rice_vinegar",
        height=80,
        key="pantry_input",
        help="Comma-separated. Saved automatically. Used in both Single and Batch modes.",
    )
    saved_indicator = ""
    if pantry_raw != st.session_state["pantry_saved"]:
        st.session_state["pantry_saved"] = pantry_raw
        save_pantry(pantry_raw)
        saved_indicator = " · saved"
    st.caption(f"Comma-separated. Persists across restarts{saved_indicator}.")

pantry_tokens   = {normalize(t) for t in parse_list(pantry_raw) if check(t)[0]}
forbidden_tokens = {normalize(t) for t in parse_list(forbidden_raw) if check(t)[0]}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single", "Batch"])

# ── Single tab ────────────────────────────────────────────────────────────────
with tab_single:
    ing_raw = st.text_input(
        "Ingredient",
        placeholder="e.g. doubanjiang",
        help="The ingredient you want to substitute. Spaces and underscores both work.",
    )
    st.caption("The ingredient you're out of or want to replace.")
    if ing_raw:
        canon, status = check(ing_raw)
        if status == "exact":
            st.markdown(f'<div style="font-size:10px;color:#6a8f5a;letter-spacing:1px;margin:-8px 0 4px">✓ exact match → <code style="background:none;color:#6a8f5a">{canon}</code></div>', unsafe_allow_html=True)
        elif status == "alias":
            st.markdown(f'<div style="font-size:10px;color:#c8892a;letter-spacing:1px;margin:-8px 0 4px">~ normalized → <code style="background:none;color:#c8892a">{canon}</code></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:10px;color:#a05050;letter-spacing:1px;margin:-8px 0 4px">✗ {status}</div>', unsafe_allow_html=True)

    run = st.button("Find substitute →", key="run_single")

    if run and ing_raw:
        canon, status = check(ing_raw)
        if canon is None:
            st.markdown(f'<div style="color:#a05050;font-size:12px;padding:16px;border:1px solid #2e1414">✗ {ing_raw!r} not in vocab. {status}</div>', unsafe_allow_html=True)
        else:
            results, neighbors = substitute(canon, pantry_tokens, forbidden_tokens)
            st.markdown("<hr>", unsafe_allow_html=True)
            res_col, nb_col = st.columns([1, 2.2])
            with res_col:
                render_result_cards(results)
            with nb_col:
                render_neighbor_panel(neighbors, pantry_tokens, forbidden_tokens)

# ── Batch tab ─────────────────────────────────────────────────────────────────
with tab_batch:
    batch_raw = st.text_area(
        "Ingredients to substitute",
        placeholder="doubanjiang\nsichuan_peppercorn\nmiso\nfish_sauce",
        height=160,
        help="One ingredient per line. Each gets top-3 substitutes from your pantry.",
    )
    st.caption("One ingredient per line. Runs the same substitution engine as Single mode.")

    run_batch = st.button("Find substitutes →", key="run_batch")

    if run_batch and batch_raw:
        ingredients = [l.strip() for l in batch_raw.splitlines() if l.strip()]
        for ing in ingredients:
            canon, status = check(ing)
            st.markdown(f"""
            <div style="font-family:'Cormorant Garamond',serif;font-size:22px;font-weight:600;
                        color:#f0e6c8;border-bottom:1px solid #2a2a20;padding-bottom:6px;margin:24px 0 12px">
              {ing}
              <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:400;
                           color:#5a5a40;margin-left:10px">
                {'→ ' + canon if canon and canon != ing else ('✗ not in vocab' if not canon else '')}
              </span>
            </div>
            """, unsafe_allow_html=True)

            if canon is None:
                st.markdown(f'<div style="color:#a05050;font-size:12px;margin-bottom:8px">✗ {ing!r} not in vocab — {status}</div>', unsafe_allow_html=True)
            else:
                results, _ = substitute(canon, pantry_tokens, forbidden_tokens)
                c1, c2, c3 = st.columns(3)
                for j, (col, result) in enumerate(zip([c1, c2, c3], results)):
                    render_result_cards([result], col=col, label_offset=j)
