import json
import numpy as np
from pathlib import Path
from difflib import get_close_matches
from huggingface_hub import hf_hub_download

CACHE = Path("scratch/cache")

vocab_path = hf_hub_download(repo_id="Kaikaku/epicure-chem", filename="vocab.json", cache_dir=CACHE)
with open(vocab_path) as f:
    vocab = json.load(f)

ALL_VOCAB = set(vocab.keys())

# Same alias map as step 4
ALIASES = {
    "soy_sauce": "light_soy_sauce",
}

def check(ingredient: str) -> dict:
    """
    Returns:
      status  : "exact" | "alias" | "fuzzy" | "missing"
      canonical: the vocab token to actually use (or None)
      suggestions: close matches if missing
    """
    # D-003: try raw string first (vocab has space-separated tokens too)
    raw_space = ingredient.strip().lower()
    if raw_space in ALL_VOCAB:
        return {"status": "exact", "canonical": raw_space, "suggestions": []}

    raw = raw_space.replace(" ", "_").replace("-", "_")

    if raw in ALL_VOCAB:
        return {"status": "exact", "canonical": raw, "suggestions": []}

    if raw in ALIASES:
        canon = ALIASES[raw]
        return {"status": "alias", "canonical": canon, "suggestions": []}

    # prefix stripping
    for prefix in ("fresh_", "dried_", "ground_", "whole_", "raw_", "roasted_"):
        stripped = raw.removeprefix(prefix)
        if stripped in ALL_VOCAB:
            return {"status": "fuzzy", "canonical": stripped, "suggestions": []}

    # D-004: suffix expansion — try adding common suffixes the user omitted
    for suffix in ("_cheese", "_pepper", "_mushroom", "_sauce", "_oil"):
        expanded = raw + suffix
        if expanded in ALL_VOCAB:
            return {"status": "fuzzy", "canonical": expanded, "suggestions": []}

    # D-004: _chili / _chile spelling swap
    if raw.endswith("_chili"):
        alt = raw[:-6] + "_chile"
        if alt in ALL_VOCAB:
            return {"status": "fuzzy", "canonical": alt, "suggestions": []}
    if raw.endswith("_chile"):
        alt = raw[:-6] + "_chili"
        if alt in ALL_VOCAB:
            return {"status": "fuzzy", "canonical": alt, "suggestions": []}

    suggestions = get_close_matches(raw, ALL_VOCAB, n=3, cutoff=0.6)
    return {"status": "missing", "canonical": None, "suggestions": suggestions}

def show_check(ingredient: str):
    r = check(ingredient)
    if r["status"] == "exact":
        print(f"  ✓  {ingredient:<35} exact")
    elif r["status"] == "alias":
        print(f"  ~  {ingredient:<35} alias  → {r['canonical']}")
    elif r["status"] == "fuzzy":
        print(f"  ~  {ingredient:<35} fuzzy  → {r['canonical']}")
    else:
        sug = ", ".join(r["suggestions"]) if r["suggestions"] else "no close match"
        print(f"  ✗  {ingredient:<35} MISSING  (closest: {sug})")

# ── Ingredients from this session ─────────────────────────────────────────────
SESSION_INGREDIENTS = [
    "doubanjiang", "sichuan_peppercorn", "shaoxing_wine", "miso",
    "fish_sauce", "gochujang", "soy_sauce", "gochugaru",
    "light_soy_sauce", "dark_soy_sauce", "black_cod",
]

# ── A broader stress-test: common ingredients a home cook might try ────────────
STRESS_TEST = [
    # exact matches expected
    "garlic", "ginger", "scallion", "sesame_oil", "rice_vinegar",
    # plausible but possibly missing
    "green_onion", "spring_onion", "white_pepper", "black_pepper",
    "msg", "dashi", "bonito_flakes", "kombu",
    # likely missing (Western pantry)
    "butter", "heavy_cream", "cheddar", "parmesan",
    # likely missing (South Asian)
    "ghee", "paneer", "tamarind", "hing", "asafoetida",
    # likely missing (Latin American)
    "chipotle", "ancho_chili", "epazote", "masa_harina",
    # common user typos / alternate forms
    "soy sauce",          # space instead of underscore
    "scallions",          # plural
    "fresh_ginger",       # with qualifier
    "dried_shiitake",     # with qualifier
]

print("=" * 70)
print("SESSION INGREDIENTS")
print("=" * 70)
for ing in SESSION_INGREDIENTS:
    show_check(ing)

print()
print("=" * 70)
print("STRESS TEST — home pantry coverage")
print("=" * 70)
for ing in STRESS_TEST:
    show_check(ing)

print()
print("=" * 70)
print("VOCAB STATS")
print("=" * 70)
print(f"  Total vocab size: {len(ALL_VOCAB)}")

# spot-check coverage by cuisine region using keyword heuristics
regions = {
    "Chinese":      ["doubanjiang", "shaoxing", "sichuan", "baijiu", "five_spice", "oyster_sauce"],
    "Japanese":     ["miso", "dashi", "mirin", "sake", "bonito", "kombu", "ponzu"],
    "Korean":       ["gochujang", "gochugaru", "kimchi", "ssamjang", "doenjang"],
    "Southeast Asian": ["fish_sauce", "lemongrass", "galangal", "kaffir", "nam_prik"],
    "South Asian":  ["ghee", "paneer", "tamarind", "cumin", "turmeric", "cardamom"],
    "Latin American": ["chipotle", "epazote", "masa", "tomatillo", "achiote"],
    "Western":      ["butter", "cream", "parmesan", "cheddar", "thyme", "rosemary"],
}

print()
print("  Spot-check by region (keywords found in vocab):")
for region, keywords in regions.items():
    found = [k for k in keywords if any(k in v for v in ALL_VOCAB)]
    print(f"    {region:<20} {len(found)}/{len(keywords)} keywords present: {found}")
