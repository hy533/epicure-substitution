# Epicure Substitution Engine

An ingredient substitution tool built on the [Epicure](https://huggingface.co/Kaikaku) flavor embeddings. Given an ingredient and a pantry, it returns the closest flavor-compatible substitute available to you — powered by shared flavor-molecule vectors, not recipe co-occurrence.

---

## How it works

Epicure provides three sibling embedding models, each a `(1790, 300)` float32 matrix of ingredient vectors trained with metapath2vec skip-gram:

| Model | HuggingFace ID | What it captures |
|---|---|---|
| **Chem** | `Kaikaku/epicure-chem` | Shared flavor molecules (FlavorDB). Ingredients close in this space taste alike. |
| **Cooc** | `Kaikaku/epicure-cooc` | Recipe co-occurrence. Ingredients close here are cooked *together*, not interchangeable. |
| **Core** | `Kaikaku/epicure-core` | Blend of Chem + Cooc. |

**This engine uses Chem only for substitution.** Cooc is intentionally reserved for a future "what completes this dish" layer. See [DECISIONS.md](DECISIONS.md) for the full rationale.

All embeddings are L2-normalized before any cosine operation.

---

## Quickstart

```bash
# Install dependencies
pip install huggingface_hub safetensors numpy streamlit

# Run the UI (downloads embeddings on first launch, ~5s)
streamlit run scratch/app.py
```

The embeddings are cached locally in `scratch/cache/` after the first download.

---

## The Streamlit UI

`scratch/app.py` — the main testing interface.

**Fields:**

| Field | What to enter |
|---|---|
| **Ingredient** | The ingredient you want to substitute. Use underscores for spaces (`fish_sauce`), though spaces also work — the engine normalizes automatically. |
| **Forbidden / allergens** | Comma-separated list of ingredients to exclude from results (allergens, dietary restrictions). Leave blank if none. |
| **Pantry** | Comma-separated list of what you have available. The engine returns the closest Chem neighbor that appears in this list. |

**Vocab check:** As you type in the Ingredient field, the engine tells you immediately whether your input is an exact vocab match, an alias (e.g. `soy_sauce → light_soy_sauce`), or missing with suggestions.

**Results:**
- **Best substitute** — the top Chem neighbor that is in your pantry and not forbidden. If no pantry item qualifies, falls back to the unconstrained top-1 and flags it as a fallback.
- **Chem neighbors list** — all top-30 neighbors ranked by cosine similarity, color-coded:
  - 🟢 `pantry` — in your available set
  - 🔴 `forbidden` — excluded (struck through)
  - dim — not in pantry, not forbidden

---

## Exploration scripts

These are the session scripts used to validate the approach. Run them from the project root.

| Script | What it does |
|---|---|
| `scratch/step1_load.py` | Downloads all three models, confirms shapes `(1790, 300)` and that vocabs are identical across siblings. |
| `scratch/step2_neighbors.py` | Defines the `neighbors(ingredient, model, k)` helper. Runs a side-by-side Chem vs Cooc comparison for 7 target ingredients to validate the Chem/Cooc distinction. |
| `scratch/step2b_analyze.py` | Deeper analysis: `soy_sauce` orphan token discovery, `gochugaru` cluster analysis. The evidence behind D-002 and the gochujang/gochugaru substitution cluster. |
| `scratch/step4_substitute.py` | The substitution function: `substitute(ingredient, available, forbidden, k)`. Includes 6 test cases covering allergen filtering, alias normalization, impossible pantries, and out-of-vocab inputs. |
| `scratch/step5_vocab.py` | Vocab gap stress test. The `check(ingredient)` helper tries raw → alias → underscore → prefix-strip → suffix-expand → fuzzy. Tests against session ingredients and a broader home-pantry list. |

---

## Vocabulary

- **1,790 canonical ingredients**, snake_case, shared identically across all three models.
- Corpus is **East-Asian heavy** (Chinese, Japanese, Korean coverage is excellent). South Asian, Latin American, and Western pantries are present but sparser in embedding quality.
- Common user inputs that are **not in vocab**: `green_onion`, `spring_onion`, `heavy_cream`, `hing`, `dried_shiitake` (as a compound — `shiitake_mushroom` is in vocab).

### Normalization pipeline

When a user types an ingredient, the engine tries these steps in order before giving up:

1. Raw string lookup (vocab has some space-separated tokens)
2. Alias map (`soy_sauce → light_soy_sauce`)
3. Underscore/hyphen normalization
4. Prefix stripping (`fresh_`, `dried_`, `ground_`, `whole_`, `raw_`, `roasted_`)
5. Suffix expansion (`_cheese`, `_pepper`, `_mushroom`, `_sauce`, `_oil`)
6. `_chili` ↔ `_chile` spelling swap
7. Fuzzy match (difflib, cutoff 0.6) with up to 3 suggestions

---

## Design decisions

All design choices made during exploration — with evidence and revert instructions — are documented in [DECISIONS.md](DECISIONS.md).

| ID | Decision |
|---|---|
| D-001 | Use Chem embeddings for substitution (not Cooc or Core) |
| D-002 | `soy_sauce` aliased to `light_soy_sauce` (orphan token) |
| D-003 | Try raw string before underscore-normalizing |
| D-004 | Strip common suffixes (`_cheese`, `_pepper`, `_chile`/`_chili`) |

---

## Known limitations

- **Vocab gaps**: only 1,790 ingredients. Very common items like `green_onion`, `heavy_cream`, `hing` are missing. The normalization pipeline handles common variants but can't bridge true gaps.
- **Fallback quality**: when no pantry item qualifies, the fallback is the unconstrained top-1 Chem neighbor — which is usually not in your pantry. A ranked list return (deferred) would let the caller find the best available option further down the list.
- **`soy_sauce` is an orphan**: the generic token sits in a different embedding region from `light_soy_sauce`/`dark_soy_sauce`. The alias map handles this for input, but user-supplied pantry items typed as `soy_sauce` are also remapped — which may not always be what they mean.
- **Corpus bias**: embedding quality is uneven across cuisines. Substitutions within East-Asian flavor profiles are well-supported; others may be noisier.

---

## Spec and design docs

- `docs/superpowers/specs/2026-06-03-substitution-function-design.md` — substitution function spec (approved design before implementation)
- `DECISIONS.md` — all design decisions with evidence and revert paths
