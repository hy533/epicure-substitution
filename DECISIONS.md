# Design Decisions Log

Exploration session: Epicure substitution engine.
Each entry has the choice, the evidence behind it, and how to revert.

---

## D-001: Use Chem embeddings as the substitution model

**Choice:** Build substitution on `epicure-chem` (shared flavor molecules), not `epicure-cooc` (recipe co-occurrence) or `epicure-core` (blend).

**Evidence:** Side-by-side comparison of Chem vs Cooc neighbors for 7 target ingredients showed Cooc returns cook-with companions (e.g. doubanjiang Cooc → `light_soy_sauce, oyster_sauce, shaoxing_wine` — the Sichuan pantry backbone), while Chem returns flavor-relatives that are plausible substitutes (e.g. doubanjiang Chem → `soybean_paste, fermented_black_bean, mala_sauce`). Cooc is reserved for a future "what completes this dish" layer.

**Revert:** Swap `"epicure-chem"` for `"epicure-cooc"` (or `"epicure-core"`) in any call to `neighbors()` / the substitution function. No other changes needed — vocabs and matrix shapes are identical across all three models.

---

## D-002: `soy_sauce` is treated as an orphan token; normalize to `light_soy_sauce`

**Choice:** When a user inputs `soy_sauce`, the normalization layer maps it to `light_soy_sauce` before querying the embedding space.

**Evidence:**
- Cross-similarity (Chem): `soy_sauce` ↔ `light_soy_sauce` = **0.236**, `soy_sauce` ↔ `dark_soy_sauce` = **0.241**, `light_soy_sauce` ↔ `dark_soy_sauce` = **0.710**. The two specific variants form a tight cluster; generic `soy_sauce` is a distant outlier.
- Raw L2 norms: `soy_sauce` = 2.54, `light_soy_sauce` = 1.66, `dark_soy_sauce` = 1.84. Higher norm on the generic token indicates it is not undertrained — it genuinely occupies a different region (probably anchored to Western-recipe usage).
- Chem neighbors of `soy_sauce`: vegetables and neutral umami (`scallion`, `bok_choy`, `enoki_mushroom`) — not fermented condiments. Chem neighbors of `light_soy_sauce`: `shaoxing_wine`, `oyster_sauce`, `doubanjiang`, `sichuan_peppercorn` — the correct substitution cluster.

**Revert:** Remove the `soy_sauce → light_soy_sauce` alias from the normalization map. Users who type `soy_sauce` will land in the vegetable/neutral-umami neighborhood instead of the fermented-condiment cluster. Consider replacing with a disambiguation prompt ("did you mean light or dark soy sauce?") as an alternative to hard aliasing.

---

## D-003: Try raw string before underscore-normalizing

**Choice:** In `check()`, attempt a vocab lookup on the raw user string (spaces preserved) before replacing spaces with underscores.

**Evidence:** `"soy sauce"` (with a space) resolves as an exact vocab match — the corpus uses mixed token forms, not exclusively underscore-separated. Normalizing spaces first would silently miss these tokens. The correct order is: raw → alias → underscore-normalized → prefix-stripped → fuzzy.

**Revert:** Remove the raw-string lookup step and always underscore-normalize first. Side effect: `"soy sauce"` and any other space-separated tokens will fall through to fuzzy matching instead of resolving exactly.

---

## D-004: Strip common suffixes in normalization (`_cheese`, `_pepper`, `_chile`/`_chili`)

**Choice:** After prefix stripping, also try stripping known suffixes before falling through to fuzzy matching.

**Evidence:** `"cheddar"` → closest vocab match is `cheddar_cheese`; `"parmesan"` → `parmesan_cheese`; `"ancho_chili"` → `ancho_chile`. These are deterministic suffix/spelling variants that fuzzy matching handles inconsistently (e.g. `"ancho_chili"` got `ancho_chile` but also noise candidates). A suffix strip pass resolves them cleanly.

**Suffix map:**
- `{bare}_cheese` → try `{bare}_cheese` in vocab when user inputs `{bare}`
- `{bare}_pepper` → try `{bare}_pepper`
- `_chili` ↔ `_chile` spelling swap

**Revert:** Remove the suffix expansion step. `"cheddar"`, `"parmesan"`, `"ancho_chili"` etc. will fall to fuzzy matching — which may or may not return the right candidate depending on edit distance.
