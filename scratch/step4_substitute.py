import json
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

CACHE = Path("scratch/cache")
MODELS = ["Kaikaku/epicure-cooc", "Kaikaku/epicure-chem", "Kaikaku/epicure-core"]

# ── Load ──────────────────────────────────────────────────────────────────────
matrices = {}
vocab = None
idx_to_name = None

for repo in MODELS:
    name = repo.split("/")[1]
    emb_path = hf_hub_download(repo_id=repo, filename="embeddings.safetensors", cache_dir=CACHE)
    vocab_path = hf_hub_download(repo_id=repo, filename="vocab.json", cache_dir=CACHE)
    tensors = load_file(emb_path)
    mat = tensors["embeddings"].astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    matrices[name] = mat / np.where(norms == 0, 1.0, norms)
    if vocab is None:
        with open(vocab_path) as f:
            vocab = json.load(f)
        idx_to_name = {v: k for k, v in vocab.items()}

# ── Alias map (D-002 and friends) ─────────────────────────────────────────────
ALIASES = {
    "soy_sauce": "light_soy_sauce",
}

def normalize(ingredient: str) -> str:
    return ALIASES.get(ingredient, ingredient)

# ── Core helpers ──────────────────────────────────────────────────────────────
def neighbors(ingredient: str, model: str, k: int = 50):
    if ingredient not in vocab:
        return None
    idx = vocab[ingredient]
    mat = matrices[model]
    sims = mat @ mat[idx]
    top = np.argsort(sims)[::-1]
    results = []
    for i in top:
        if i == idx:
            continue
        results.append((idx_to_name[i], float(sims[i])))
        if len(results) == k:
            break
    return results

def substitute(ingredient: str, available: set, forbidden: set = None, k: int = 50) -> dict:
    ingredient = normalize(ingredient)
    available_n  = {normalize(x) for x in available}
    forbidden_n  = {normalize(x) for x in forbidden} if forbidden else set()

    nb = neighbors(ingredient, "epicure-chem", k)
    if nb is None:
        return {"result": None, "sim": None, "constrained": False, "error": f"{ingredient!r} not in vocab"}

    for name, sim in nb:
        if name in available_n and name not in forbidden_n:
            return {"result": name, "sim": round(sim, 3), "constrained": True}

    # fallback: best unconstrained neighbor
    name, sim = nb[0]
    return {"result": name, "sim": round(sim, 3), "constrained": False}

# ── Tests ─────────────────────────────────────────────────────────────────────
def show(label, result):
    if result.get("error"):
        print(f"  {label:<55} → ERROR: {result['error']}")
        return
    flag = "✓" if result["constrained"] else "✗ (fallback)"
    print(f"  {label:<55} → {result['result']:<30} sim={result['sim']}  {flag}")

print("=" * 100)
print("TEST 1: allergen / forbidden set")
print("  Scenario: substitute doubanjiang; have fermented_black_bean, miso, gochujang, tomato_paste")
print("            forbidden: soy-based (soybean_paste, fermented_black_bean)")
print()
pantry = {"fermented_black_bean", "miso", "gochujang", "tomato_paste", "fish_sauce"}
allergens = {"soybean_paste", "fermented_black_bean"}
show("substitute(doubanjiang, pantry, forbidden=allergens)",
     substitute("doubanjiang", pantry, forbidden=allergens))

print()
print("TEST 2: available-only (no forbidden)")
print("  Scenario: substitute miso; have dashi, sake, soy_sauce (alias→light_soy_sauce), natto")
print()
pantry2 = {"dashi", "sake", "soy_sauce", "natto", "kombu"}
show("substitute(miso, pantry2)",
     substitute("miso", pantry2))

print()
print("TEST 3: alias normalization — soy_sauce in both ingredient and available")
print("  Scenario: substitute soy_sauce; have light_soy_sauce, dark_soy_sauce, fish_sauce")
print()
pantry3 = {"light_soy_sauce", "dark_soy_sauce", "fish_sauce"}
show("substitute(soy_sauce, pantry3)  [soy_sauce→light_soy_sauce alias]",
     substitute("soy_sauce", pantry3))

print()
print("TEST 4: constrained=False fallback — impossible pantry")
print("  Scenario: substitute doubanjiang; only have butter and cream")
print()
pantry4 = {"butter", "cream", "flour"}
show("substitute(doubanjiang, tiny_pantry={butter,cream,flour})",
     substitute("doubanjiang", pantry4))

print()
print("TEST 5: gochujang substitution with Korean pantry")
print("  Scenario: out of gochujang; have gochugaru, ssamjang, kimchi, miso")
print()
pantry5 = {"gochugaru", "ssamjang", "kimchi", "miso", "rice_vinegar"}
show("substitute(gochujang, korean_pantry)",
     substitute("gochujang", pantry5))

print()
print("TEST 6: out-of-vocab ingredient")
show("substitute(black_cod, pantry)",
     substitute("black_cod", {"salmon", "tuna"}))

print("=" * 100)
