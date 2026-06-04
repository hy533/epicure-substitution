import json
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

MODELS = ["Kaikaku/epicure-cooc", "Kaikaku/epicure-chem", "Kaikaku/epicure-core"]
CACHE = Path("scratch/cache")

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

def neighbors(ingredient, model, k=10):
    if ingredient not in vocab:
        return None
    idx = vocab[ingredient]
    sims = matrices[model] @ matrices[model][idx]
    top = np.argsort(sims)[::-1]
    results = []
    for i in top:
        if i == idx:
            continue
        results.append((idx_to_name[i], float(sims[i])))
        if len(results) == k:
            break
    return results

def sim(a, b, model):
    if a not in vocab or b not in vocab:
        return None
    ia, ib = vocab[a], vocab[b]
    return float(matrices[model][ia] @ matrices[model][ib])

K = 10

# ── Option 1: soy_sauce variants ──────────────────────────────────────────────
print("=" * 70)
print("OPTION 1: soy_sauce vs light_soy_sauce vs dark_soy_sauce")
print("=" * 70)

soy_variants = ["soy_sauce", "light_soy_sauce", "dark_soy_sauce"]

# Cross-similarity matrix (Chem)
print("\nCross-similarity (Chem):")
print(f"  {'':25}", end="")
for v in soy_variants:
    print(f"  {v:25}", end="")
print()
for a in soy_variants:
    print(f"  {a:25}", end="")
    for b in soy_variants:
        s = sim(a, b, "epicure-chem")
        print(f"  {s:25.3f}", end="")
    print()

print("\nChem neighbors — soy_sauce:")
for name, s in neighbors("soy_sauce", "epicure-chem", K):
    print(f"  {name:<40} {s:.3f}")

print("\nChem neighbors — light_soy_sauce:")
for name, s in neighbors("light_soy_sauce", "epicure-chem", K):
    print(f"  {name:<40} {s:.3f}")

print("\nChem neighbors — dark_soy_sauce:")
for name, s in neighbors("dark_soy_sauce", "epicure-chem", K):
    print(f"  {name:<40} {s:.3f}")

# L2 norms of raw (pre-normalized) embeddings as a proxy for density
raw_emb_path = hf_hub_download(repo_id="Kaikaku/epicure-chem", filename="embeddings.safetensors", cache_dir=CACHE)
raw_mat = load_file(raw_emb_path)["embeddings"].astype(np.float32)
print("\nRaw L2 norm (embedding magnitude — higher = better trained / more central):")
for v in soy_variants:
    idx = vocab[v]
    norm = float(np.linalg.norm(raw_mat[idx]))
    print(f"  {v:<30} norm={norm:.4f}")

# ── Option 2: gochugaru analysis ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("OPTION 2: gochugaru — chem vs cooc neighbors")
print("=" * 70)

print(f"\n{'CHEM':<45}  COOC")
print(f"  {'-'*43}  {'-'*43}")
chem_nb = neighbors("gochugaru", "epicure-chem", K)
cooc_nb = neighbors("gochugaru", "epicure-cooc", K)
for (cn, cs), (on, os) in zip(chem_nb, cooc_nb):
    print(f"  {cn:<43} {cs:.3f}    {on:<43} {os:.3f}")

print("\nSimilarity between gochugaru and gochujang:")
print(f"  Chem: {sim('gochugaru', 'gochujang', 'epicure-chem'):.3f}")
print(f"  Cooc: {sim('gochugaru', 'gochujang', 'epicure-cooc'):.3f}")

print("\nSimilarity between gochugaru and cheongyang_chili:")
print(f"  Chem: {sim('gochugaru', 'cheongyang_chili', 'epicure-chem'):.3f}")
print(f"  Cooc: {sim('gochugaru', 'cheongyang_chili', 'epicure-cooc'):.3f}")
