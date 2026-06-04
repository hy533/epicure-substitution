import json
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

MODELS = ["Kaikaku/epicure-cooc", "Kaikaku/epicure-chem", "Kaikaku/epicure-core"]
CACHE = Path("scratch/cache")

# --- Load all three ---
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
    norms = np.where(norms == 0, 1.0, norms)
    matrices[name] = mat / norms

    if vocab is None:
        with open(vocab_path) as f:
            vocab = json.load(f)
        idx_to_name = {v: k for k, v in vocab.items()}

def neighbors(ingredient: str, model: str, k: int = 10):
    if ingredient not in vocab:
        return None
    idx = vocab[ingredient]
    mat = matrices[model]
    sims = mat @ mat[idx]          # cosine sim (all rows are L2-normalized)
    top = np.argsort(sims)[::-1]   # descending
    results = []
    for i in top:
        if i == idx:
            continue
        results.append((idx_to_name[i], float(sims[i])))
        if len(results) == k:
            break
    return results

# --- Step 3: side-by-side Chem vs Cooc ---
TARGETS = [
    "doubanjiang",
    "sichuan_peppercorn",
    "shaoxing_wine",
    "miso",
    "fish_sauce",
    "gochujang",
    "soy_sauce",
]

K = 8
COL = 30

print(f"{'Ingredient':<20}  {'CHEM neighbors (flavor-relatives)':<{K*COL//2}}  COOC neighbors (cook-with companions)")
print("=" * 140)

for ing in TARGETS:
    if ing not in vocab:
        print(f"{ing:<20}  *** NOT IN VOCAB ***")
        continue

    chem_nb = neighbors(ing, "epicure-chem", K)
    cooc_nb = neighbors(ing, "epicure-cooc", K)

    print(f"\n>>> {ing}")
    print(f"  {'CHEM':<65}  COOC")
    print(f"  {'-'*63}  {'-'*63}")
    for i in range(K):
        c_name, c_sim = chem_nb[i]
        o_name, o_sim = cooc_nb[i]
        print(f"  {c_name:<45} {c_sim:.3f}    {o_name:<45} {o_sim:.3f}")
