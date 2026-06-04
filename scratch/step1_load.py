import json
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

MODELS = ["Kaikaku/epicure-cooc", "Kaikaku/epicure-chem", "Kaikaku/epicure-core"]
CACHE = Path("scratch/cache")
CACHE.mkdir(parents=True, exist_ok=True)

matrices = {}
vocabs = {}

for repo in MODELS:
    name = repo.split("/")[1]
    print(f"\n--- {name} ---")

    emb_path = hf_hub_download(repo_id=repo, filename="embeddings.safetensors", cache_dir=CACHE)
    vocab_path = hf_hub_download(repo_id=repo, filename="vocab.json", cache_dir=CACHE)

    tensors = load_file(emb_path)
    key = list(tensors.keys())[0]
    mat = tensors[key].astype(np.float32)

    with open(vocab_path) as f:
        vocab = json.load(f)

    # L2-normalize each row
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_norm = mat / norms

    matrices[name] = mat_norm
    vocabs[name] = vocab

    print(f"  tensor key:  {key}")
    print(f"  shape:       {mat.shape}")
    print(f"  dtype:       {mat.dtype}")
    print(f"  vocab size:  {len(vocab)}")
    print(f"  sample keys: {list(vocab.items())[:5]}")

# Verify vocabs are identical across all three
names = list(vocabs.keys())
v0 = set(vocabs[names[0]].keys())
v1 = set(vocabs[names[1]].keys())
v2 = set(vocabs[names[2]].keys())

print("\n--- Vocab comparison ---")
print(f"  cooc == chem: {v0 == v1}")
print(f"  cooc == core: {v0 == v2}")
if v0 != v1:
    print(f"  cooc-chem diff: {v0.symmetric_difference(v1)}")
if v0 != v2:
    print(f"  cooc-core diff: {v0.symmetric_difference(v2)}")

# Verify indices are also identical (not just keys)
idx_match_01 = all(vocabs[names[0]][k] == vocabs[names[1]][k] for k in v0 & v1)
idx_match_02 = all(vocabs[names[0]][k] == vocabs[names[1]][k] for k in v0 & v2)
print(f"  indices match cooc==chem: {idx_match_01}")
print(f"  indices match cooc==core: {idx_match_02}")

print("\nAll three models loaded and L2-normalized. Ready for step 2.")
