# Substitution Function Design

**Date:** 2026-06-03
**Scope:** Exploration script — throwaway, single file.

---

## What it does

Given a target ingredient and a pantry (available set), returns the closest
flavor-relative (Chem embedding neighbor) that exists in the pantry and is not
forbidden. Falls back to the best unconstrained Chem neighbor if the constraint
cannot be satisfied.

---

## Signature

```python
substitute(ingredient, available, forbidden=None, k=50) -> dict
```

| Param | Type | Description |
|---|---|---|
| `ingredient` | `str` | Ingredient to substitute. Alias-normalized before lookup. |
| `available` | `set[str]` | Pantry / store list. Alias-normalized before set check. |
| `forbidden` | `set[str] \| None` | Allergen / exclude list. Optional. |
| `k` | `int` | How many Chem neighbors to scan. Default 50. |

**Returns:** `{"result": str, "sim": float, "constrained": bool}`

- `constrained=True` — a neighbor satisfying the available+forbidden filter was found.
- `constrained=False` — no match in available set; top-1 unconstrained Chem neighbor returned as best-effort.

---

## Logic

1. Normalize `ingredient` through alias map (D-002: `soy_sauce → light_soy_sauce`).
2. Normalize all strings in `available` and `forbidden` through the same alias map.
3. Fetch top-k Chem neighbors via `neighbors(ingredient, "epicure-chem", k)`.
4. Walk neighbors in rank order; return first that is in `available` and not in `forbidden`.
5. If none found, return top-1 unconstrained neighbor with `constrained=False`.

---

## Decisions baked in

- **D-001:** Chem model for substitution (not Cooc or Core).
- **D-002:** `soy_sauce → light_soy_sauce` alias in normalization.
- Top-1 return only (ranked list deferred).
- k=50 default gives the filter enough candidates without scanning the whole vocab.
