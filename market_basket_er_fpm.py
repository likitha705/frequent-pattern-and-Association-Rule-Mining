"""
Frequent Pattern and Association Rule Mining at Scale via Graph-based
Entity Resolution and Data Deduplication, applied to Retail Market Basket
Analysis for Supermarket Chains.

Pipeline
--------
1. Load (or synthesize) raw transactions with messy product names.
2. Graph-based entity resolution: build a similarity graph over product
   names, take connected components as canonical product entities.
3. Deduplicate transactions (duplicate scans / repeated items / duplicate
   receipts) after mapping items to canonical entities.
4. Mine frequent itemsets with FP-Growth.
5. Generate association rules (support, confidence, lift).
6. Compare mining results with and without entity resolution.

Usage
-----
    python market_basket_er_fpm.py                      # synthetic data
    python market_basket_er_fpm.py --input data.csv     # your own data
    python market_basket_er_fpm.py --min-support 0.02 --min-confidence 0.4

Input CSV format: columns `transaction_id,item` (one row per item).
"""

import argparse
import os
import random
from difflib import SequenceMatcher

import networkx as nx
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# ----------------------------------------------------------------------
# 1. Data loading / synthetic generation
# ----------------------------------------------------------------------
CANONICAL_PRODUCTS = [
    "whole milk", "bread", "butter", "eggs", "cheese", "yogurt",
    "orange juice", "apple", "banana", "tomato", "onion", "potato",
    "rice", "wheat flour", "sugar", "tea powder", "coffee", "biscuits",
    "chocolate", "cooking oil", "salt", "chicken", "pasta", "tomato sauce",
]

# Items that tend to be bought together (to plant real patterns)
BUNDLES = [
    ["bread", "butter", "eggs"],
    ["whole milk", "tea powder", "sugar"],
    ["pasta", "tomato sauce", "cheese"],
    ["rice", "cooking oil", "onion", "tomato"],
    ["biscuits", "coffee", "chocolate"],
]


def make_typo(name: str, rng: random.Random) -> str:
    """Create a noisy variant of a product name (typo, case, spacing)."""
    variants = [
        name.upper(),
        name.title(),
        name.replace(" ", ""),
        name + " ",
        name + "s",
    ]
    if len(name) > 4:
        i = rng.randrange(1, len(name) - 1)
        variants.append(name[:i] + name[i + 1:])          # dropped char
        variants.append(name[:i] + name[i] + name[i:])    # doubled char
    return rng.choice(variants)


def generate_synthetic(n_transactions=5000, noise=0.35, dup_rate=0.05, seed=42):
    """Synthetic retail data with name noise and duplicate receipts."""
    rng = random.Random(seed)
    rows = []
    tid = 0
    for _ in range(n_transactions):
        tid += 1
        basket = set()
        if rng.random() < 0.7:
            basket.update(rng.choice(BUNDLES))
        basket.update(rng.sample(CANONICAL_PRODUCTS, rng.randint(1, 4)))
        for item in basket:
            shown = make_typo(item, rng) if rng.random() < noise else item
            rows.append((tid, shown))
            if rng.random() < 0.05:                       # duplicate scan
                rows.append((tid, shown))
        if rng.random() < dup_rate:                       # duplicate receipt
            tid += 1
            for item in basket:
                rows.append((tid, item))
    return pd.DataFrame(rows, columns=["transaction_id", "item"])


def load_data(path):
    df = pd.read_csv(path)
    missing = {"transaction_id", "item"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    return df[["transaction_id", "item"]].dropna()


# ----------------------------------------------------------------------
# 2. Graph-based entity resolution
# ----------------------------------------------------------------------
def normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower().strip() if ch.isalnum() or ch == " ")


def resolve_entities(items, threshold=0.85):
    """
    Build a graph where nodes are distinct raw item names and an edge links
    two names whose similarity >= threshold. Each connected component is one
    real-world product entity; its most frequent name becomes canonical.
    Returns: dict raw_name -> canonical_name, and the graph.
    """
    counts = pd.Series(items).value_counts()
    raw_names = list(counts.index)
    norm = {r: normalize(r) for r in raw_names}

    G = nx.Graph()
    G.add_nodes_from(raw_names)

    # Blocking: only compare names sharing the same first letter to avoid
    # O(n^2) over the whole catalogue (scales to large product lists).
    blocks = {}
    for r in raw_names:
        key = norm[r][:1] if norm[r] else ""
        blocks.setdefault(key, []).append(r)

    for block in blocks.values():
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                a, b = block[i], block[j]
                if norm[a] == norm[b] or norm[a].replace(" ", "") == norm[b].replace(" ", ""):
                    sim = 1.0
                else:
                    sim = SequenceMatcher(None, norm[a], norm[b]).ratio()
                if sim >= threshold:
                    G.add_edge(a, b, weight=sim)

    mapping = {}
    for comp in nx.connected_components(G):
        canonical = max(comp, key=lambda n: counts[n])
        canonical = normalize(canonical).strip()
        for n in comp:
            mapping[n] = canonical
    return mapping, G


# ----------------------------------------------------------------------
# 3. Deduplication
# ----------------------------------------------------------------------
def deduplicate(df):
    """Remove repeated items within a basket, then duplicate receipts."""
    before_rows = len(df)
    df = df.drop_duplicates(["transaction_id", "item"])
    baskets = df.groupby("transaction_id")["item"].apply(lambda s: tuple(sorted(s)))
    baskets = baskets.drop_duplicates()   # identical basket seen twice = dup receipt
    out = baskets.explode().reset_index()
    out.columns = ["transaction_id", "item"]
    print(f"  dedup: {before_rows} rows -> {len(out)} rows, "
          f"{baskets.shape[0]} unique baskets")
    return out


# ----------------------------------------------------------------------
# 4/5. Frequent pattern mining + rules
# ----------------------------------------------------------------------
def mine(df, min_support, min_confidence):
    baskets = df.groupby("transaction_id")["item"].apply(list).tolist()
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit(baskets).transform(baskets, sparse=False),
                          columns=te.columns_)
    freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return freq, pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    return freq, rules


def format_rules(rules):
    r = rules.copy()
    r["antecedents"] = r["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    r["consequents"] = r["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return r[["antecedents", "consequents", "support", "confidence", "lift"]]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="CSV with transaction_id,item (default: synthetic)")
    p.add_argument("--transactions", type=int, default=5000,
                   help="synthetic transaction count")
    p.add_argument("--sim-threshold", type=float, default=0.85,
                   help="name similarity threshold for graph edges")
    p.add_argument("--min-support", type=float, default=0.02)
    p.add_argument("--min-confidence", type=float, default=0.4)
    p.add_argument("--out", default="output", help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("[1] Loading data")
    raw = load_data(args.input) if args.input else generate_synthetic(args.transactions)
    print(f"  {len(raw)} rows, {raw['transaction_id'].nunique()} transactions, "
          f"{raw['item'].nunique()} distinct raw item names")

    print("[2] Graph-based entity resolution")
    mapping, G = resolve_entities(raw["item"], args.sim_threshold)
    n_entities = len(set(mapping.values()))
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  {G.number_of_nodes()} raw names -> {n_entities} resolved entities")
    pd.DataFrame(sorted(mapping.items()), columns=["raw_name", "canonical"]) \
        .to_csv(f"{args.out}/entity_mapping.csv", index=False)

    print("[3] Deduplication")
    resolved = raw.assign(item=raw["item"].map(mapping))
    clean = deduplicate(resolved)

    print("[4] Mining WITHOUT entity resolution (baseline)")
    base_freq, base_rules = mine(deduplicate(raw), args.min_support, args.min_confidence)
    print(f"  {len(base_freq)} frequent itemsets, {len(base_rules)} rules")

    print("[5] Mining WITH entity resolution + dedup")
    freq, rules = mine(clean, args.min_support, args.min_confidence)
    print(f"  {len(freq)} frequent itemsets, {len(rules)} rules")

    if not rules.empty:
        out = format_rules(rules)
        out.to_csv(f"{args.out}/association_rules.csv", index=False)
        print("\nTop 10 rules by lift:")
        print(out.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    freq.assign(itemsets=freq["itemsets"].apply(lambda s: ", ".join(sorted(s)))) \
        .to_csv(f"{args.out}/frequent_itemsets.csv", index=False)

    print(f"\nResults saved in ./{args.out}/")


if __name__ == "__main__":
    main()
