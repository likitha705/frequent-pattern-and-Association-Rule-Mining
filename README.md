# Frequent Pattern and Association Rule Mining at Scale via Graph-based Entity Resolution and Data Deduplication

**Application:** Retail Market Basket Analysis for Supermarket Chains
**Course:** Data Mining & Warehousing (final project)

## Overview

Supermarket transaction data is messy. The same product shows up as `Whole Milk`, `whole milk`, `WHOLEMILK` or `whole mik`, and receipts get scanned or recorded twice. Mined directly, this noise splits the support of a product across many spellings, so real buying patterns are missed and near-duplicate rules appear.

This project cleans the data first and mines afterwards:

1. **Graph-based entity resolution:** product names become nodes, and names with similarity above a threshold are linked by edges. Each connected component is one real product. The most frequent spelling becomes its canonical name.
2. **Deduplication:** repeated scans within a basket and duplicate receipts are removed.
3. **Frequent pattern mining:** FP-Growth finds frequent itemsets.
4. **Association rules:** rules are ranked by support, confidence and lift.
5. **Comparison:** results are shown with and without entity resolution.

To keep entity resolution scalable, names are compared only inside blocks (same first letter) instead of across the whole catalogue.

## Project structure

```
.
├── market_basket_er_fpm.py   # core pipeline (also runnable from the command line)
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Usage

### Dashboard

```bash
streamlit run app.py
```

Choose synthetic data or upload a CSV, set the similarity threshold, minimum support and minimum confidence in the sidebar, then click **Run pipeline**. The dashboard shows:

- summary metrics (rows removed, entities merged, rules found)
- association rules table with a lift filter and a support/confidence chart
- frequent itemsets
- resolved product entities and their name variants
- comparison with and without entity resolution

CSV downloads are available for rules and the entity mapping.

### Command line

```bash
python market_basket_er_fpm.py                       # synthetic data
python market_basket_er_fpm.py --input data.csv      # your own data
python market_basket_er_fpm.py --min-support 0.02 --min-confidence 0.4 --sim-threshold 0.85
```

Results are saved in the `output/` folder (`association_rules.csv`, `frequent_itemsets.csv`, `entity_mapping.csv`).

## Input format

A CSV with one row per purchased item:

| transaction_id | item       |
|----------------|------------|
| 1              | Whole Milk |
| 1              | bread      |
| 2              | whole milk |

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--sim-threshold` | 0.85 | Minimum name similarity to link two names in the graph |
| `--min-support` | 0.02 | Minimum fraction of transactions containing an itemset |
| `--min-confidence` | 0.4 | Minimum confidence for a rule |

## Technologies

Python, pandas, NetworkX, mlxtend (FP-Growth), Streamlit

## Author

Gonepalli Likitha (23ME1A5420)
