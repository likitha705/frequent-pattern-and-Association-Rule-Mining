"""
Streamlit dashboard for:
Frequent Pattern and Association Rule Mining at Scale via Graph-based
Entity Resolution and Data Deduplication (Retail Market Basket Analysis).

Run:
    streamlit run app.py

Keep this file in the same folder as market_basket_er_fpm.py.
"""

import pandas as pd
import streamlit as st

from market_basket_er_fpm import (
    deduplicate,
    format_rules,
    generate_synthetic,
    load_data,
    mine,
    resolve_entities,
)

st.set_page_config(page_title="Market Basket Mining", layout="wide")
st.title("Market Basket Analysis")
st.caption(
    "Graph-based entity resolution + deduplication, then FP-Growth "
    "association rule mining for supermarket transactions."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Synthetic data", "Upload CSV"])
    uploaded = None
    n_tx = 5000
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "CSV with columns: transaction_id, item", type="csv"
        )
    else:
        n_tx = st.slider("Number of transactions", 1000, 50000, 5000, step=1000)

    st.header("Entity resolution")
    sim = st.slider("Name similarity threshold", 0.50, 1.00, 0.85, 0.01)

    st.header("Mining")
    min_sup = st.slider("Minimum support", 0.005, 0.20, 0.02, 0.005)
    min_conf = st.slider("Minimum confidence", 0.10, 0.95, 0.40, 0.05)

    run = st.button("Run pipeline", type="primary", use_container_width=True)


# ---------------------------------------------------------------- pipeline
@st.cache_data(show_spinner=False)
def run_pipeline(raw: pd.DataFrame, sim, min_sup, min_conf):
    mapping, G = resolve_entities(raw["item"], sim)
    resolved = raw.assign(item=raw["item"].map(mapping))
    clean = deduplicate(resolved)

    base_freq, base_rules = mine(deduplicate(raw), min_sup, min_conf)
    freq, rules = mine(clean, min_sup, min_conf)

    return {
        "mapping": mapping,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "clean_rows": len(clean),
        "clean_tx": clean["transaction_id"].nunique(),
        "base_freq": base_freq,
        "base_rules": base_rules,
        "freq": freq,
        "rules": rules,
    }


if not run and "result" not in st.session_state:
    st.info("Set the options in the sidebar and click **Run pipeline**.")
    st.stop()

if run:
    if source == "Upload CSV":
        if uploaded is None:
            st.error("Please upload a CSV file first.")
            st.stop()
        try:
            raw = load_data(uploaded)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
            st.stop()
    else:
        raw = generate_synthetic(n_tx)

    with st.spinner("Resolving entities, deduplicating and mining..."):
        result = run_pipeline(raw, sim, min_sup, min_conf)
    st.session_state["result"] = result
    st.session_state["raw"] = raw

result = st.session_state["result"]
raw = st.session_state["raw"]

# ---------------------------------------------------------------- metrics
n_raw_names = raw["item"].nunique()
n_entities = len(set(result["mapping"].values()))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw rows", f"{len(raw):,}", f"{len(raw) - result['clean_rows']:,} removed", delta_color="off")
c2.metric("Transactions after dedup", f"{result['clean_tx']:,}", f"{raw['transaction_id'].nunique() - result['clean_tx']:,} removed", delta_color="off")
c3.metric("Distinct item names", f"{n_raw_names:,}", f"-> {n_entities:,} entities", delta_color="off")
c4.metric("Rules found", f"{len(result['rules']):,}", f"{len(result['rules']) - len(result['base_rules']):+,} vs baseline", delta_color="off")

tab_rules, tab_items, tab_er, tab_cmp = st.tabs(
    ["Association rules", "Frequent itemsets", "Entity resolution", "With vs without ER"]
)

# ---------------------------------------------------------------- rules
with tab_rules:
    rules = result["rules"]
    if rules.empty:
        st.warning("No rules found. Try lowering minimum support or confidence.")
    else:
        view = format_rules(rules).reset_index(drop=True)
        min_lift = st.slider("Minimum lift", 1.0, float(max(2.0, view["lift"].max())), 1.0, 0.1)
        view = view[view["lift"] >= min_lift]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download rules (CSV)",
            view.to_csv(index=False).encode(),
            "association_rules.csv",
            "text/csv",
        )
        st.subheader("Support vs confidence")
        st.scatter_chart(view, x="support", y="confidence", size="lift")

# ---------------------------------------------------------------- itemsets
with tab_items:
    freq = result["freq"]
    if freq.empty:
        st.warning("No frequent itemsets at this support.")
    else:
        f = freq.copy()
        f["itemsets"] = f["itemsets"].apply(lambda s: ", ".join(sorted(s)))
        f["size"] = freq["itemsets"].apply(len)
        f = f.sort_values("support", ascending=False).reset_index(drop=True)
        st.dataframe(f, use_container_width=True, hide_index=True)
        st.subheader("Top 15 itemsets by support")
        st.bar_chart(f.head(15).set_index("itemsets")["support"])

# ---------------------------------------------------------------- ER
with tab_er:
    st.write(
        f"Similarity graph: **{result['n_nodes']}** nodes (raw names), "
        f"**{result['n_edges']}** edges. Each connected component is one product entity."
    )
    m = pd.DataFrame(sorted(result["mapping"].items()), columns=["raw_name", "canonical"])
    groups = (
        m.groupby("canonical")["raw_name"]
        .apply(lambda s: " | ".join(sorted(s)))
        .reset_index(name="raw name variants")
    )
    groups["variants"] = m.groupby("canonical")["raw_name"].count().values
    groups = groups.sort_values("variants", ascending=False)
    only_merged = st.checkbox("Show only merged entities (2+ variants)", value=True)
    if only_merged:
        groups = groups[groups["variants"] > 1]
    st.dataframe(groups, use_container_width=True, hide_index=True)
    st.download_button(
        "Download entity mapping (CSV)",
        m.to_csv(index=False).encode(),
        "entity_mapping.csv",
        "text/csv",
    )

# ---------------------------------------------------------------- compare
with tab_cmp:
    cmp_df = pd.DataFrame(
        {
            "Without ER (dedup only)": [
                n_raw_names,
                len(result["base_freq"]),
                len(result["base_rules"]),
            ],
            "With ER + dedup": [n_entities, len(result["freq"]), len(result["rules"])],
        },
        index=["Distinct items", "Frequent itemsets", "Association rules"],
    )
    st.dataframe(cmp_df, use_container_width=True)
    st.caption(
        "Resolving name variants merges split counts, so true patterns reach the "
        "support threshold and fake near-duplicate rules disappear."
    )
