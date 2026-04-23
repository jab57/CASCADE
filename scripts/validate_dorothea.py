#!/usr/bin/env python3
"""
Validate CASCADE network propagation predictions against DoRothEA TF regulons.

For each TF present in both the epithelial_cell network and DoRothEA, compares
CASCADE's top-k BFS-propagated knockdown targets against DoRothEA's curated
regulon targets (levels A/B/C). Reports Jaccard overlap, directional concordance,
and Fisher's exact test p-values.

Usage:
    python scripts/validate_dorothea.py
    python scripts/validate_dorothea.py --top-k 100 --levels A B --output results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from tools.loader import load_network, NETWORKS_DIR
from tools.perturb import _build_adjacency, _propagate_effect
from tools.dorothea import load_dorothea_regulons
from tools.gene_id_mapper import get_mapper


def build_id_maps(mapper):
    ensembl_to_symbol = dict(mapper.cache.get("ensembl_to_symbol", {}))
    symbol_to_ensembl = dict(mapper.cache.get("symbol_to_ensembl", {}))
    return ensembl_to_symbol, symbol_to_ensembl


def find_overlapping_tfs(network_df, dorothea_df, ensembl_to_symbol, symbol_to_ensembl):
    """Find TFs present in both the network (as regulators) and DoRothEA (as source)."""
    dorothea_tfs = set(dorothea_df["source"].str.upper().unique())
    network_regulators = set(network_df["regulator"].unique())
    target_counts = network_df.groupby("regulator").size().to_dict()

    overlapping = []
    for ensembl_id in network_regulators:
        symbol = ensembl_to_symbol.get(ensembl_id)
        if symbol is None:
            continue
        if symbol.upper() in dorothea_tfs:
            overlapping.append({
                "ensembl_id": ensembl_id,
                "symbol": symbol.upper(),
                "num_targets": target_counts.get(ensembl_id, 0),
            })

    overlapping.sort(key=lambda g: g["num_targets"], reverse=True)
    return overlapping


def compare_tf(tf_info, adj, dorothea_df, ensembl_to_symbol, top_k, universe_size):
    ensembl_id = tf_info["ensembl_id"]
    symbol = tf_info["symbol"]

    # CASCADE: BFS knockdown propagation
    effects = _propagate_effect(adj, ensembl_id, initial_effect=-1.0, depth=3)
    effects.pop(ensembl_id, None)
    if not effects:
        return None

    # Take top-k by magnitude, resolve to symbols
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    cascade_genes = {}
    for ens_id, effect in sorted_effects:
        sym = ensembl_to_symbol.get(ens_id)
        if sym:
            cascade_genes[sym.upper()] = "down" if effect < 0 else "up"

    if not cascade_genes:
        return None

    # DoRothEA: curated regulon targets for this TF
    mask = dorothea_df["source"].str.upper() == symbol
    tf_regulon = dorothea_df[mask]
    if tf_regulon.empty:
        return None

    dorothea_genes = {}
    for _, row in tf_regulon.iterrows():
        dorothea_genes[row["target"].upper()] = "down" if row["mor"] < 0 else "up"

    # Overlap and Jaccard
    cascade_set = set(cascade_genes.keys())
    dorothea_set = set(dorothea_genes.keys())
    overlap = cascade_set & dorothea_set
    union = cascade_set | dorothea_set
    jaccard = len(overlap) / len(union) if union else 0

    # Directional concordance among overlapping genes
    concordant = sum(
        1 for g in overlap
        if g in cascade_genes and g in dorothea_genes
        and cascade_genes[g] == dorothea_genes[g]
    )
    discordant = len(overlap) - concordant
    direction_concordance = concordant / len(overlap) if overlap else 0

    # Fisher's exact test (one-tailed, greater)
    a = len(overlap)
    b = len(cascade_set - dorothea_set)
    c = len(dorothea_set - cascade_set)
    d = max(0, universe_size - a - b - c)
    _, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

    return {
        "symbol": symbol,
        "ensembl_id": ensembl_id,
        "cascade_top_k": len(cascade_genes),
        "dorothea_targets": len(dorothea_genes),
        "overlap_count": len(overlap),
        "overlap_genes": ", ".join(sorted(overlap)) if overlap else "",
        "jaccard": round(jaccard, 4),
        "direction_concordance": round(direction_concordance, 4),
        "concordant": concordant,
        "discordant": discordant,
        "fisher_p": fisher_p,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate CASCADE predictions against DoRothEA TF regulons"
    )
    parser.add_argument("--top-k", type=int, default=100,
                        help="Top CASCADE predicted targets to compare (default: 100)")
    parser.add_argument("--max-tfs", type=int, default=100,
                        help="Max TFs to test (default: 100)")
    parser.add_argument("--levels", nargs="+", default=["A", "B", "C"],
                        help="DoRothEA confidence levels (default: A B C)")
    parser.add_argument("--cell-type", type=str, default="epithelial_cell")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(PROJECT_ROOT / "scripts" / "dorothea_validation_results.csv")

    print("=" * 70)
    print("CASCADE vs DoRothEA TF Regulon Validation")
    print("=" * 70)
    print(f"Cell type:          {args.cell_type}")
    print(f"Top-k targets:      {args.top_k}")
    print(f"DoRothEA levels:    {args.levels}")
    print()

    print("Loading network...")
    network_path = NETWORKS_DIR / args.cell_type / "network.tsv"
    network_df = load_network(network_path)
    print(f"  {len(network_df)} edges, {network_df['regulator'].nunique()} regulators, "
          f"{network_df['target'].nunique()} targets")

    print("Building adjacency list...")
    adj = _build_adjacency(network_df)

    print("Loading DoRothEA regulons...")
    dorothea_df = load_dorothea_regulons(levels=args.levels)
    print(f"  {len(dorothea_df):,} interactions, {dorothea_df['source'].nunique()} TFs, "
          f"{dorothea_df['target'].nunique()} targets")

    print("Initializing gene ID mapper...")
    mapper = get_mapper()
    ensembl_to_symbol, symbol_to_ensembl = build_id_maps(mapper)
    print(f"  Cache: {len(ensembl_to_symbol)} ensembl->symbol mappings")

    # Universe = all unique genes in DoRothEA (targets + TFs)
    universe_size = dorothea_df["target"].nunique()
    print(f"DoRothEA target universe: {universe_size} genes")
    print()

    print("Finding overlapping TFs...")
    overlapping = find_overlapping_tfs(network_df, dorothea_df, ensembl_to_symbol, symbol_to_ensembl)
    print(f"  TFs in both network and DoRothEA: {len(overlapping)}")

    test_tfs = overlapping[:args.max_tfs]
    print(f"  Testing {len(test_tfs)} TFs (sorted by network target count)")
    print()

    results = []
    for i, tf_info in enumerate(test_tfs):
        sys.stdout.write(f"\r  [{i+1}/{len(test_tfs)}] {tf_info['symbol']} "
                         f"({tf_info['num_targets']} network targets)...")
        sys.stdout.flush()
        result = compare_tf(tf_info, adj, dorothea_df, ensembl_to_symbol, args.top_k, universe_size)
        if result:
            results.append(result)

    print(f"\r  Done: {len(results)} TFs with results" + " " * 40)
    print()

    if not results:
        print("No results produced.")
        return

    # BH-FDR correction
    pvals = [r["fisher_p"] for r in results]
    _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
    for r, padj in zip(results, pvals_corrected):
        r["fisher_p_adj"] = padj

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    jaccards = [r["jaccard"] for r in results]
    overlaps = [r["overlap_count"] for r in results]
    concordances = [r["direction_concordance"] for r in results if r["overlap_count"] > 0]
    sig_raw = sum(1 for r in results if r["fisher_p"] < 0.05)
    sig_adj = sum(1 for r in results if r["fisher_p_adj"] < 0.05)
    any_overlap = sum(1 for o in overlaps if o > 0)

    print(f"TFs tested:                {len(results)}")
    print(f"TFs with any overlap:      {any_overlap}/{len(results)} "
          f"({100*any_overlap/len(results):.0f}%)")
    print(f"Mean Jaccard overlap:      {sum(jaccards)/len(jaccards):.4f}")
    print(f"Median Jaccard overlap:    {sorted(jaccards)[len(jaccards)//2]:.4f}")
    print(f"Mean overlap count:        {sum(overlaps)/len(overlaps):.1f} genes")
    print(f"Max overlap count:         {max(overlaps)} genes")
    if concordances:
        print(f"Mean direction concordance:{sum(concordances)/len(concordances):.4f}")
    print(f"Fisher p < 0.05 (raw):     {sig_raw}/{len(results)} "
          f"({100*sig_raw/len(results):.0f}%)")
    print(f"Fisher p < 0.05 (BH-FDR):  {sig_adj}/{len(results)} "
          f"({100*sig_adj/len(results):.0f}%)")
    print()

    top_by_overlap = sorted(results, key=lambda r: r["overlap_count"], reverse=True)[:15]
    print("Top 15 TFs by overlap count:")
    print(f"  {'TF':<12} {'Network':>8} {'DoRothEA':>9} {'Overlap':>8} "
          f"{'Jaccard':>8} {'Concord':>8} {'p-adj':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in top_by_overlap:
        print(f"  {r['symbol']:<12} {r['cascade_top_k']:>8} {r['dorothea_targets']:>9} "
              f"{r['overlap_count']:>8} {r['jaccard']:>8.4f} "
              f"{r['direction_concordance']:>8.4f} {r['fisher_p_adj']:>10.2e}")

    print(f"\nWriting results to {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Done.")


if __name__ == "__main__":
    main()
