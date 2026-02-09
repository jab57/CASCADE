#!/usr/bin/env python3
"""
Validate CASCADE network predictions against LINCS L1000 experimental knockdown data.

For each gene that is both a regulator in the epithelial cell network AND has
LINCS knockdown data, compares CASCADE's predicted top affected genes with
LINCS experimental top affected genes. Reports Jaccard overlap, directional
concordance, and Fisher's exact test p-values.

Usage:
    python scripts/validate_lincs.py
    python scripts/validate_lincs.py --top-k 50 --max-genes 100 --output results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats

from tools.loader import load_network, NETWORKS_DIR
from tools.perturb import _build_adjacency, _propagate_effect
from tools.lincs import load_lincs_data, get_knockdown_effects
from tools.gene_id_mapper import get_mapper


def build_id_maps(mapper):
    """Build fast lookup dicts from the mapper's cache (no API calls)."""
    ensembl_to_symbol = dict(mapper.cache.get("ensembl_to_symbol", {}))
    symbol_to_ensembl = dict(mapper.cache.get("symbol_to_ensembl", {}))
    return ensembl_to_symbol, symbol_to_ensembl


def find_overlapping_genes(network_df, lincs_df, ensembl_to_symbol):
    """Find genes that are regulators in the network AND have LINCS knockdown data."""
    network_regulators = set(network_df["regulator"].unique())
    lincs_knockdowns = set(lincs_df["gene_ko"].str.upper().unique())

    # Count targets per regulator for sorting
    target_counts = network_df.groupby("regulator").size().to_dict()

    overlapping = []
    resolved = 0

    for ensembl_id in network_regulators:
        symbol = ensembl_to_symbol.get(ensembl_id)
        if symbol is None:
            continue
        resolved += 1

        if symbol.upper() in lincs_knockdowns:
            overlapping.append({
                "ensembl_id": ensembl_id,
                "symbol": symbol.upper(),
                "num_targets": target_counts.get(ensembl_id, 0)
            })

    # Sort by target count descending (most informative genes first)
    overlapping.sort(key=lambda g: g["num_targets"], reverse=True)

    print(f"Network regulators: {len(network_regulators)}")
    print(f"  Resolved from cache: {resolved}")
    print(f"  Not in cache (skipped): {len(network_regulators) - resolved}")
    print(f"LINCS knockdowns: {len(lincs_knockdowns)}")
    print(f"Overlapping genes: {len(overlapping)}")

    return overlapping


def compare_gene(gene_info, adj, ensembl_to_symbol, top_k, universe_size):
    """Compare CASCADE prediction vs LINCS experimental data for one gene.

    Uses _propagate_effect directly and resolves symbols from cache only
    (no API calls).
    """
    ensembl_id = gene_info["ensembl_id"]
    symbol = gene_info["symbol"]

    # CASCADE: propagate through network (no API calls)
    effects = _propagate_effect(adj, ensembl_id, initial_effect=-1.0, depth=2)
    effects.pop(ensembl_id, None)

    if not effects:
        return None

    # Sort by magnitude and take top-k
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    # Convert to symbols using cache only
    cascade_genes = set()
    cascade_directions = {}
    for ens_id, effect in sorted_effects:
        sym = ensembl_to_symbol.get(ens_id)
        if sym:
            sym = sym.upper()
            cascade_genes.add(sym)
            cascade_directions[sym] = "down" if effect < 0 else "up"

    # LINCS experimental affected genes
    lincs_effects = get_knockdown_effects(symbol, direction="any", top_k=top_k)

    if not lincs_effects:
        return None

    lincs_genes = set()
    lincs_directions = {}
    for g in lincs_effects:
        sym = g["gene"].upper()
        lincs_genes.add(sym)
        lincs_directions[sym] = g["direction"]

    if not cascade_genes or not lincs_genes:
        return None

    # Jaccard overlap
    overlap = cascade_genes & lincs_genes
    union = cascade_genes | lincs_genes
    jaccard = len(overlap) / len(union) if union else 0

    # Directional concordance (among overlapping genes)
    concordant = 0
    discordant = 0
    for gene in overlap:
        if gene in cascade_directions and gene in lincs_directions:
            if cascade_directions[gene] == lincs_directions[gene]:
                concordant += 1
            else:
                discordant += 1
    direction_concordance = concordant / (concordant + discordant) if (concordant + discordant) > 0 else 0

    # Fisher's exact test
    a = len(overlap)
    b = len(cascade_genes - lincs_genes)
    c = len(lincs_genes - cascade_genes)
    d = max(0, universe_size - a - b - c)

    _, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

    return {
        "symbol": symbol,
        "ensembl_id": ensembl_id,
        "cascade_top_k": len(cascade_genes),
        "lincs_top_k": len(lincs_genes),
        "overlap_count": len(overlap),
        "overlap_genes": ", ".join(sorted(overlap)) if overlap else "",
        "jaccard": round(jaccard, 4),
        "direction_concordance": round(direction_concordance, 4),
        "concordant": concordant,
        "discordant": discordant,
        "fisher_p": fisher_p,
        "total_network_affected": len(effects)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate CASCADE predictions against LINCS L1000 experimental data"
    )
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of top affected genes to compare (default: 50)")
    parser.add_argument("--max-genes", type=int, default=50,
                        help="Maximum number of overlapping genes to test (default: 50)")
    parser.add_argument("--cell-type", type=str, default="epithelial_cell",
                        help="Cell type network to use (default: epithelial_cell)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path (default: scripts/validation_results.csv)")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(PROJECT_ROOT / "scripts" / "validation_results.csv")

    print("=" * 70)
    print("CASCADE vs LINCS L1000 Validation")
    print("=" * 70)
    print(f"Cell type: {args.cell_type}")
    print(f"Top-k genes compared: {args.top_k}")
    print(f"Max genes to test: {args.max_genes}")
    print()

    # Load data
    print("Loading network...")
    network_path = NETWORKS_DIR / args.cell_type / "network.tsv"
    network_df = load_network(network_path)
    print(f"  {len(network_df)} edges, {network_df['regulator'].nunique()} regulators, "
          f"{network_df['target'].nunique()} targets")

    print("Building adjacency list...")
    adj = _build_adjacency(network_df)

    print("Loading LINCS data...")
    lincs_df = load_lincs_data()

    print("Initializing gene ID mapper...")
    mapper = get_mapper()
    ensembl_to_symbol, symbol_to_ensembl = build_id_maps(mapper)
    print(f"  Cache has {len(ensembl_to_symbol)} ensembl->symbol mappings")

    universe_size = lincs_df["gene"].nunique()
    print(f"LINCS universe size: {universe_size} genes")
    print()

    # Find overlapping genes
    print("Finding overlapping genes...")
    overlapping = find_overlapping_genes(network_df, lincs_df, ensembl_to_symbol)

    if not overlapping:
        print("No overlapping genes found. Cannot validate.")
        return

    # Limit to max-genes
    test_genes = overlapping[:args.max_genes]
    print(f"\nTesting {len(test_genes)} genes (sorted by network target count)...")
    print()

    # Compare each gene
    results = []
    for i, gene_info in enumerate(test_genes):
        sys.stdout.write(f"\r  [{i+1}/{len(test_genes)}] {gene_info['symbol']} "
                         f"({gene_info['num_targets']} targets)...")
        sys.stdout.flush()

        result = compare_gene(gene_info, adj, ensembl_to_symbol, args.top_k, universe_size)
        if result:
            results.append(result)

    print(f"\r  Completed: {len(results)} genes with results out of {len(test_genes)} tested"
          + " " * 30)
    print()

    if not results:
        print("No genes produced comparable results.")
        return

    # Summary statistics
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    jaccards = [r["jaccard"] for r in results]
    overlaps = [r["overlap_count"] for r in results]
    fisher_ps = [r["fisher_p"] for r in results]
    concordances = [r["direction_concordance"] for r in results if r["overlap_count"] > 0]

    significant = sum(1 for p in fisher_ps if p < 0.05)
    any_overlap = sum(1 for o in overlaps if o > 0)

    print(f"Genes tested:              {len(results)}")
    print(f"Genes with any overlap:    {any_overlap}/{len(results)} "
          f"({100*any_overlap/len(results):.0f}%)")
    print(f"Mean Jaccard overlap:      {sum(jaccards) / len(jaccards):.4f}")
    print(f"Median Jaccard overlap:    {sorted(jaccards)[len(jaccards)//2]:.4f}")
    print(f"Mean overlap count:        {sum(overlaps) / len(overlaps):.1f} genes")
    print(f"Max overlap count:         {max(overlaps)} genes")
    if concordances:
        print(f"Mean direction concordance:{sum(concordances) / len(concordances):.4f}")
    print(f"Fisher's exact p < 0.05:   {significant}/{len(results)} "
          f"({100*significant/len(results):.0f}%)")
    print()

    # Top genes by overlap
    top_by_overlap = sorted(results, key=lambda r: r["overlap_count"], reverse=True)[:10]
    print("Top 10 genes by overlap:")
    print(f"  {'Gene':<12} {'Overlap':>8} {'Jaccard':>8} {'Concordance':>12} {'Fisher p':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
    for r in top_by_overlap:
        print(f"  {r['symbol']:<12} {r['overlap_count']:>8} {r['jaccard']:>8.4f} "
              f"{r['direction_concordance']:>12.4f} {r['fisher_p']:>10.2e}")

    # Write CSV
    print(f"\nWriting results to {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Done.")


if __name__ == "__main__":
    main()
