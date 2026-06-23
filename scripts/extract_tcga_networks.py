#!/usr/bin/env python3
"""
Extract TCGA ARACNe networks from Bioconductor aracne.networks tarball
and write CSVs compatible with CASCADE's load_tcga_network() loader.

Usage:
    python scripts/extract_tcga_networks.py \
        --tarball /path/to/aracne.networks_1.38.0.tar.gz \
        --output-dir data/networks/tcga

Requires:
    pip install rdata requests

The tarball is the Bioconductor experiment data package:
    https://bioconductor.org/packages/release/data/experiment/src/contrib/aracne.networks_1.38.0.tar.gz
"""

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
import urllib.parse
import urllib.error
from collections import defaultdict

MYGENE_URL = "https://mygene.info/v3/gene"

# Map from Bioconductor dataset name → our cancer type key
CANCER_TYPE_MAP = {
    "blca": "blca",
    "brca": "brca",
    "cesc": "cesc",
    "coad": "coad",
    "hnsc": "hnsc",
    "kirc": "kirc",
    "lihc": "lihc",
    "luad": "luad",
    "lusc": "lusc",
    "ov":   "ov",
    "paad": "paad",
    "prad": "prad",
    "stad": "stad",
    "ucec": "ucec",
}

# Known .rda filenames inside the package (data/ directory)
# Format: aracne.networks/data/regulon{ct}.rda, variable name: regulon{ct}
RDA_NAMES = {
    "blca": "regulonblca.rda",
    "brca": "regulonbrca.rda",
    "cesc": "reguloncesc.rda",
    "coad": "reguloncoad.rda",
    "hnsc": "regulonhnsc.rda",
    "kirc": "regulonkirc.rda",
    "lihc": "regulonlihc.rda",
    "luad": "regulonluad.rda",
    "lusc": "regulonlusc.rda",
    "ov":   "regulonov.rda",
    "paad": "regulonpaad.rda",
    "prad": "regulonprad.rda",
    "stad": "regulonstad.rda",
    "ucec": "regulonucec.rda",
}


def load_rda_from_tarball(tarball_path: str, cancer_type: str):
    """
    Load a single .rda regulon object from inside the tarball.
    Returns the parsed rdata object (dict of Entrez ID → regulon entries).
    """
    import rdata

    rda_name = RDA_NAMES[cancer_type]
    inner_path = f"aracne.networks/data/{rda_name}"

    print(f"  Reading {inner_path} from tarball ...")
    with tarfile.open(tarball_path, "r:gz") as tf:
        member = tf.getmember(inner_path)
        fh = tf.extractfile(member)
        raw_bytes = fh.read()

    parsed = rdata.read_rda(io.BytesIO(raw_bytes))
    var_name = f"regulon{cancer_type}"
    if var_name in parsed:
        return parsed[var_name]
    return next(iter(parsed.values()))


def collect_entrez_ids(networks: dict) -> list:
    """Collect all unique Entrez IDs across all loaded networks."""
    all_ids = set()
    for cancer_type, regulon in networks.items():
        for reg_id, reg_data in regulon.items():
            all_ids.add(str(reg_id))
            try:
                targets = list(reg_data["tfmode"].coords[
                    list(reg_data["tfmode"].dims)[0]
                ].values)
                for t in targets:
                    all_ids.add(str(t))
            except Exception:
                pass
    return sorted(all_ids)


def entrez_to_symbol_batch(entrez_ids: list, batch_size: int = 1000) -> dict:
    """
    Convert Entrez IDs → gene symbols via MyGene.info batch POST.
    Returns dict: entrez_id_str → symbol.
    """
    print(f"Converting {len(entrez_ids):,} Entrez IDs to symbols via MyGene.info ...")
    id_to_symbol = {}
    unresolved = []

    for i in range(0, len(entrez_ids), batch_size):
        batch = [str(x) for x in entrez_ids[i:i + batch_size]]
        batch_num = i // batch_size + 1
        total_batches = (len(entrez_ids) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} IDs) ...", end=" ", flush=True)

        payload = json.dumps({
            "ids": batch,
            "fields": "symbol",
            "species": "human",
        }).encode("utf-8")

        req = urllib.request.Request(
            MYGENE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                hits = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP {exc.code}: {body[:200]}")
            continue
        except urllib.error.URLError as exc:
            print(f"URLError: {exc}")
            continue

        resolved_this_batch = 0
        for hit in hits:
            entrez_str = str(hit.get("query", ""))
            if hit.get("notfound"):
                unresolved.append(entrez_str)
                continue
            symbol = hit.get("symbol", "")
            if symbol:
                id_to_symbol[entrez_str] = symbol.upper()
                resolved_this_batch += 1
            else:
                unresolved.append(entrez_str)

        print(f"resolved {resolved_this_batch}/{len(batch)}")

        if i + batch_size < len(entrez_ids):
            time.sleep(0.3)

    total = len(entrez_ids)
    n_resolved = len(id_to_symbol)
    rate = n_resolved / total if total else 0
    print(f"  Total resolved: {n_resolved:,}/{total:,} ({rate:.1%})")
    if unresolved[:20]:
        print(f"  Unresolved sample: {unresolved[:20]}")
    return id_to_symbol


def regulon_to_edges(regulon, id_to_symbol: dict):
    """
    Convert a regulon dict (Entrez-keyed) to a list of edge dicts:
        {"Regulator": symbol, "Target": symbol, "MoA": float, "Likelihood": float}
    Skips edges where either endpoint cannot be mapped to a symbol.
    """
    edges = []
    skipped = 0

    for reg_id, reg_data in regulon.items():
        reg_symbol = id_to_symbol.get(str(reg_id))
        if not reg_symbol:
            skipped += 1
            continue

        try:
            tfmode = reg_data["tfmode"]
            likelihood_arr = reg_data["likelihood"]

            dim_name = list(tfmode.dims)[0]
            target_ids = [str(x) for x in tfmode.coords[dim_name].values]
            moa_values = tfmode.values.tolist()
            likelihood_values = likelihood_arr.tolist() if hasattr(likelihood_arr, "tolist") else list(likelihood_arr)
        except Exception as exc:
            print(f"  WARNING: skipping regulator {reg_id}: {exc}")
            skipped += 1
            continue

        for tgt_id, moa, lik in zip(target_ids, moa_values, likelihood_values):
            tgt_symbol = id_to_symbol.get(tgt_id)
            if not tgt_symbol:
                skipped += 1
                continue
            if tgt_symbol == reg_symbol:
                continue
            edges.append({
                "Regulator": reg_symbol,
                "Target":    tgt_symbol,
                "MoA":       round(float(moa), 6),
                "Likelihood": round(float(lik), 6),
            })

    return edges, skipped


def write_csv(edges: list, output_path: str) -> None:
    """Write edges to CSV with columns: Regulator, Target, MoA, Likelihood."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Regulator", "Target", "MoA", "Likelihood"])
        writer.writeheader()
        writer.writerows(edges)
    print(f"  Wrote {len(edges):,} edges -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract TCGA ARACNe networks from Bioconductor tarball to CSV."
    )
    parser.add_argument(
        "--tarball",
        default=os.path.join(
            os.environ.get("TEMP", "/tmp"),
            "aracne.networks.tar.gz"
        ),
        help="Path to aracne.networks tarball (default: $TEMP/aracne.networks.tar.gz)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/networks/tcga",
        help="Root output directory (default: data/networks/tcga)",
    )
    parser.add_argument(
        "--cancer-type",
        choices=list(CANCER_TYPE_MAP.keys()),
        default=None,
        help="Process only one cancer type (default: all 8)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.tarball):
        print(f"ERROR: tarball not found: {args.tarball}")
        print("Download from:")
        print("  https://bioconductor.org/packages/release/data/experiment/src/contrib/aracne.networks_1.38.0.tar.gz")
        sys.exit(1)

    cancer_types = [args.cancer_type] if args.cancer_type else list(CANCER_TYPE_MAP.keys())

    # Step 1: Load all requested networks from tarball
    print(f"\nLoading {len(cancer_types)} network(s) from tarball ...")
    networks = {}
    for ct in cancer_types:
        try:
            networks[ct] = load_rda_from_tarball(args.tarball, ct)
            print(f"  {ct}: {len(networks[ct]):,} regulators loaded")
        except Exception as exc:
            print(f"  ERROR loading {ct}: {exc}")

    if not networks:
        print("No networks loaded. Exiting.")
        sys.exit(1)

    # Step 2: Collect all unique Entrez IDs
    all_entrez = collect_entrez_ids(networks)
    print(f"\nTotal unique Entrez IDs: {len(all_entrez):,}")

    # Step 3: Batch convert Entrez → symbol
    id_to_symbol = entrez_to_symbol_batch(all_entrez)

    if not id_to_symbol:
        print("ERROR: No symbols resolved. Check network connectivity.")
        sys.exit(1)

    # Step 4: Convert each network to edges and write CSV
    print(f"\nConverting and writing CSVs ...")
    for ct, regulon in networks.items():
        print(f"\n  Processing {ct} ...")
        edges, skipped = regulon_to_edges(regulon, id_to_symbol)
        print(f"  {ct}: {len(edges):,} edges, {skipped:,} endpoints skipped (no symbol)")

        csv_path = os.path.join(args.output_dir, ct, "network.csv")
        write_csv(edges, csv_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
