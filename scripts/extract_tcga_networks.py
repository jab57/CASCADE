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
    https://bioconductor.org/packages/3.23/data/experiment/src/contrib/aracne.networks_1.38.0.tar.gz
"""

import argparse
import csv
import hashlib
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

ZENODO_RECORD_ID = "22918956"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

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


def _parse_rda_bytes(raw_bytes: bytes, cancer_type: str):
    """Parse raw .rda bytes into the rdata regulon object for one cancer type."""
    import rdata

    parsed = rdata.read_rda(io.BytesIO(raw_bytes))
    var_name = f"regulon{cancer_type}"
    if var_name in parsed:
        return parsed[var_name]
    return next(iter(parsed.values()))


def load_rda_from_tarball(tarball_path: str, cancer_type: str):
    """
    Load a single .rda regulon object from inside the tarball.
    Returns the parsed rdata object (dict of Entrez ID -> regulon entries).
    """
    rda_name = RDA_NAMES[cancer_type]
    inner_path = f"aracne.networks/data/{rda_name}"

    print(f"  Reading {inner_path} from tarball ...")
    with tarfile.open(tarball_path, "r:gz") as tf:
        member = tf.getmember(inner_path)
        fh = tf.extractfile(member)
        raw_bytes = fh.read()

    return _parse_rda_bytes(raw_bytes, cancer_type)


def load_rda_from_dir(rda_dir: str, cancer_type: str):
    """Load a single regulon{type}.rda file from a directory of .rda files."""
    path = os.path.join(rda_dir, RDA_NAMES[cancer_type])
    print(f"  Reading {path} ...")
    with open(path, "rb") as fh:
        raw_bytes = fh.read()
    return _parse_rda_bytes(raw_bytes, cancer_type)


def _md5_of_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_from_zenodo(cancer_types: list, dest_dir: str) -> str:
    """
    Download regulon{type}.rda files from the Zenodo record into dest_dir,
    verifying each against the MD5 checksum published in the record.
    Files already present with a matching checksum are reused.
    Returns dest_dir. Raises RuntimeError on any download or checksum failure.
    """
    print(f"Fetching file list from Zenodo record {ZENODO_RECORD_ID} ...")
    try:
        with urllib.request.urlopen(ZENODO_API_URL, timeout=60) as resp:
            record = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"could not reach {ZENODO_API_URL}: {exc}") from exc

    files = {f["key"]: f for f in record.get("files", [])}
    os.makedirs(dest_dir, exist_ok=True)

    for ct in cancer_types:
        name = RDA_NAMES[ct]
        entry = files.get(name)
        if entry is None:
            raise RuntimeError(f"{name} not found in Zenodo record {ZENODO_RECORD_ID}")
        algo, _, expected = entry["checksum"].partition(":")
        if algo != "md5":
            raise RuntimeError(f"unexpected checksum type {algo!r} for {name}")

        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest) and _md5_of_file(dest) == expected:
            print(f"  {name}: already downloaded, checksum OK")
            continue

        print(f"  {name}: downloading {entry['size'] / 1e6:.1f} MB ...")
        tmp = dest + ".part"
        try:
            with urllib.request.urlopen(entry["links"]["self"], timeout=120) as resp, \
                    open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(f"download of {name} failed: {exc}") from exc

        actual = _md5_of_file(tmp)
        if actual != expected:
            os.remove(tmp)
            raise RuntimeError(f"checksum mismatch for {name}: expected {expected}, got {actual}")
        os.replace(tmp, dest)
        print(f"  {name}: checksum OK")

    return dest_dir


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
        description="Extract TCGA ARACNe networks (Zenodo record 22918956 by default) to CSV."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--rda-dir",
        default=None,
        help="Directory of regulon{type}.rda files already downloaded from the Zenodo record",
    )
    source.add_argument(
        "--tarball",
        default=None,
        help="Path to the Bioconductor aracne.networks tarball (fallback source)",
    )
    parser.add_argument(
        "--download-dir",
        default=os.path.join(os.environ.get("TEMP", "/tmp"), "aracne_zenodo_22918956"),
        help="Where Zenodo files are downloaded when neither --rda-dir nor --tarball "
             "is given (default: $TEMP/aracne_zenodo_22918956)",
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
        help="Process only one cancer type (default: all 14)",
    )
    args = parser.parse_args()

    cancer_types = [args.cancer_type] if args.cancer_type else list(CANCER_TYPE_MAP.keys())

    # Step 0: choose the input source (Zenodo unless told otherwise)
    if args.tarball:
        if not os.path.exists(args.tarball):
            print(f"ERROR: tarball not found: {args.tarball}")
            print("Download from:")
            print("  https://bioconductor.org/packages/3.23/data/experiment/src/contrib/aracne.networks_1.38.0.tar.gz")
            sys.exit(1)
        load_one = lambda ct: load_rda_from_tarball(args.tarball, ct)
        source_label = "tarball"
    else:
        rda_dir = args.rda_dir
        if rda_dir is None:
            try:
                rda_dir = download_from_zenodo(cancer_types, args.download_dir)
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                print("Download the regulon*.rda files manually from "
                      f"https://doi.org/10.5281/zenodo.{ZENODO_RECORD_ID} and pass --rda-dir,")
                print("or use --tarball with the Bioconductor aracne.networks package.")
                sys.exit(1)
        elif not os.path.isdir(rda_dir):
            print(f"ERROR: --rda-dir is not a directory: {rda_dir}")
            sys.exit(1)
        load_one = lambda ct: load_rda_from_dir(rda_dir, ct)
        source_label = f"Zenodo files in {rda_dir}"

    # Step 1: Load all requested networks
    print(f"\nLoading {len(cancer_types)} network(s) from {source_label} ...")
    networks = {}
    for ct in cancer_types:
        try:
            networks[ct] = load_one(ct)
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
