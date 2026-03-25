#!/usr/bin/env python3
"""
Pre-build the gene ID cache for CASCADE.

Fetches Ensembl ID -> gene symbol mappings for all genes in the GREmLN model
vocabulary using the Ensembl REST API batch endpoint. The resulting cache file
(cache/gene_id_cache.json) is committed to the repo so that TCGA embedding
analysis runs without any API calls at query time.

Usage:
    python scripts/prebuild_gene_cache.py

Runtime: ~5-10 minutes depending on network speed (~20 batches of 1000 genes).
"""

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
CACHE_FILE = ROOT / "cache" / "gene_id_cache.json"
ENSEMBL_BATCH_URL = "https://rest.ensembl.org/lookup/id"
BATCH_SIZE = 1000
RETRY_WAIT = 5  # seconds between retries on rate-limit


def load_existing_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"symbol_to_ensembl": {}, "ensembl_to_symbol": {}}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def fetch_batch(ensembl_ids: list[str]) -> dict[str, str]:
    """Fetch symbol for a batch of Ensembl IDs. Returns {ensembl_id: symbol}."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"ids": ensembl_ids}
    for attempt in range(3):
        try:
            resp = requests.post(
                ENSEMBL_BATCH_URL, headers=headers,
                json=payload, timeout=30,
            )
            if resp.status_code == 429:
                print(f"  Rate limited, waiting {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
                continue
            resp.raise_for_status()
            data = resp.json()
            return {
                ens_id: info["display_name"].upper()
                for ens_id, info in data.items()
                if info and "display_name" in info
            }
        except Exception as e:
            print(f"  Batch attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(RETRY_WAIT)
    return {}


def get_model_vocab() -> list[str]:
    """Return all Ensembl IDs from the GREmLN model vocabulary."""
    sys.path.insert(0, str(ROOT))
    from scGraphLLM import GeneVocab
    vocab = GeneVocab.load_default()
    return [g for g in vocab.genes if g.upper().startswith("ENSG")]


def main() -> None:
    print("Loading model vocabulary...")
    vocab_genes = get_model_vocab()
    print(f"  {len(vocab_genes)} Ensembl IDs in model vocab")

    cache = load_existing_cache()
    already_cached = set(cache["ensembl_to_symbol"].keys())
    to_fetch = [g for g in vocab_genes if g not in already_cached]
    print(f"  {len(already_cached)} already cached, {len(to_fetch)} to fetch")

    if not to_fetch:
        print("Cache is complete.")
        return

    batches = [to_fetch[i:i + BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]
    print(f"Fetching {len(batches)} batches of up to {BATCH_SIZE} genes...")

    total_fetched = 0
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}/{len(batches)} ({len(batch)} genes)...", end=" ", flush=True)
        result = fetch_batch(batch)
        for ens, sym in result.items():
            cache["ensembl_to_symbol"][ens] = sym
            cache["symbol_to_ensembl"][sym] = ens
        total_fetched += len(result)
        print(f"{len(result)} resolved")
        save_cache(cache)  # save incrementally

    print(f"\nDone. {total_fetched} new mappings added.")
    print(f"Cache now covers {len(cache['ensembl_to_symbol'])} Ensembl IDs "
          f"({len(cache['ensembl_to_symbol']) / len(vocab_genes) * 100:.1f}% of model vocab).")
    print(f"Saved to {CACHE_FILE}")


if __name__ == "__main__":
    main()
