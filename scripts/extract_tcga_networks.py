#!/usr/bin/env python3
"""
Install the TCGA ARACNe tumor-state networks used by CASCADE.

The networks are the data files of the Bioconductor package ``aracne.networks``
(Giorgi & Alvarez). Their authors publish the files on Zenodo
(https://doi.org/10.5281/zenodo.22918956) under CC BY-NC-ND 4.0 (attribution,
non-commercial use, no sharing of modified versions). CASCADE does not ship these
networks: this script downloads them from the official source and builds the CSVs
that load_tcga_network() reads, locally.

What it does:
  1. Shows the license notice and requires --accept-license.
  2. Downloads the 14 regulon{type}.rda files from the Zenodo record (default),
     reads them from a directory (--rda-dir), or reads them from a Bioconductor
     aracne.networks 1.36.0/1.38.0 tarball (--source bioconductor or --tarball),
     which carries the package's Columbia license.
  3. Checks each .rda against the SHA-256 recorded in scripts/data/
     tcga_entrez_to_symbol.json.gz (identical in the Zenodo record and in
     aracne.networks 1.36.0 and 1.38.0).
  4. Converts Entrez IDs to gene symbols with that file's frozen mapping (no
     network call), builds the network CSV, checks its SHA-256, and only then
     moves it to data/networks/tcga/<type>/network.csv, so every install
     reproduces exactly the networks used by CASCADE and its paper.

Usage:
    pip install rdata==1.1.0
    python scripts/extract_tcga_networks.py --accept-license
    python scripts/extract_tcga_networks.py --accept-license --cancer-type brca coad
    python scripts/extract_tcga_networks.py --accept-license --rda-dir /path/to/rda_files
    python scripts/extract_tcga_networks.py --accept-license --source bioconductor
    python scripts/extract_tcga_networks.py --accept-license --tarball aracne.networks_1.38.0.tar.gz
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

ZENODO_RECORD_ID = "22918956"
ZENODO_RECORD = f"https://doi.org/10.5281/zenodo.{ZENODO_RECORD_ID}"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
BIOC_LICENSE_URL = ("https://bioconductor.org/packages/release/data/experiment/"
                    "licenses/aracne.networks/LICENSE")
# Version-pinned Bioconductor URLs (the generic "release/" URL changes at every
# Bioconductor release). Network data are identical in 1.36.0 and 1.38.0; from 1.39.x
# the package no longer contains the .rda data files.
DOWNLOAD_URLS = [
    "https://bioconductor.org/packages/3.23/data/experiment/src/contrib/aracne.networks_1.38.0.tar.gz",
    "https://bioconductor.org/packages/3.22/data/experiment/src/contrib/aracne.networks_1.36.0.tar.gz",
]
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "tcga_entrez_to_symbol.json.gz")

# Map from Bioconductor dataset name -> our cancer type key
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

# Known .rda filenames (Zenodo record root; aracne.networks/data/ in the tarball)
# Format: regulon{ct}.rda, variable name: regulon{ct}
RDA_NAMES = {ct: f"regulon{ct}.rda" for ct in CANCER_TYPE_MAP}

LICENSE_NOTICE = {
    "zenodo": f"""
The TCGA networks are the data files of the Bioconductor package aracne.networks
(Giorgi FM, Alvarez MJ), published by its authors on Zenodo ({ZENODO_RECORD})
under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 license
(CC BY-NC-ND 4.0). In summary (read the full text before continuing): credit the
authors and the record; non-commercial use only; do not share modified versions of
the networks (the CSVs this script builds are for your own use).

Full license: {ZENODO_LICENSE_URL}

CASCADE does not redistribute these networks. By passing --accept-license you
confirm that you have read the license and that your use complies with it.
""",
    "bioconductor": f"""
The TCGA networks come from the Bioconductor package aracne.networks, distributed
under a Columbia University software evaluation license. In summary (read the full
text before continuing): use is limited to non-commercial academic or educational
research; you may not redistribute the package or make it available to third
parties; commercial use requires a license from Columbia University. The same
network data are also available under CC BY-NC-ND 4.0 (--source zenodo, the default).

Full license: {BIOC_LICENSE_URL}

CASCADE does not redistribute these networks. By passing --accept-license you
confirm that you have read the license and that your use complies with it.
""",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_symbol_maps() -> dict:
    """Load the frozen Entrez -> symbol mapping and per-network checksums."""
    with gzip.open(MAP_PATH, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def symbol_map_for(data: dict, cancer_type: str) -> dict:
    t = data["types"][cancer_type]
    unmapped = set(t["unmapped"])
    m = {e: s for e, s in data["base"].items() if e not in unmapped}
    m.update(t["override"])
    return m


def parse_rda_bytes(raw_bytes: bytes, cancer_type: str):
    """Parse raw .rda bytes into the rdata regulon object for one cancer type."""
    import rdata

    parsed = rdata.read_rda(io.BytesIO(raw_bytes))
    var_name = f"regulon{cancer_type}"
    if var_name in parsed:
        return parsed[var_name]
    return next(iter(parsed.values()))


def read_rda_from_dir(rda_dir: str, cancer_type: str) -> bytes:
    """Read one regulon{type}.rda file from a directory of .rda files."""
    path = os.path.join(rda_dir, RDA_NAMES[cancer_type])
    print(f"  Reading {path} ...")
    with open(path, "rb") as fh:
        return fh.read()


def read_rda_from_tarball(tf: tarfile.TarFile, cancer_type: str) -> bytes:
    """Read one .rda file from an open aracne.networks tarball."""
    member = tf.extractfile(f"aracne.networks/data/{RDA_NAMES[cancer_type]}")
    return member.read() if member else b""


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


def download_bioconductor_tarball(dest: str) -> str:
    """Download the pinned aracne.networks tarball, trying each pinned URL in turn."""
    for url in DOWNLOAD_URLS:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:  # try the next mirror/version
            print(f"  failed: {exc}")
    sys.exit("ERROR: could not download aracne.networks. Download it manually from "
             "https://bioconductor.org/packages/aracne.networks and pass --tarball.")


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


def write_csv(edges: list, path: str) -> bytes:
    """Write edges to CSV (columns: Regulator, Target, MoA, Likelihood); return its bytes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Regulator", "Target", "MoA", "Likelihood"],
                                lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(edges)
    with open(path, "rb") as fh:
        return fh.read()


def install_csv(edges: list, path: str, expected_sha256: str) -> bool:
    """Write the network CSV to a temp file and move it to ``path`` only if its
    SHA-256 matches. On a mismatch no new file is written to ``path`` (an
    earlier verified file there is left untouched)."""
    tmp_path = path + ".tmp"
    try:
        written = write_csv(edges, tmp_path)
        if sha256(written) != expected_sha256:
            return False
        os.replace(tmp_path, path)
        return True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Install the TCGA ARACNe networks (aracne.networks data) for CASCADE."
    )
    ap.add_argument("--accept-license", action="store_true",
                    help="Confirm you have read and comply with the license of the chosen source.")
    ap.add_argument("--source", choices=["zenodo", "bioconductor"], default="zenodo",
                    help="Zenodo record (CC BY-NC-ND 4.0, default) or Bioconductor tarball "
                         "(Columbia license).")
    ap.add_argument("--rda-dir", default=None,
                    help="Directory of regulon{type}.rda files already downloaded from the "
                         "Zenodo record (implies --source zenodo, no download).")
    ap.add_argument("--tarball", default=None,
                    help="Use a local aracne.networks_*.tar.gz (implies --source bioconductor).")
    ap.add_argument("--download-dir",
                    default=os.path.join(os.environ.get("TEMP", "/tmp"), "aracne_zenodo_22918956"),
                    help="Where Zenodo files are cached when downloading "
                         "(default: $TEMP/aracne_zenodo_22918956)")
    ap.add_argument("--cancer-type", nargs="+", choices=list(CANCER_TYPE_MAP), metavar="TYPE",
                    help=f"Only these types (default: all 14): {', '.join(CANCER_TYPE_MAP)}")
    ap.add_argument("--output-dir", default="data/networks/tcga",
                    help="Root output directory (default: data/networks/tcga)")
    ap.add_argument("--keep-tarball", action="store_true",
                    help="Do not delete a downloaded Bioconductor tarball.")
    args = ap.parse_args()

    if args.rda_dir and args.tarball:
        ap.error("--rda-dir and --tarball are mutually exclusive")
    if args.tarball:
        args.source = "bioconductor"
    if args.rda_dir:
        args.source = "zenodo"

    print(LICENSE_NOTICE[args.source])
    if not args.accept_license:
        sys.exit("Not accepted: re-run with --accept-license once you have read the license.")

    try:
        import rdata  # noqa: F401
    except ImportError:
        sys.exit('ERROR: this script needs the "rdata" package: pip install rdata==1.1.0')

    maps = load_symbol_maps()
    types = args.cancer_type or list(CANCER_TYPE_MAP)

    # Source data: the .rda bytes of each requested network.
    raw = {}
    tmpdir = None
    tarball = args.tarball
    try:
        if args.source == "zenodo":
            rda_dir = args.rda_dir
            if rda_dir is None:
                try:
                    rda_dir = download_from_zenodo(types, args.download_dir)
                except RuntimeError as exc:
                    sys.exit(f"ERROR: {exc}\nDownload the regulon*.rda files manually from "
                             f"{ZENODO_RECORD} and pass --rda-dir, or use --source bioconductor.")
            elif not os.path.isdir(rda_dir):
                sys.exit(f"ERROR: --rda-dir is not a directory: {rda_dir}")
            for ct in types:
                try:
                    raw[ct] = read_rda_from_dir(rda_dir, ct)
                except OSError as exc:
                    sys.exit(f"ERROR: could not read {RDA_NAMES[ct]} ({exc}).")
        else:
            if not tarball:
                tmpdir = tempfile.mkdtemp(prefix="aracne_networks_")
                tarball = download_bioconductor_tarball(os.path.join(tmpdir, "aracne.networks.tar.gz"))
            elif not os.path.exists(tarball):
                sys.exit(f"ERROR: tarball not found: {tarball}\nDownload from:\n  {DOWNLOAD_URLS[0]}")
            with tarfile.open(tarball, "r:gz") as tf:
                names = set(tf.getnames())
                missing = [ct for ct in types if f"aracne.networks/data/{RDA_NAMES[ct]}" not in names]
                if missing:
                    sys.exit("ERROR: this aracne.networks tarball has no network data files "
                             f"({', '.join(missing)}). Use version 1.36.0 or 1.38.0, e.g.\n  "
                             + DOWNLOAD_URLS[0] + "\nor use --source zenodo.")
                for ct in types:
                    raw[ct] = read_rda_from_tarball(tf, ct)
    finally:
        if tmpdir and not args.keep_tarball:
            try:
                os.remove(tarball)
                os.rmdir(tmpdir)
            except OSError:
                pass

    failures = []
    for ct in types:
        expected = maps["types"][ct]
        print(f"\n[{ct}] checking source data ...")
        if sha256(raw[ct]) != expected["rda_sha256"]:
            failures.append(ct)
            print(f"  ERROR: {RDA_NAMES[ct]} differs from the version CASCADE was built "
                  "with; skipping (results would not match).")
            continue
        regulon = parse_rda_bytes(raw[ct], ct)
        edges, skipped = regulon_to_edges(regulon, symbol_map_for(maps, ct))
        csv_path = os.path.join(args.output_dir, ct, "network.csv")
        if not install_csv(edges, csv_path, expected["csv_sha256"]):
            failures.append(ct)
            print(f"  ERROR: rebuilt {csv_path} does not match the expected checksum; "
                  "not installed.")
            continue
        print(f"  {len(edges):,} edges, {skipped:,} endpoints skipped (no symbol) -> "
              f"{csv_path} (checksum verified)")

    if failures:
        sys.exit(f"\nFAILED for: {', '.join(failures)}")
    print(f"\nTCGA networks installed: {', '.join(types)}")


if __name__ == "__main__":
    main()
