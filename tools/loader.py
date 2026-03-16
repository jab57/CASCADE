"""
Network and model loading utilities for CASCADE perturbation analysis.
"""

from pathlib import Path
import pandas as pd

# Default paths
BASE_DIR = Path(__file__).parent.parent
NETWORKS_DIR = BASE_DIR / "data" / "networks"
TCGA_NETWORKS_DIR = BASE_DIR / "data" / "networks" / "tcga"
MODEL_PATH = BASE_DIR / "models" / "model.ckpt"

VALID_TCGA_CANCER_TYPES = frozenset(
    ["brca", "coad", "hnsc", "luad", "lusc", "ov", "prad", "ucec"]
)

# In-process network cache: avoids re-reading TSV files on repeated calls
# for the same cell type within a server session.
_network_cache: dict = {}


def load_network(network_path: Path | str) -> pd.DataFrame:
    """
    Load a gene regulatory network from TSV file.

    Results are cached in-process so repeated calls for the same network
    (e.g., multiple analyses on the same cell type, or cross_cell_comparison)
    do not re-read from disk.

    Args:
        network_path: Path to the network TSV file

    Returns:
        DataFrame with columns: regulator, target, mi (mutual information),
        scc (spearman correlation), count, log_p
    """
    network_path = Path(network_path)
    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    cache_key = str(network_path.resolve())
    if cache_key in _network_cache:
        return _network_cache[cache_key]

    df = pd.read_csv(network_path, sep="\t")

    # Normalize column names for easier access
    df.columns = [
        col.replace(".values", "").replace(".", "_")
        for col in df.columns
    ]

    _network_cache[cache_key] = df
    return df


def load_tcga_network(cancer_type: str) -> pd.DataFrame:
    """
    Load a TCGA ARACNe network CSV as a DataFrame compatible with CASCADE BFS propagation.

    Input CSV columns (exported from aracne.networks .rda via scripts/extract_tcga_networks.py):
        Regulator, Target, MoA, Likelihood
    Output columns expected by CASCADE BFS:
        regulator, target, mi, scc, count, log_p

    Networks use gene symbols natively — GeneIDMapper is never called in this code path.
    Results are cached in-process (same cache as load_network).

    Args:
        cancer_type: One of brca, coad, hnsc, luad, lusc, ov, prad, ucec

    Returns:
        DataFrame with columns: regulator, target, mi, scc, count, log_p
        On error (unknown type, missing file), returns a dict with "error" key.
    """
    if cancer_type not in VALID_TCGA_CANCER_TYPES:
        return {"error": f"Unknown TCGA cancer type '{cancer_type}'. "
                         f"Valid options: {sorted(VALID_TCGA_CANCER_TYPES)}"}

    csv_path = TCGA_NETWORKS_DIR / cancer_type / "network.csv"
    if not csv_path.exists():
        return {"error": f"TCGA network file not found: {csv_path}. "
                         "Run scripts/extract_tcga_networks.py to generate it."}

    cache_key = str(csv_path.resolve())
    if cache_key in _network_cache:
        return _network_cache[cache_key]

    df = pd.read_csv(csv_path)

    # Map CSV columns to CASCADE BFS-expected column names
    df = df.rename(columns={"Regulator": "regulator", "Target": "target", "Likelihood": "mi"})
    df["scc"] = 0.0
    df["count"] = 0
    df["log_p"] = 0.0

    _network_cache[cache_key] = df
    return df


def get_available_cell_types(networks_dir: Path | str = NETWORKS_DIR) -> list[str]:
    """
    Get list of available cell types with pre-computed networks.

    Args:
        networks_dir: Directory containing cell type subdirectories

    Returns:
        List of cell type names
    """
    networks_dir = Path(networks_dir)
    if not networks_dir.exists():
        return []

    cell_types = []
    for subdir in networks_dir.iterdir():
        if subdir.is_dir() and (subdir / "network.tsv").exists():
            cell_types.append(subdir.name)

    return sorted(cell_types)


def load_cascade_model(model_path: Path | str = MODEL_PATH):
    """
    Load GREmLN model checkpoint (optional, for advanced embedding-based analysis).

    Args:
        model_path: Path to model checkpoint

    Returns:
        Tuple of (model, device)
    """
    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CASCADE] Loading model on: {device}")

    # weights_only=False is required: the checkpoint contains PyTorch Lightning
    # state that cannot be loaded with weights_only=True. Only load checkpoints
    # from trusted sources (e.g., the bundled models/model.ckpt).
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    return checkpoint, device
