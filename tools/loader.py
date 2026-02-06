"""
Network and model loading utilities for CASCADE perturbation analysis.
"""

from pathlib import Path
import pandas as pd

# Default paths
BASE_DIR = Path(__file__).parent.parent
NETWORKS_DIR = BASE_DIR / "data" / "networks"
MODEL_PATH = BASE_DIR / "models" / "model.ckpt"


def load_network(network_path: Path | str) -> pd.DataFrame:
    """
    Load a gene regulatory network from TSV file.

    Args:
        network_path: Path to the network TSV file

    Returns:
        DataFrame with columns: regulator, target, mi (mutual information),
        scc (spearman correlation), count, log_p
    """
    network_path = Path(network_path)
    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    df = pd.read_csv(network_path, sep="\t")

    # Normalize column names for easier access
    df.columns = [
        col.replace(".values", "").replace(".", "_")
        for col in df.columns
    ]

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

    # Load checkpoint - the actual class depends on what's in the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    return checkpoint, device
