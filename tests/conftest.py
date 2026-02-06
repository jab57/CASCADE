"""
Shared test fixtures for CASCADE test suite.

Provides mock networks, mock model checkpoints, and temporary data
directories so tests run without real data files or GPU.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Mock regulatory network
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_network_df():
    """A small regulatory network for testing perturbation propagation.

    Network topology:
        ENSG_TF1 --0.8--> ENSG_TARGET1
        ENSG_TF1 --0.6--> ENSG_TARGET2
        ENSG_TF2 --0.5--> ENSG_TARGET2
        ENSG_TF2 --0.4--> ENSG_TARGET3
        ENSG_TARGET1 --0.3--> ENSG_DOWNSTREAM1  (2nd hop)

    TF1 is a master-regulator-like node (2 targets).
    TF2 is a secondary regulator (2 targets).
    TARGET1 feeds into DOWNSTREAM1 (indirect effect).
    """
    return pd.DataFrame({
        "regulator": [
            "ENSG_TF1", "ENSG_TF1",
            "ENSG_TF2", "ENSG_TF2",
            "ENSG_TARGET1",
        ],
        "target": [
            "ENSG_TARGET1", "ENSG_TARGET2",
            "ENSG_TARGET2", "ENSG_TARGET3",
            "ENSG_DOWNSTREAM1",
        ],
        "mi": [0.8, 0.6, 0.5, 0.4, 0.3],
        "scc": [0.7, 0.5, 0.4, 0.3, 0.2],
        "count": [100, 80, 60, 50, 40],
        "log_p": [-10.0, -8.0, -6.0, -5.0, -4.0],
    })


@pytest.fixture
def mock_network_tsv(tmp_path, mock_network_df):
    """Write mock network to a TSV file and return the path."""
    network_dir = tmp_path / "data" / "networks" / "test_cell_type"
    network_dir.mkdir(parents=True)
    tsv_path = network_dir / "network.tsv"
    mock_network_df.to_csv(tsv_path, sep="\t", index=False)
    return tsv_path


@pytest.fixture
def mock_networks_dir(tmp_path, mock_network_df):
    """Create a networks directory with two cell types for cell-type discovery tests."""
    networks_dir = tmp_path / "data" / "networks"
    for cell_type in ["cd8_t_cells", "epithelial_cell"]:
        cell_dir = networks_dir / cell_type
        cell_dir.mkdir(parents=True)
        mock_network_df.to_csv(cell_dir / "network.tsv", sep="\t", index=False)
    return networks_dir


# ---------------------------------------------------------------------------
# Mock CascadeModel
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cascade_model():
    """A mock CascadeModel that returns fake embeddings and similarities."""
    model = MagicMock()
    model._loaded = True

    # Vocabulary: 5 genes + 3 special tokens
    genes = ["<PAD>", "<MASK>", "<CLS>",
             "ENSG_TF1", "ENSG_TF2", "ENSG_TARGET1",
             "ENSG_TARGET2", "ENSG_TARGET3"]
    gene_to_node = {g: i for i, g in enumerate(genes)}

    model.vocab = MagicMock()
    model.vocab.genes = genes
    model.vocab.gene_to_node = gene_to_node

    model.is_gene_in_vocab.side_effect = lambda g: g in gene_to_node

    # Return fake embeddings (8 genes, 4-dim for simplicity)
    np.random.seed(42)
    embeddings = np.random.randn(len(genes), 4).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    def get_all_similarities(gene):
        idx = gene_to_node.get(gene)
        if idx is None:
            return None
        sims = normalized @ normalized[idx]
        special = {"<PAD>", "<MASK>", "<CLS>"}
        rows = []
        for i, g in enumerate(genes):
            if g in special or g == gene:
                continue
            rows.append({"ensembl_id": g, "similarity": float(sims[i])})
        df = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
        return df

    model.get_all_similarities.side_effect = get_all_similarities

    def compute_similarity(g1, g2):
        i1 = gene_to_node.get(g1)
        i2 = gene_to_node.get(g2)
        if i1 is None or i2 is None:
            return None
        return float(np.dot(normalized[i1], normalized[i2]))

    model.compute_similarity.side_effect = compute_similarity

    return model


# ---------------------------------------------------------------------------
# Gene ID mapper mock (avoids Ensembl API calls)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gene_id_mapper():
    """Mock GeneIDMapper with a small pre-loaded cache."""
    with patch("tools.gene_id_mapper._mapper", None):
        mapper = MagicMock()
        mapper.cache = {
            "symbol_to_ensembl": {
                "MYC": "ENSG00000136997",
                "TP53": "ENSG00000141510",
                "BRCA1": "ENSG00000012048",
                "APC": "ENSG00000134982",
            },
            "ensembl_to_symbol": {
                "ENSG00000136997": "MYC",
                "ENSG00000141510": "TP53",
                "ENSG00000012048": "BRCA1",
                "ENSG00000134982": "APC",
            },
        }
        mapper.symbol_to_ensembl.side_effect = lambda s: mapper.cache["symbol_to_ensembl"].get(s.upper())
        mapper.ensembl_to_symbol.side_effect = lambda e: mapper.cache["ensembl_to_symbol"].get(e)
        mapper.get_cache_stats.return_value = {
            "cached_symbols": 4,
            "cached_ensembls": 4,
            "cache_file": "mock_cache.pkl",
        }
        yield mapper


# ---------------------------------------------------------------------------
# LINCS mock data
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_lincs_df():
    """Small LINCS-like DataFrame for testing."""
    return pd.DataFrame({
        "gene": ["CDKN1A", "CDKN1A", "MYC", "MYC", "GAPDH"],
        "gene_ko": ["TP53", "RB1", "BRD4", "MYC", "CTRL"],
        "effect": [-1.5, -0.8, -2.1, 0.1, 0.0],
        "direction": [-1, -1, -1, 1, 0],
    })


# ---------------------------------------------------------------------------
# Super-enhancer mock data
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_se_df():
    """Small super-enhancer DataFrame for testing."""
    return pd.DataFrame({
        "gene_symbol": ["MYC", "MYC", "BCL2", "IRF4"],
        "cell_name": ["K562", "HepG2", "GM12878", "GM12878"],
        "chrom": ["chr8", "chr8", "chr18", "chr6"],
        "start": [128700000, 128700000, 60790000, 336000],
        "end": [128800000, 128800000, 60900000, 450000],
    })
