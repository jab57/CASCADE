"""
Tests for TCGA ARACNe network loader (tools/loader.py: load_tcga_network).
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_csv(tmp_path: Path, cancer_type: str, rows: list[dict]) -> Path:
    """Write a minimal network CSV to tmp_path/tcga/{cancer_type}/network.csv."""
    out = tmp_path / cancer_type / "network.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out


SAMPLE_EDGES = [
    {"Regulator": "ESR1",  "Target": "GATA3", "MoA":  1.0, "Likelihood": 0.95},
    {"Regulator": "ESR1",  "Target": "FOXA1", "MoA":  1.0, "Likelihood": 0.90},
    {"Regulator": "TP53",  "Target": "CDKN1A","MoA":  1.0, "Likelihood": 0.88},
    {"Regulator": "MYC",   "Target": "CDK4",  "MoA": -1.0, "Likelihood": 0.75},  # repressing
]


# ---------------------------------------------------------------------------
# load_tcga_network
# ---------------------------------------------------------------------------

class TestLoadTcgaNetwork:

    def test_returns_dataframe_for_valid_cancer_type(self, tmp_path):
        from tools.loader import load_tcga_network, TCGA_NETWORKS_DIR, _network_cache
        _make_fake_csv(tmp_path, "brca", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df = load_tcga_network("brca")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_columns_mapped_correctly(self, tmp_path):
        from tools.loader import load_tcga_network
        _make_fake_csv(tmp_path, "brca", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df = load_tcga_network("brca")

        assert set(df.columns) >= {"regulator", "target", "mi", "scc", "count", "log_p"}

    def test_likelihood_mapped_to_mi(self, tmp_path):
        from tools.loader import load_tcga_network
        _make_fake_csv(tmp_path, "brca", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df = load_tcga_network("brca")

        assert df["mi"].iloc[0] == pytest.approx(0.95)

    def test_scc_count_logp_defaults(self, tmp_path):
        from tools.loader import load_tcga_network
        _make_fake_csv(tmp_path, "brca", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df = load_tcga_network("brca")

        assert (df["scc"] == 0.0).all()
        assert (df["count"] == 0).all()
        assert (df["log_p"] == 0.0).all()

    def test_signed_mi_column_present(self, tmp_path):
        from tools.loader import load_tcga_network
        _make_fake_csv(tmp_path, "brca", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df = load_tcga_network("brca")

        assert "signed_mi" in df.columns

    def test_signed_mi_activation_positive(self, tmp_path):
        from tools.loader import load_tcga_network
        _make_fake_csv(tmp_path, "brca", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df = load_tcga_network("brca")

        # ESR1→GATA3 MoA=+1 → signed_mi == +Likelihood
        row = df[df["target"] == "GATA3"].iloc[0]
        assert row["signed_mi"] == pytest.approx(0.95)

    def test_signed_mi_repression_negative(self, tmp_path):
        from tools.loader import load_tcga_network
        _make_fake_csv(tmp_path, "brca", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df = load_tcga_network("brca")

        # MYC→CDK4 MoA=-1 → signed_mi == -Likelihood
        row = df[df["target"] == "CDK4"].iloc[0]
        assert row["signed_mi"] == pytest.approx(-0.75)

    def test_unknown_cancer_type_returns_error(self):
        from tools.loader import load_tcga_network
        result = load_tcga_network("gbm")
        assert isinstance(result, dict)
        assert "error" in result

    def test_unknown_cancer_type_laml_returns_error(self):
        from tools.loader import load_tcga_network
        result = load_tcga_network("laml")
        assert isinstance(result, dict)
        assert "error" in result

    def test_unknown_cancer_type_lists_valid_options(self):
        from tools.loader import load_tcga_network
        result = load_tcga_network("unknown_type")
        assert "brca" in result["error"]

    def test_missing_file_returns_error(self, tmp_path):
        from tools.loader import load_tcga_network
        # brca dir exists but no network.csv
        (tmp_path / "brca").mkdir(parents=True, exist_ok=True)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            result = load_tcga_network("brca")

        assert isinstance(result, dict)
        assert "error" in result

    def test_result_is_cached(self, tmp_path):
        from tools.loader import load_tcga_network, _network_cache
        _make_fake_csv(tmp_path, "luad", SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            df1 = load_tcga_network("luad")
            df2 = load_tcga_network("luad")

        assert df1 is df2  # Same object from cache

    def test_all_valid_cancer_types_accepted(self, tmp_path):
        """Loader accepts all 14 valid types without returning an error for the type check."""
        from tools.loader import load_tcga_network, VALID_TCGA_CANCER_TYPES
        valid_types = ["blca", "brca", "cesc", "coad", "hnsc", "kirc", "lihc",
                       "luad", "lusc", "ov", "paad", "prad", "stad", "ucec"]
        assert set(valid_types) == VALID_TCGA_CANCER_TYPES

        for ct in valid_types:
            _make_fake_csv(tmp_path, ct, SAMPLE_EDGES)

        with patch("tools.loader.TCGA_NETWORKS_DIR", tmp_path):
            for ct in valid_types:
                df = load_tcga_network(ct)
                assert isinstance(df, pd.DataFrame), f"Expected DataFrame for {ct}"


# ---------------------------------------------------------------------------
# Regression: existing cell-type load_network path unchanged
# ---------------------------------------------------------------------------

class TestLoadNetworkRegression:

    def test_cell_type_network_still_loads_tsv(self, tmp_path):
        """load_network() still reads TSV files — TCGA changes are purely additive."""
        from tools.loader import load_network, _network_cache

        tsv_path = tmp_path / "network.tsv"
        tsv_path.write_text(
            "regulator\ttarget\tmi\tscc\tcount\tlog_p\n"
            "ENSG001\tENSG002\t0.5\t0.3\t10\t-5.0\n"
        )

        df = load_network(tsv_path)
        assert isinstance(df, pd.DataFrame)
        assert "regulator" in df.columns
        assert df["mi"].iloc[0] == pytest.approx(0.5)
        # signed_mi is derived from mi * sign(scc); scc=0.3 > 0 → signed_mi == mi
        assert "signed_mi" in df.columns
        assert df["signed_mi"].iloc[0] == pytest.approx(0.5)
