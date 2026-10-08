"""
Tests for verify_installation.py: the optional TCGA check and the cell-type loop.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

import tools.loader as loader

SCRIPT = Path(__file__).resolve().parent.parent / "verify_installation.py"


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("verify_installation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install(tcga_dir: Path, cancer_types):
    for ct in cancer_types:
        (tcga_dir / ct).mkdir(parents=True, exist_ok=True)
        (tcga_dir / ct / "network.csv").write_text("Regulator,Target,MoA,Likelihood\r\n")


def test_tcga_none_installed_is_skip_not_failure(verify, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(loader, "TCGA_NETWORKS_DIR", tmp_path / "tcga")
    assert verify.check_tcga_networks() == (1, 0)
    out = capsys.readouterr().out
    assert "0/14 installed" in out
    assert "--accept-license" in out and "extract_tcga_networks.py" in out


def test_tcga_partial_install_is_skip_with_count(verify, monkeypatch, tmp_path, capsys):
    tcga = tmp_path / "tcga"
    _install(tcga, ["brca", "coad", "ov"])
    monkeypatch.setattr(loader, "TCGA_NETWORKS_DIR", tcga)
    assert verify.check_tcga_networks() == (1, 0)
    assert "3/14 installed" in capsys.readouterr().out


def test_tcga_all_installed_passes(verify, monkeypatch, tmp_path, capsys):
    tcga = tmp_path / "tcga"
    _install(tcga, sorted(loader.VALID_TCGA_CANCER_TYPES))
    monkeypatch.setattr(loader, "TCGA_NETWORKS_DIR", tcga)
    assert verify.check_tcga_networks() == (1, 0)
    assert "all 14 installed" in capsys.readouterr().out


def test_networks_check_ignores_tcga_folder(verify, monkeypatch, tmp_path):
    """data/networks/tcga holds TCGA CSVs, not a GREmLN cell type with network.tsv."""
    (tmp_path / "epithelial_cell").mkdir()
    (tmp_path / "tcga").mkdir()
    monkeypatch.setattr(loader, "NETWORKS_DIR", tmp_path)
    monkeypatch.setattr(
        loader, "load_network",
        lambda path: pd.DataFrame({"regulator": ["A"], "target": ["B"], "mi": [0.5]}),
    )
    assert verify.check_networks() == (1, 0)
