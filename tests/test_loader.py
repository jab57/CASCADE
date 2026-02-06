"""Tests for tools/loader.py — network loading and cell type discovery."""

import pytest
import pandas as pd
from pathlib import Path

from tools.loader import load_network, get_available_cell_types


class TestLoadNetwork:
    def test_loads_tsv(self, mock_network_tsv):
        df = load_network(mock_network_tsv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "regulator" in df.columns
        assert "target" in df.columns
        assert "mi" in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_network(tmp_path / "nonexistent.tsv")

    def test_column_normalization(self, tmp_path):
        """Columns with '.values' suffix should be stripped."""
        tsv = tmp_path / "network.tsv"
        df = pd.DataFrame({
            "regulator.values": ["A"],
            "target.values": ["B"],
            "mi.values": [0.5],
        })
        df.to_csv(tsv, sep="\t", index=False)
        loaded = load_network(tsv)
        assert "regulator_values" in loaded.columns or "regulator" in loaded.columns


class TestGetAvailableCellTypes:
    def test_finds_cell_types(self, mock_networks_dir):
        cell_types = get_available_cell_types(mock_networks_dir)
        assert "cd8_t_cells" in cell_types
        assert "epithelial_cell" in cell_types
        assert len(cell_types) == 2

    def test_sorted_output(self, mock_networks_dir):
        cell_types = get_available_cell_types(mock_networks_dir)
        assert cell_types == sorted(cell_types)

    def test_empty_for_missing_dir(self, tmp_path):
        result = get_available_cell_types(tmp_path / "nonexistent")
        assert result == []

    def test_ignores_dirs_without_network_tsv(self, tmp_path):
        """Directories without network.tsv should not appear."""
        networks_dir = tmp_path / "networks"
        (networks_dir / "empty_cell").mkdir(parents=True)
        result = get_available_cell_types(networks_dir)
        assert result == []
