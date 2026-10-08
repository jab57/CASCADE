"""
Tests for scripts/extract_tcga_networks.py (Zenodo download, frozen symbol map,
checksum-gated install).

No network access: urllib is mocked.
"""

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "extract_tcga_networks.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("extract_tcga_networks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(payload: bytes):
    """A context-manager response whose read() drains payload in chunks."""
    buf = io.BytesIO(payload)
    resp = MagicMock()
    resp.read.side_effect = lambda n=-1: buf.read(n)
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _record(files: dict) -> bytes:
    return json.dumps({
        "files": [
            {
                "key": name,
                "size": len(data),
                "checksum": "md5:" + hashlib.md5(data).hexdigest(),
                "links": {"self": f"https://zenodo.org/files/{name}"},
            }
            for name, data in files.items()
        ]
    }).encode()


def _fake_urlopen(record_bytes: bytes, files: dict, tampered: dict | None = None):
    tampered = tampered or {}

    def opener(url, timeout=None):
        if url == "https://zenodo.org/api/records/22918956":
            return _response(record_bytes)
        name = url.rsplit("/", 1)[-1]
        return _response(tampered.get(name, files[name]))

    return opener


# ---------------------------------------------------------------------------
# Zenodo download
# ---------------------------------------------------------------------------

def test_download_writes_verified_files(mod, tmp_path):
    files = {"regulonbrca.rda": b"brca-bytes", "regulonov.rda": b"ov-bytes"}
    opener = _fake_urlopen(_record(files), files)
    with patch.object(mod.urllib.request, "urlopen", side_effect=opener):
        out = mod.download_from_zenodo(["brca", "ov"], str(tmp_path))
    assert out == str(tmp_path)
    assert (tmp_path / "regulonbrca.rda").read_bytes() == b"brca-bytes"
    assert (tmp_path / "regulonov.rda").read_bytes() == b"ov-bytes"
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_checksum_mismatch(mod, tmp_path):
    files = {"regulonbrca.rda": b"good-bytes"}
    opener = _fake_urlopen(_record(files), files, tampered={"regulonbrca.rda": b"bad-bytes"})
    with patch.object(mod.urllib.request, "urlopen", side_effect=opener):
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            mod.download_from_zenodo(["brca"], str(tmp_path))
    assert not (tmp_path / "regulonbrca.rda").exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_reuses_valid_existing_file(mod, tmp_path):
    files = {"regulonbrca.rda": b"brca-bytes"}
    (tmp_path / "regulonbrca.rda").write_bytes(b"brca-bytes")
    calls = []

    def opener(url, timeout=None):
        calls.append(url)
        return _fake_urlopen(_record(files), files)(url, timeout)

    with patch.object(mod.urllib.request, "urlopen", side_effect=opener):
        mod.download_from_zenodo(["brca"], str(tmp_path))
    assert calls == ["https://zenodo.org/api/records/22918956"]


def test_download_missing_file_in_record(mod, tmp_path):
    files = {"regulonov.rda": b"ov-bytes"}
    opener = _fake_urlopen(_record(files), files)
    with patch.object(mod.urllib.request, "urlopen", side_effect=opener):
        with pytest.raises(RuntimeError, match="not found in Zenodo record"):
            mod.download_from_zenodo(["brca"], str(tmp_path))


def test_download_unreachable_record(mod, tmp_path):
    err = mod.urllib.error.URLError("offline")
    with patch.object(mod.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="could not reach"):
            mod.download_from_zenodo(["brca"], str(tmp_path))


def test_read_rda_from_dir_returns_file_bytes(mod, tmp_path):
    (tmp_path / "regulonbrca.rda").write_bytes(b"x")
    assert mod.read_rda_from_dir(str(tmp_path), "brca") == b"x"


def test_rda_names_cover_all_cancer_types(mod):
    assert set(mod.RDA_NAMES) == set(mod.CANCER_TYPE_MAP)
    assert all(name == f"regulon{ct}.rda" for ct, name in mod.RDA_NAMES.items())


# ---------------------------------------------------------------------------
# Frozen symbol map
# ---------------------------------------------------------------------------

def test_frozen_map_covers_all_types_with_checksums(mod):
    maps = mod.load_symbol_maps()
    assert set(maps["types"]) == set(mod.CANCER_TYPE_MAP)
    for ct, t in maps["types"].items():
        assert len(t["rda_sha256"]) == 64 and len(t["csv_sha256"]) == 64, ct


def test_frozen_map_keeps_historical_symbols(mod):
    """MyGene.info now returns NUCLEOLIN / FASTKD4 for these; the frozen map must not."""
    m = mod.symbol_map_for(mod.load_symbol_maps(), "brca")
    assert m["4691"] == "NCL"
    assert m["9238"] == "TBRG4"


def test_symbol_map_for_applies_unmapped_and_override(mod):
    data = {
        "base": {"1": "AAA", "2": "BBB", "3": "CCC"},
        "types": {"x": {"unmapped": ["2"], "override": {"3": "ZZZ", "9": "NEW"}}},
    }
    assert mod.symbol_map_for(data, "x") == {"1": "AAA", "3": "ZZZ", "9": "NEW"}


# ---------------------------------------------------------------------------
# Checksum-gated install
# ---------------------------------------------------------------------------

EDGES = [{"Regulator": "A", "Target": "B", "MoA": 1.0, "Likelihood": 0.5}]


def _expected_sha(mod, tmp_path):
    ref = tmp_path / "ref" / "network.csv"
    return mod.sha256(mod.write_csv(EDGES, str(ref)))


def test_install_csv_moves_file_when_checksum_matches(mod, tmp_path):
    target = tmp_path / "out" / "network.csv"
    assert mod.install_csv(EDGES, str(target), _expected_sha(mod, tmp_path)) is True
    assert target.read_bytes().startswith(b"Regulator,Target,MoA,Likelihood\r\n")
    assert not (tmp_path / "out" / "network.csv.tmp").exists()


def test_install_csv_leaves_existing_file_on_mismatch(mod, tmp_path):
    target = tmp_path / "out" / "network.csv"
    target.parent.mkdir()
    target.write_bytes(b"previous verified file")
    assert mod.install_csv(EDGES, str(target), "0" * 64) is False
    assert target.read_bytes() == b"previous verified file"
    assert not (tmp_path / "out" / "network.csv.tmp").exists()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_requires_accept_license(mod, monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--cancer-type", "brca", "--output-dir", str(out)])
    with patch.object(mod, "download_from_zenodo") as dl:
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert "--accept-license" in str(exc.value)
    dl.assert_not_called()
    assert not out.exists()


def test_main_rejects_both_tarball_and_rda_dir(mod, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--accept-license", "--tarball", "a.tgz", "--rda-dir", "d"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


def test_main_zenodo_failure_exits_with_fallback_hint(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--accept-license", "--cancer-type", "brca",
                                      "--download-dir", str(tmp_path)])
    monkeypatch.setitem(sys.modules, "rdata", MagicMock())  # the test must not need rdata installed
    with patch.object(mod, "download_from_zenodo", side_effect=RuntimeError("offline")):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    msg = str(exc.value)
    assert "offline" in msg and "--rda-dir" in msg and "--source bioconductor" in msg


def _synthetic_run(mod, monkeypatch, tmp_path, rda_bytes, rda_sha, csv_sha):
    rda_dir = tmp_path / "rda"
    rda_dir.mkdir()
    (rda_dir / "regulonbrca.rda").write_bytes(rda_bytes)
    out = tmp_path / "out"
    maps = {"base": {}, "types": {"brca": {"unmapped": [], "override": {},
                                           "rda_sha256": rda_sha, "csv_sha256": csv_sha}}}
    monkeypatch.setattr(sys, "argv", ["x", "--accept-license", "--rda-dir", str(rda_dir),
                                      "--cancer-type", "brca", "--output-dir", str(out)])
    monkeypatch.setitem(sys.modules, "rdata", MagicMock())
    return out, maps


def test_main_installs_when_all_checksums_match(mod, monkeypatch, tmp_path):
    out, maps = _synthetic_run(mod, monkeypatch, tmp_path, b"rda", hashlib.sha256(b"rda").hexdigest(),
                               _expected_sha(mod, tmp_path))
    with patch.object(mod, "load_symbol_maps", return_value=maps), \
            patch.object(mod, "parse_rda_bytes", return_value={}), \
            patch.object(mod, "regulon_to_edges", return_value=(EDGES, 0)):
        mod.main()
    assert (out / "brca" / "network.csv").exists()


def test_main_skips_source_file_with_wrong_checksum(mod, monkeypatch, tmp_path):
    out, maps = _synthetic_run(mod, monkeypatch, tmp_path, b"tampered", hashlib.sha256(b"rda").hexdigest(),
                               _expected_sha(mod, tmp_path))
    with patch.object(mod, "load_symbol_maps", return_value=maps), \
            patch.object(mod, "parse_rda_bytes", return_value={}), \
            patch.object(mod, "regulon_to_edges", return_value=(EDGES, 0)):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert "FAILED for: brca" in str(exc.value)
    assert not (out / "brca" / "network.csv").exists()


def test_main_does_not_install_csv_with_wrong_checksum(mod, monkeypatch, tmp_path):
    out, maps = _synthetic_run(mod, monkeypatch, tmp_path, b"rda", hashlib.sha256(b"rda").hexdigest(),
                               "0" * 64)
    with patch.object(mod, "load_symbol_maps", return_value=maps), \
            patch.object(mod, "parse_rda_bytes", return_value={}), \
            patch.object(mod, "regulon_to_edges", return_value=(EDGES, 0)):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert "FAILED for: brca" in str(exc.value)
    assert not (out / "brca" / "network.csv").exists()
