"""
Tests for the Zenodo / directory input modes of scripts/extract_tcga_networks.py.

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


def _fake_urlopen(record_bytes: bytes, files: dict, tampered: dict = None):
    tampered = tampered or {}

    def opener(url, timeout=None):
        if url == "https://zenodo.org/api/records/22918956":
            return _response(record_bytes)
        name = url.rsplit("/", 1)[-1]
        return _response(tampered.get(name, files[name]))

    return opener


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


def test_load_from_dir_uses_regulon_filename(mod, tmp_path):
    (tmp_path / "regulonbrca.rda").write_bytes(b"x")
    sentinel = {"1": "regulon"}
    with patch.object(mod, "_parse_rda_bytes", return_value=sentinel) as parse:
        result = mod.load_rda_from_dir(str(tmp_path), "brca")
    assert result is sentinel
    parse.assert_called_once_with(b"x", "brca")


def test_rda_names_cover_all_cancer_types(mod):
    assert set(mod.RDA_NAMES) == set(mod.CANCER_TYPE_MAP)
    assert all(name == f"regulon{ct}.rda" for ct, name in mod.RDA_NAMES.items())


def test_main_rejects_both_tarball_and_rda_dir(mod, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--tarball", "a.tgz", "--rda-dir", "d"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


def test_main_zenodo_failure_exits_with_fallback_hint(mod, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--cancer-type", "brca", "--download-dir", str(tmp_path)])
    with patch.object(mod, "download_from_zenodo", side_effect=RuntimeError("offline")):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "offline" in out and "--rda-dir" in out and "--tarball" in out
