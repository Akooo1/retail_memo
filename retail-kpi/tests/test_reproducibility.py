"""Повторяемость: два прогона с одними параметрами дают одинаковые байты."""

import json

from retail_kpi import config
from retail_kpi.pipeline import main, run_pipeline, sha256_file


def file_hashes(out_dir):
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["files"]


def test_default_run_is_byte_identical(tmp_path):
    """The exact README run (default parameters) twice -> identical file hashes."""
    first = run_pipeline(tmp_path / "a")
    second = run_pipeline(tmp_path / "b")
    hashes_a, hashes_b = file_hashes(tmp_path / "a"), file_hashes(tmp_path / "b")
    assert hashes_a == hashes_b
    assert len(hashes_a) == 8
    for name in hashes_a:
        assert sha256_file(first[name]) == sha256_file(second[name])


def test_manifest_records_parameters(tmp_path):
    run_pipeline(tmp_path, seed=3, n_clients=50, n_months=4)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_origin"] == config.DATA_ORIGIN
    assert manifest["params"]["seed"] == 3
    assert set(manifest["versions"]) == {"python", "numpy", "pandas"}


def test_different_seed_changes_data(tmp_path):
    run_pipeline(tmp_path / "a", seed=1, n_clients=100, n_months=4)
    run_pipeline(tmp_path / "b", seed=2, n_clients=100, n_months=4)
    a, b = file_hashes(tmp_path / "a"), file_hashes(tmp_path / "b")
    assert a["data/transactions_monthly.csv"] != b["data/transactions_monthly.csv"]


def test_cli_entry_point(tmp_path, capsys):
    assert main(["--out", str(tmp_path), "--clients", "60", "--months", "4", "--seed", "5"]) == 0
    assert "СИНТЕТИЧЕСКИЕ ДАННЫЕ" in capsys.readouterr().out
    assert (tmp_path / "mart" / "kpi_monthly.csv").exists()
