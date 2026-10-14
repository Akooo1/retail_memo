"""Полный прогон: генерация CSV -> витрина -> KPI -> вход для LLM -> manifest.

Запуск: ``python -m retail_kpi --out build``. Одинаковые параметры дают
байт-в-байт одинаковые файлы; ``manifest.json`` содержит их SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from retail_kpi import config
from retail_kpi.generate import generate_all, write_raw
from retail_kpi.kpi import kpi_by_month_segment, segment_migration
from retail_kpi.llm_input import build_llm_payload, render_prompt
from retail_kpi.mart import build_client_month, load_raw


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV with fixed settings (no index, ``\\n`` line endings) for stable bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def run_pipeline(
    out_dir: Path,
    seed: int = config.SEED,
    n_clients: int = config.N_CLIENTS,
    n_months: int = config.N_MONTHS,
) -> dict[str, Path]:
    """Run every step into ``out_dir`` and return ``{relative name: path}`` of all outputs.

    The mart is built from the CSVs just written, not from in-memory frames,
    so the run also proves the mart can be rebuilt from ``data/`` alone.
    """
    out_dir = Path(out_dir)
    outputs: dict[str, Path] = {}

    raw = generate_all(seed=seed, n_clients=n_clients, n_months=n_months)
    for name, path in write_raw(raw, out_dir / "data").items():
        outputs[f"data/{name}.csv"] = path

    loaded = load_raw(out_dir / "data")
    mart = build_client_month(loaded["clients"], loaded["products"], loaded["transactions_monthly"])
    kpi = kpi_by_month_segment(mart)
    migration = segment_migration(mart)
    outputs["mart/client_month.csv"] = write_csv(mart, out_dir / "mart" / "client_month.csv")
    outputs["mart/kpi_monthly.csv"] = write_csv(kpi, out_dir / "mart" / "kpi_monthly.csv")
    outputs["mart/segment_migration.csv"] = write_csv(
        migration, out_dir / "mart" / "segment_migration.csv"
    )

    payload = build_llm_payload(kpi, migration)
    llm_dir = out_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    payload_path = llm_dir / "llm_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt_path = llm_dir / "llm_prompt.txt"
    prompt_path.write_text(render_prompt(payload), encoding="utf-8")
    outputs["llm/llm_payload.json"] = payload_path
    outputs["llm/llm_prompt.txt"] = prompt_path

    manifest = {
        "data_origin": config.DATA_ORIGIN,
        "disclaimer": config.DISCLAIMER,
        "params": {
            "seed": seed,
            "n_clients": n_clients,
            "n_months": n_months,
            "start_month": config.START_MONTH,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "files": {name: sha256_file(path) for name, path in sorted(outputs.items())},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs["manifest.json"] = manifest_path
    return outputs


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: parse arguments, run the pipeline, print a short summary."""
    parser = argparse.ArgumentParser(
        prog="python -m retail_kpi",
        description="Синтетическая розничная витрина: генерация, сегменты, KPI.",
    )
    parser.add_argument("--out", type=Path, default=Path("build"), help="папка для результатов")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--clients", type=int, default=config.N_CLIENTS)
    parser.add_argument("--months", type=int, default=config.N_MONTHS)
    args = parser.parse_args(argv)

    outputs = run_pipeline(args.out, seed=args.seed, n_clients=args.clients, n_months=args.months)
    print(config.DISCLAIMER)
    for name, path in outputs.items():
        print(f"  {name:<28} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
