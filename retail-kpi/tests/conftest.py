"""Общие фикстуры: маленький прогон конвейера и ручной датасет из 3 клиентов."""

import pandas as pd
import pytest

from retail_kpi.kpi import kpi_by_month_segment, segment_migration
from retail_kpi.mart import build_client_month, load_raw
from retail_kpi.pipeline import run_pipeline


@pytest.fixture(scope="session")
def small_run(tmp_path_factory):
    """Full pipeline on 400 clients x 6 months, plus the frames it produced."""
    out_dir = tmp_path_factory.mktemp("small_run")
    outputs = run_pipeline(out_dir, seed=7, n_clients=400, n_months=6)
    raw = load_raw(out_dir / "data")
    mart = build_client_month(raw["clients"], raw["products"], raw["transactions_monthly"])
    return {
        "out_dir": out_dir,
        "outputs": outputs,
        "raw": raw,
        "mart": mart,
        "kpi": kpi_by_month_segment(mart),
        "migration": segment_migration(mart),
    }


@pytest.fixture
def tiny_tables():
    """Three hand-made clients x 4 months whose KPIs are computed by hand in the tests.

    * SYN-000001: income 5M, balance 10->40M (Mass -> Affluent), active only in month 1.
    * SYN-000002: income 20M (Affluent by income), never active.
    * SYN-000003: income 50M (Premium), balance 200M, active every month.
    """
    months = ["2025-01", "2025-02", "2025-03", "2025-04"]
    clients = pd.DataFrame(
        {
            "client_id": ["SYN-000001", "SYN-000002", "SYN-000003"],
            "display_name": ["[SYN] Test A", "[SYN] Test B", "[SYN] Test C"],
            "declared_income_uzs": [5_000_000, 20_000_000, 50_000_000],
            "is_synthetic": True,
        }
    )
    balances = {
        "SYN-000001": [10_000_000, 20_000_000, 30_000_000, 40_000_000],
        "SYN-000002": [1_000_000] * 4,
        "SYN-000003": [200_000_000] * 4,
    }
    counts = {"SYN-000001": [5, 0, 0, 0], "SYN-000002": [0, 0, 0, 0], "SYN-000003": [3, 3, 3, 3]}
    rows = []
    for cid in clients["client_id"]:
        for i, month in enumerate(months):
            rows.append(
                {
                    "client_id": cid,
                    "month": month,
                    "txn_count": counts[cid][i],
                    "txn_volume_uzs": counts[cid][i] * 100_000,
                    "end_balance_uzs": balances[cid][i],
                    "is_synthetic": True,
                }
            )
    transactions = pd.DataFrame(rows)
    products = pd.DataFrame(
        [
            ("SYN-000001", "card", "2024-01", None),
            ("SYN-000001", "loan", "2025-02", "2025-04"),
            ("SYN-000002", "deposit", "2025-03", None),
            ("SYN-000003", "card", "2020-05", "2025-02"),
        ],
        columns=["client_id", "product_type", "open_month", "close_month"],
    )
    products["is_synthetic"] = True
    return {"clients": clients, "products": products, "transactions_monthly": transactions}


@pytest.fixture
def tiny_mart(tiny_tables):
    """Client-month mart built from ``tiny_tables``."""
    return build_client_month(
        tiny_tables["clients"], tiny_tables["products"], tiny_tables["transactions_monthly"]
    )
