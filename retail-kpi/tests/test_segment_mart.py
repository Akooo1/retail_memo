"""Правила сегментации и витрина client-month на ручном датасете."""

import pandas as pd
import pytest

from retail_kpi.mart import MART_COLUMNS, build_client_month, product_holdings
from retail_kpi.segment import assign_segment, tier_by_thresholds


def test_tier_boundaries_are_inclusive():
    values = pd.Series([0, 24_999_999, 25_000_000, 149_999_999, 150_000_000])
    assert tier_by_thresholds(values, 25_000_000, 150_000_000).tolist() == [0, 0, 1, 1, 2]


def test_segment_is_the_higher_of_balance_and_income_tier():
    balance = pd.Series([1_000_000, 30_000_000, 1_000_000, 200_000_000])
    income = pd.Series([1_000_000, 1_000_000, 45_000_000, 16_000_000])
    assert assign_segment(balance, income).tolist() == ["Mass", "Affluent", "Premium", "Premium"]


def by_client(mart, client_id, column):
    return mart.loc[mart["client_id"] == client_id, column].tolist()


def test_mart_grain_and_columns(tiny_mart):
    assert list(tiny_mart.columns) == MART_COLUMNS
    assert len(tiny_mart) == 3 * 4
    assert not tiny_mart.duplicated(["client_id", "month"]).any()
    assert tiny_mart["is_synthetic"].all()
    assert "display_name" not in tiny_mart.columns


def test_rolling_balance_and_segments(tiny_mart):
    assert by_client(tiny_mart, "SYN-000001", "avg_balance_3m_uzs") == [
        10_000_000, 15_000_000, 20_000_000, 30_000_000,
    ]
    assert by_client(tiny_mart, "SYN-000001", "segment") == ["Mass", "Mass", "Mass", "Affluent"]
    assert set(by_client(tiny_mart, "SYN-000002", "segment")) == {"Affluent"}
    assert set(by_client(tiny_mart, "SYN-000003", "segment")) == {"Premium"}


def as_list(values):
    return [None if pd.isna(v) else bool(v) for v in values]


def test_dormancy_and_churn_flags(tiny_mart):
    a = tiny_mart[tiny_mart["client_id"] == "SYN-000001"]
    assert as_list(a["dormant_3m"]) == [None, None, False, True]
    assert as_list(a["in_churn_base"]) == [None, None, None, True]
    assert as_list(a["new_churn"]) == [None, None, None, True]
    b = tiny_mart[tiny_mart["client_id"] == "SYN-000002"]
    assert as_list(b["dormant_3m"]) == [None, None, True, True]
    assert as_list(b["in_churn_base"]) == [None, None, None, False]  # already dormant
    assert as_list(b["new_churn"]) == [None, None, None, False]


def test_product_flags_respect_open_and_close_months(tiny_mart):
    assert by_client(tiny_mart, "SYN-000001", "has_loan") == [False, True, True, False]
    assert by_client(tiny_mart, "SYN-000001", "n_products") == [1, 2, 2, 1]
    assert by_client(tiny_mart, "SYN-000002", "has_deposit") == [False, False, True, True]
    assert by_client(tiny_mart, "SYN-000003", "has_card") == [True, False, False, False]


def test_product_holdings_handles_no_products():
    empty = pd.DataFrame(columns=["client_id", "product_type", "open_month", "close_month"])
    assert product_holdings(empty, ["2025-01"]).empty


def test_mart_rejects_missing_month(tiny_tables):
    tx = tiny_tables["transactions_monthly"]
    tx = tx[~((tx["client_id"] == "SYN-000002") & (tx["month"] == "2025-03"))]
    with pytest.raises(ValueError, match="one row per month"):
        build_client_month(tiny_tables["clients"], tiny_tables["products"], tx)


def test_mart_rejects_unknown_client(tiny_tables):
    clients = tiny_tables["clients"].iloc[:2]
    with pytest.raises(ValueError, match="unknown client_id"):
        build_client_month(clients, tiny_tables["products"], tiny_tables["transactions_monthly"])


def test_mart_from_csv_matches_rebuild(small_run):
    on_disk = pd.read_csv(small_run["out_dir"] / "mart" / "client_month.csv")
    assert len(on_disk) == len(small_run["mart"])
    assert on_disk["segment"].tolist() == small_run["mart"]["segment"].tolist()
