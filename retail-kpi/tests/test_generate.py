"""Генератор: форма таблиц, маркировка синтетики, инварианты сумм, детерминизм."""

import pandas as pd
import pytest

from retail_kpi import config
from retail_kpi.generate import generate_all, make_client_ids, month_labels


def test_month_labels_are_consecutive():
    assert month_labels("2025-11", 3) == ["2025-11", "2025-12", "2026-01"]


def test_table_shapes(small_run):
    raw = small_run["raw"]
    assert len(raw["clients"]) == 400
    assert len(raw["transactions_monthly"]) == 400 * 6
    assert not raw["transactions_monthly"].duplicated(["client_id", "month"]).any()
    assert raw["clients"]["client_id"].is_unique


def test_every_raw_row_is_marked_synthetic(small_run):
    for name, frame in small_run["raw"].items():
        assert frame["is_synthetic"].all(), name


def test_ids_and_names_carry_synthetic_markers(small_run):
    clients = small_run["raw"]["clients"]
    assert clients["client_id"].str.fullmatch(r"SYN-\d{6}").all()
    assert clients["display_name"].str.startswith(config.NAME_PREFIX).all()
    assert make_client_ids(2) == ["SYN-000001", "SYN-000002"]


def test_amounts_are_consistent(small_run):
    tx = small_run["raw"]["transactions_monthly"]
    assert (tx[["txn_count", "txn_volume_uzs", "end_balance_uzs"]] >= 0).all().all()
    assert ((tx["txn_count"] > 0) == (tx["txn_volume_uzs"] > 0)).all()
    income = small_run["raw"]["clients"]["declared_income_uzs"]
    assert income.between(config.INCOME_MIN_UZS, config.INCOME_MAX_UZS).all()


def test_products_are_valid(small_run):
    products = small_run["raw"]["products"]
    assert set(products["product_type"]) <= set(config.PRODUCTS)
    assert not products.duplicated(["client_id", "product_type"]).any()
    closed = products.dropna(subset=["close_month"])
    assert (closed["close_month"] > closed["open_month"]).all()
    assert set(products["client_id"]) <= set(small_run["raw"]["clients"]["client_id"])


def test_same_seed_gives_identical_tables():
    a = generate_all(seed=11, n_clients=200, n_months=5)
    b = generate_all(seed=11, n_clients=200, n_months=5)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_gives_different_data():
    a = generate_all(seed=11, n_clients=200, n_months=5)
    b = generate_all(seed=12, n_clients=200, n_months=5)
    assert not a["transactions_monthly"].equals(b["transactions_monthly"])


@pytest.mark.parametrize("kwargs", [{"n_clients": 0}, {"n_months": 3}])
def test_rejects_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        generate_all(**kwargs)
