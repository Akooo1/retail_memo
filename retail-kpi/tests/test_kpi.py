"""KPI и миграция: точные значения на ручном датасете и инварианты на прогоне."""

import math

import pytest

from retail_kpi.kpi import (
    KPI_COLUMNS,
    kpi_by_month_segment,
    migration_summary,
    safe_ratio,
    segment_migration,
)


def row(kpi, month, segment):
    rows = kpi[(kpi["month"] == month) & (kpi["segment"] == segment)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_safe_ratio():
    assert safe_ratio(1, 3) == 0.3333
    assert math.isnan(safe_ratio(1, 0))


def test_exact_kpis_for_last_month(tiny_mart):
    kpi = kpi_by_month_segment(tiny_mart)
    assert list(kpi.columns) == KPI_COLUMNS
    r = row(kpi, "2025-04", "ALL")
    assert r["clients"] == 3
    assert r["active_clients"] == 1
    assert r["active_share"] == 0.3333
    assert r["dormant_3m_share"] == 0.6667       # clients 1 and 2
    assert r["churn_rate"] == 0.5                # base: clients 1, 3; new churn: client 1
    assert r["avg_balance_uzs"] == 80_333_333    # (40M + 1M + 200M) / 3
    assert r["median_balance_uzs"] == 40_000_000
    assert r["txn_count_per_active"] == 3.0
    assert r["txn_volume_per_active_uzs"] == 300_000
    assert r["products_per_client"] == 0.6667   # card(1) + deposit(2) + nothing(3)
    assert r["card_penetration"] == 0.3333
    assert r["deposit_penetration"] == 0.3333
    assert r["loan_penetration"] == 0.0
    assert r["is_synthetic"]


def test_undefined_kpis_are_nan_not_zero(tiny_mart):
    kpi = kpi_by_month_segment(tiny_mart)
    first = row(kpi, "2025-01", "ALL")
    assert math.isnan(first["dormant_3m_share"]) and math.isnan(first["churn_rate"])
    third = row(kpi, "2025-03", "ALL")
    assert third["dormant_3m_share"] == 0.3333 and math.isnan(third["churn_rate"])
    affluent = row(kpi, "2025-04", "Affluent")  # clients 1 and 2, nobody active
    assert affluent["active_clients"] == 0 and math.isnan(affluent["txn_count_per_active"])


def test_empty_segment_is_absent(tiny_mart):
    kpi = kpi_by_month_segment(tiny_mart)
    assert kpi[(kpi["month"] == "2025-04") & (kpi["segment"] == "Mass")].empty


def test_segments_add_up_to_all(small_run):
    kpi = small_run["kpi"]
    for month, group in kpi.groupby("month"):
        total = group.loc[group["segment"] == "ALL", "clients"].item()
        assert group.loc[group["segment"] != "ALL", "clients"].sum() == total == 400


def test_shares_are_within_bounds(small_run):
    kpi = small_run["kpi"]
    share_cols = [c for c in kpi.columns if c.endswith(("_share", "_rate", "_penetration"))]
    values = kpi[share_cols].stack().dropna()
    assert ((values >= 0) & (values <= 1)).all()


def test_exact_migration(tiny_mart):
    migration = segment_migration(tiny_mart)
    got = {
        (r.from_segment, r.to_segment): (r.direction, r.clients)
        for r in migration.itertuples()
    }
    assert got == {
        ("Mass", "Affluent"): ("up", 1),
        ("Affluent", "Affluent"): ("stable", 1),
        ("Premium", "Premium"): ("stable", 1),
    }
    summary = migration_summary(migration)
    assert summary["clients"] == 3
    assert summary["upgraded_share"] == 0.3333
    assert summary["downgraded_share"] == 0.0
    assert summary["stable_share"] == 0.6667


def test_migration_shares_sum_to_one(small_run):
    migration = small_run["migration"]
    assert migration["clients"].sum() == 400
    sums = migration.groupby("from_segment")["share_of_from"].sum()
    assert sums.round(3).eq(1.0).all()
    summary = migration_summary(migration)
    total = summary["upgraded_share"] + summary["downgraded_share"] + summary["stable_share"]
    assert total == pytest.approx(1.0, abs=1e-3)
