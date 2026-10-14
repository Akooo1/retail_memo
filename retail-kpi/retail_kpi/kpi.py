"""KPI по месяцам и сегментам и миграция между сегментами.

Все KPI считаются из витрины client-month (``mart.build_client_month``).
Значение NaN (пустая ячейка в CSV) означает «не определено», а не ноль:
например, отток нельзя посчитать, пока нет трёх месяцев истории.
"""

from __future__ import annotations

import math

import pandas as pd

from retail_kpi import config

KPI_COLUMNS = [
    "month", "segment", "clients", "active_clients", "active_share", "dormant_3m_share",
    "churn_rate", "avg_balance_uzs", "median_balance_uzs", "txn_count_per_active",
    "txn_volume_per_active_uzs", "products_per_client", "card_penetration",
    "deposit_penetration", "loan_penetration", "is_synthetic",
]


def safe_ratio(numerator: float, denominator: float, ndigits: int = 4) -> float:
    """Return ``numerator / denominator`` rounded, or NaN when the denominator is 0."""
    if denominator == 0:
        return math.nan
    return round(float(numerator) / float(denominator), ndigits)


def group_kpis(group: pd.DataFrame) -> dict[str, float]:
    """Compute every KPI for one group of client-month rows (one month, one segment)."""
    clients = len(group)
    active = int(group["is_active"].sum())
    dormant_known = group["dormant_3m"].notna()
    churn_base = group["in_churn_base"].fillna(False).astype(bool)
    has_churn_history = group["in_churn_base"].notna().any()
    return {
        "clients": clients,
        "active_clients": active,
        "active_share": safe_ratio(active, clients),
        "dormant_3m_share": safe_ratio(
            group.loc[dormant_known, "dormant_3m"].astype(bool).sum(), dormant_known.sum()
        ),
        "churn_rate": (
            safe_ratio(group.loc[churn_base, "new_churn"].astype(bool).sum(), churn_base.sum())
            if has_churn_history
            else math.nan
        ),
        "avg_balance_uzs": int(round(group["end_balance_uzs"].mean())),
        "median_balance_uzs": int(round(group["end_balance_uzs"].median())),
        "txn_count_per_active": safe_ratio(group["txn_count"].sum(), active, 2),
        "txn_volume_per_active_uzs": safe_ratio(group["txn_volume_uzs"].sum(), active, 0),
        "products_per_client": safe_ratio(group["n_products"].sum(), clients),
        "card_penetration": safe_ratio(group["has_card"].sum(), clients),
        "deposit_penetration": safe_ratio(group["has_deposit"].sum(), clients),
        "loan_penetration": safe_ratio(group["has_loan"].sum(), clients),
    }


def kpi_by_month_segment(mart: pd.DataFrame) -> pd.DataFrame:
    """Return one KPI row per (month, segment) plus an ``ALL`` row per month.

    Segments with no clients in a month are absent rather than zero.
    """
    rows = []
    for month, month_rows in mart.groupby("month", sort=True):
        rows.append({"month": month, "segment": config.ALL_SEGMENTS_LABEL, **group_kpis(month_rows)})
        for segment in config.SEGMENTS:
            seg_rows = month_rows[month_rows["segment"] == segment]
            if len(seg_rows):
                rows.append({"month": month, "segment": segment, **group_kpis(seg_rows)})
    kpi = pd.DataFrame(rows)
    kpi["txn_volume_per_active_uzs"] = kpi["txn_volume_per_active_uzs"].round().astype("Int64")
    kpi["is_synthetic"] = True
    return kpi[KPI_COLUMNS]


def segment_migration(
    mart: pd.DataFrame, from_month: str | None = None, to_month: str | None = None
) -> pd.DataFrame:
    """Count clients by (segment in ``from_month``) -> (segment in ``to_month``).

    Defaults to the first and the last month of the mart. ``share_of_from``
    is the share of the from-segment that ended up in the to-segment;
    ``direction`` is ``up``, ``down`` or ``stable`` by segment rank.
    """
    months = sorted(mart["month"].unique())
    from_month = from_month or months[0]
    to_month = to_month or months[-1]
    start = mart.loc[mart["month"] == from_month, ["client_id", "segment"]]
    end = mart.loc[mart["month"] == to_month, ["client_id", "segment"]]
    pairs = start.merge(end, on="client_id", suffixes=("_from", "_to"), validate="one_to_one")

    counts = (
        pairs.groupby(["segment_from", "segment_to"]).size().rename("clients").reset_index()
        .rename(columns={"segment_from": "from_segment", "segment_to": "to_segment"})
    )
    counts["share_of_from"] = (
        counts["clients"] / counts.groupby("from_segment")["clients"].transform("sum")
    ).round(4)
    rank_from = counts["from_segment"].map(config.SEGMENT_RANK)
    rank_to = counts["to_segment"].map(config.SEGMENT_RANK)
    counts["direction"] = "stable"
    counts.loc[rank_to > rank_from, "direction"] = "up"
    counts.loc[rank_to < rank_from, "direction"] = "down"
    counts = counts.sort_values(
        ["from_segment", "to_segment"], key=lambda s: s.map(config.SEGMENT_RANK)
    ).reset_index(drop=True)
    counts.insert(0, "from_month", from_month)
    counts.insert(1, "to_month", to_month)
    counts["is_synthetic"] = True
    return counts[
        ["from_month", "to_month", "from_segment", "to_segment", "direction",
         "clients", "share_of_from", "is_synthetic"]
    ]


def migration_summary(migration: pd.DataFrame) -> dict[str, float]:
    """Collapse the migration table into total client count and up/down/stable shares."""
    total = int(migration["clients"].sum())
    by_direction = migration.groupby("direction")["clients"].sum()
    return {
        "from_month": migration["from_month"].iloc[0],
        "to_month": migration["to_month"].iloc[0],
        "clients": total,
        "upgraded_share": safe_ratio(by_direction.get("up", 0), total),
        "downgraded_share": safe_ratio(by_direction.get("down", 0), total),
        "stable_share": safe_ratio(by_direction.get("stable", 0), total),
    }
