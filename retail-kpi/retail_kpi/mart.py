"""Витрина client-month: одна строка = один синтетический клиент в одном месяце.

Витрина строится только из CSV, а не из скрытых параметров генератора,
поэтому её можно пересчитать из ``data/`` без запуска генерации.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_kpi import config
from retail_kpi.segment import assign_segment

MART_COLUMNS = [
    "client_id", "month", "segment", "declared_income_uzs", "end_balance_uzs",
    "avg_balance_3m_uzs", "txn_count", "txn_volume_uzs", "is_active", "dormant_3m",
    "in_churn_base", "new_churn", "has_card", "has_deposit", "has_loan", "n_products",
    "is_synthetic",
]


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read ``clients``, ``products`` and ``transactions_monthly`` CSVs from ``data_dir``."""
    data_dir = Path(data_dir)
    clients = pd.read_csv(data_dir / "clients.csv", dtype={"client_id": str, "display_name": str})
    products = pd.read_csv(
        data_dir / "products.csv",
        dtype={"client_id": str, "product_type": str, "open_month": str, "close_month": str},
    )
    transactions = pd.read_csv(
        data_dir / "transactions_monthly.csv", dtype={"client_id": str, "month": str}
    )
    return {"clients": clients, "products": products, "transactions_monthly": transactions}


def product_holdings(products: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Return per client-month flags ``has_<product>`` and ``n_products`` (open products only).

    A product is open in month m if ``open_month <= m`` and it is not closed,
    or ``close_month > m`` (the close month itself counts as closed).
    """
    flag_cols = [f"has_{p}" for p in config.PRODUCTS]
    grid = products[["client_id", "product_type", "open_month", "close_month"]].merge(
        pd.DataFrame({"month": months}), how="cross"
    )
    is_open = (grid["open_month"] <= grid["month"]) & (
        grid["close_month"].isna() | (grid["close_month"] > grid["month"])
    )
    held = grid.loc[is_open, ["client_id", "month", "product_type"]]
    if held.empty:
        return pd.DataFrame(columns=["client_id", "month", *flag_cols, "n_products"])
    flags = (
        held.assign(held=True)
        .pivot_table(index=["client_id", "month"], columns="product_type", values="held", aggfunc="any")
        .reindex(columns=list(config.PRODUCTS))
        .fillna(False)
        .astype(bool)
    )
    flags.columns = flag_cols
    flags["n_products"] = flags[flag_cols].sum(axis=1).astype("int64")
    return flags.reset_index()


def activity_flags(client_month: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_active``, ``dormant_3m``, ``in_churn_base`` and ``new_churn`` columns.

    * ``dormant_3m`` — no transactions in this and the two previous months;
      empty (NA) for the first two months, where there is not enough history.
    * ``in_churn_base`` — the client was NOT dormant in the previous month,
      i.e. can churn this month; NA when the previous month is undefined.
    * ``new_churn`` — the client became dormant exactly in this month.

    Expects rows sorted by ``client_id, month`` with every client present
    in every month (checked by ``build_client_month``).
    """
    out = client_month.copy()
    out["is_active"] = out["txn_count"] > 0
    inactive = (~out["is_active"]).astype(float)
    by_client = inactive.groupby(out["client_id"])
    window = inactive.copy()
    for lag in range(1, config.DORMANT_MONTHS):
        window = window + by_client.shift(lag)  # NaN while history is too short
    dormant = (window == config.DORMANT_MONTHS).astype("boolean").mask(window.isna())
    in_churn_base = ~dormant.groupby(out["client_id"]).shift(1)
    out["dormant_3m"] = dormant
    out["in_churn_base"] = in_churn_base
    out["new_churn"] = (dormant & in_churn_base).mask(in_churn_base.isna())
    return out


def rolling_balance(client_month: pd.DataFrame) -> pd.Series:
    """Average end balance over the current and previous months (window from config).

    Early months use whatever history exists (1 or 2 months).
    """
    by_client = client_month.groupby("client_id")["end_balance_uzs"]
    lagged = [client_month["end_balance_uzs"]] + [
        by_client.shift(lag) for lag in range(1, config.BALANCE_WINDOW_MONTHS)
    ]
    return pd.concat(lagged, axis=1).mean(axis=1).round().astype("int64")


def build_client_month(
    clients: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    """Join raw tables into the segmented client-month mart (columns: ``MART_COLUMNS``)."""
    months = sorted(transactions["month"].unique())
    per_client = transactions.groupby("client_id")["month"].nunique()
    if (per_client != len(months)).any():
        raise ValueError("every client must have exactly one row per month")
    if transactions.duplicated(["client_id", "month"]).any():
        raise ValueError("duplicate (client_id, month) rows in transactions")

    cm = transactions.drop(columns=["is_synthetic"], errors="ignore").merge(
        clients[["client_id", "declared_income_uzs"]], on="client_id", how="left", validate="many_to_one"
    )
    if cm["declared_income_uzs"].isna().any():
        raise ValueError("transactions reference unknown client_id")
    cm = cm.sort_values(["client_id", "month"], kind="stable").reset_index(drop=True)

    cm["avg_balance_3m_uzs"] = rolling_balance(cm)
    cm["segment"] = assign_segment(cm["avg_balance_3m_uzs"], cm["declared_income_uzs"])
    cm = activity_flags(cm)

    holdings = product_holdings(products, months)
    cm = cm.merge(holdings, on=["client_id", "month"], how="left")
    for col in [f"has_{p}" for p in config.PRODUCTS]:
        cm[col] = cm[col].fillna(False).astype(bool)
    cm["n_products"] = cm["n_products"].fillna(0).astype("int64")
    cm["is_synthetic"] = True
    return cm[MART_COLUMNS]
