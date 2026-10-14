"""Генератор синтетических клиентов розничного банка (суммы в UZS).

Все случайные величины берутся из одного numpy.random.Generator с
фиксированным seed и в фиксированном порядке, поэтому одинаковые
параметры всегда дают одинаковые CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retail_kpi import config

MALE_FIRST_NAMES = (
    "Aziz", "Bobur", "Jasur", "Sardor", "Otabek", "Rustam", "Akmal", "Dilshod",
    "Sherzod", "Javohir", "Farrux", "Ulug'bek", "Timur", "Bekzod", "Sanjar",
    "Islom", "Doniyor", "Jamshid", "Shoxrux", "Mirzo",
)
FEMALE_FIRST_NAMES = (
    "Dilnoza", "Madina", "Gulnora", "Nodira", "Shahnoza", "Zarina", "Malika",
    "Kamola", "Sevara", "Nilufar", "Feruza", "Mohira", "Dildora", "Barno",
    "Lola", "Munisa", "Sabina", "Umida", "Yulduz", "Zilola",
)
SURNAME_STEMS = (
    "Karimov", "Rahimov", "Yusupov", "Tursunov", "Abdullayev", "Ismoilov",
    "Nazarov", "Sobirov", "Qodirov", "Ergashev", "Mirzayev", "Alimov",
    "Xolmatov", "Saidov", "Usmonov", "Hasanov", "Jo'rayev", "Rasulov",
    "Normatov", "Bakirov",
)


def month_labels(start: str, n_months: int) -> list[str]:
    """Return ``n_months`` consecutive month labels ``YYYY-MM`` from ``start``."""
    return [str(p) for p in pd.period_range(start=start, periods=n_months, freq="M")]


def make_client_ids(n_clients: int) -> list[str]:
    """Return IDs ``SYN-000001`` ... — the prefix marks every client as synthetic."""
    return [f"{config.CLIENT_ID_PREFIX}{i:06d}" for i in range(1, n_clients + 1)]


def make_display_names(rng: np.random.Generator, n_clients: int) -> list[str]:
    """Return random Uzbek-style names, each prefixed with ``[SYN] ``."""
    is_female = rng.random(n_clients) < 0.5
    male = rng.choice(MALE_FIRST_NAMES, n_clients)
    female = rng.choice(FEMALE_FIRST_NAMES, n_clients)
    stems = rng.choice(SURNAME_STEMS, n_clients)
    names = []
    for fem, m, f, stem in zip(is_female, male, female, stems):
        surname = f"{stem}a" if fem else stem
        names.append(f"{config.NAME_PREFIX}{surname} {f if fem else m}")
    return names


def draw_latent_profile(
    rng: np.random.Generator, n_clients: int, n_months: int
) -> dict[str, np.ndarray]:
    """Draw hidden per-client traits that drive activity, balances and products.

    These traits are never written to CSV: the mart must recover segments
    and churn from observable data only.
    """
    income = np.exp(rng.normal(np.log(config.INCOME_MEDIAN_UZS), config.INCOME_SIGMA, n_clients))
    income = np.clip(np.round(income, -4), config.INCOME_MIN_UZS, config.INCOME_MAX_UZS)
    wealth_mult = np.exp(
        rng.normal(np.log(config.WEALTH_MULT_MEDIAN), config.WEALTH_MULT_SIGMA, n_clients)
    )
    engagement = rng.beta(5.0, 1.5, n_clients)
    is_churner = rng.random(n_clients) < config.CHURNER_SHARE
    churn_at = rng.integers(2, n_months, n_clients)
    dormant_from_start = rng.random(n_clients) < config.DORMANT_SHARE
    # Index of the first month without regular activity; n_months = never.
    churn_month = np.where(is_churner, churn_at, n_months)
    churn_month = np.where(dormant_from_start, 0, churn_month)
    return {
        "income": income.astype(np.int64),
        "base_balance": income * wealth_mult,
        "engagement": engagement,
        "churn_month": churn_month,
        "balance_drift": rng.normal(0.0, config.BALANCE_DRIFT_SIGMA, n_clients),
    }


def simulate_monthly_activity(
    rng: np.random.Generator, profile: dict[str, np.ndarray], n_months: int
) -> dict[str, np.ndarray]:
    """Simulate a (clients x months) matrix of txn count, volume and end balance."""
    n_clients = len(profile["income"])
    after_churn = np.arange(n_months)[None, :] >= profile["churn_month"][:, None]

    p_active = np.where(after_churn, config.P_ACTIVE_AFTER_CHURN, profile["engagement"][:, None])
    is_active = rng.random((n_clients, n_months)) < p_active
    lam = 4.0 + 12.0 * profile["engagement"]
    counts = rng.poisson(lam[:, None], (n_clients, n_months)) + 1
    counts = np.where(is_active, counts, 0)

    ticket = profile["income"] * config.TICKET_SHARE_OF_INCOME
    ticket_noise = np.exp(rng.normal(0.0, 0.35, (n_clients, n_months)))
    volume = np.round(counts * ticket[:, None] * ticket_noise, -3)

    shocks = rng.normal(0.0, config.BALANCE_MONTHLY_SIGMA, (n_clients, n_months))
    log_path = np.cumsum(shocks + profile["balance_drift"][:, None], axis=1)
    decay_path = np.cumprod(np.where(after_churn, config.CHURN_BALANCE_DECAY, 1.0), axis=1)
    balance = np.round(profile["base_balance"][:, None] * np.exp(log_path) * decay_path, -3)

    return {
        "txn_count": counts.astype(np.int64),
        "txn_volume_uzs": volume.astype(np.int64),
        "end_balance_uzs": balance.astype(np.int64),
    }


def draw_products(
    rng: np.random.Generator,
    profile: dict[str, np.ndarray],
    client_ids: list[str],
    months: list[str],
) -> pd.DataFrame:
    """Draw card / deposit / loan holdings with open and (optional) close months."""
    n_clients = len(client_ids)
    n_months = len(months)
    pre_period = [
        str(p)
        for p in pd.period_range(
            end=pd.Period(months[0], freq="M") - 1,
            periods=config.PRE_PERIOD_OPEN_MONTHS,
            freq="M",
        )
    ]
    balance_scale = np.minimum(profile["base_balance"] / 30_000_000, 1.0)
    hold_prob = {
        "card": 0.55 + 0.40 * profile["engagement"],
        "deposit": 0.08 + 0.55 * balance_scale,
        "loan": 0.30 - 0.15 * balance_scale,
    }
    churned_in_period = profile["churn_month"] < n_months

    frames = []
    for product in config.PRODUCTS:
        # Every array is drawn for all clients so the RNG stream does not
        # depend on which clients end up holding the product.
        holds = rng.random(n_clients) < hold_prob[product]
        opened_before = rng.random(n_clients) < 0.85
        pre_idx = rng.integers(0, len(pre_period), n_clients)
        in_idx = rng.integers(0, n_months, n_clients)
        close_draw = rng.random(n_clients)
        random_close_idx = rng.integers(0, n_months, n_clients)
        churn_lag = rng.integers(0, 2, n_clients)

        open_idx = np.where(opened_before, -1, in_idx)
        close_prob = np.where(churned_in_period, 0.5, 0.03)
        close_idx = np.where(churned_in_period, profile["churn_month"] + churn_lag, random_close_idx)
        closes = (close_draw < close_prob) & (close_idx > open_idx) & (close_idx < n_months)

        for i in np.flatnonzero(holds):
            frames.append(
                {
                    "client_id": client_ids[i],
                    "product_type": product,
                    "open_month": pre_period[pre_idx[i]] if opened_before[i] else months[in_idx[i]],
                    "close_month": months[close_idx[i]] if closes[i] else None,
                }
            )

    products = pd.DataFrame(frames, columns=["client_id", "product_type", "open_month", "close_month"])
    products["product_type"] = pd.Categorical(products["product_type"], categories=config.PRODUCTS)
    products = products.sort_values(["client_id", "product_type"], kind="stable").reset_index(drop=True)
    products["product_type"] = products["product_type"].astype(str)
    products["is_synthetic"] = True
    return products


def generate_all(
    seed: int = config.SEED,
    n_clients: int = config.N_CLIENTS,
    n_months: int = config.N_MONTHS,
    start_month: str = config.START_MONTH,
) -> dict[str, pd.DataFrame]:
    """Generate the three raw tables: ``clients``, ``products``, ``transactions_monthly``."""
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    if n_months < config.DORMANT_MONTHS + 1:
        raise ValueError(f"n_months must be >= {config.DORMANT_MONTHS + 1} to measure churn")

    rng = np.random.default_rng(seed)
    months = month_labels(start_month, n_months)
    client_ids = make_client_ids(n_clients)
    names = make_display_names(rng, n_clients)
    profile = draw_latent_profile(rng, n_clients, n_months)
    activity = simulate_monthly_activity(rng, profile, n_months)
    products = draw_products(rng, profile, client_ids, months)

    clients = pd.DataFrame(
        {
            "client_id": client_ids,
            "display_name": names,
            "declared_income_uzs": profile["income"],
            "is_synthetic": True,
        }
    )
    transactions = pd.DataFrame(
        {
            "client_id": np.repeat(client_ids, n_months),
            "month": np.tile(months, n_clients),
            "txn_count": activity["txn_count"].ravel(),
            "txn_volume_uzs": activity["txn_volume_uzs"].ravel(),
            "end_balance_uzs": activity["end_balance_uzs"].ravel(),
            "is_synthetic": True,
        }
    )
    return {"clients": clients, "products": products, "transactions_monthly": transactions}


def write_raw(tables: dict[str, pd.DataFrame], data_dir: Path) -> dict[str, Path]:
    """Write each raw table to ``<data_dir>/<name>.csv`` and return the paths."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in tables.items():
        path = data_dir / f"{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        paths[name] = path
    return paths
