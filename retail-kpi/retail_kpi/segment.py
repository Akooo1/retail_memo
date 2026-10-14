"""Правила сегментации Mass / Affluent / Premium.

Сегмент = старший из двух уровней: по среднему остатку за 3 месяца и по
заявленному доходу. Пороги включительные (значение, равное порогу,
попадает в старший сегмент) и задаются в ``config``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_kpi import config


def tier_by_thresholds(values: pd.Series, affluent_min: float, premium_min: float) -> np.ndarray:
    """Map values to tier ranks: 0 = Mass, 1 = Affluent, 2 = Premium (inclusive bounds)."""
    values = np.asarray(values, dtype=float)
    return np.select([values >= premium_min, values >= affluent_min], [2, 1], default=0)


def assign_segment(avg_balance_3m_uzs: pd.Series, declared_income_uzs: pd.Series) -> pd.Series:
    """Return the segment name: the higher of the balance tier and the income tier."""
    balance_tier = tier_by_thresholds(
        avg_balance_3m_uzs, config.BALANCE_AFFLUENT_MIN_UZS, config.BALANCE_PREMIUM_MIN_UZS
    )
    income_tier = tier_by_thresholds(
        declared_income_uzs, config.INCOME_AFFLUENT_MIN_UZS, config.INCOME_PREMIUM_MIN_UZS
    )
    tier = np.maximum(balance_tier, income_tier)
    return pd.Series(np.asarray(config.SEGMENTS)[tier], index=avg_balance_3m_uzs.index, name="segment")
