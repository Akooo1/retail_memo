"""Вход для LLM: только агрегированные KPI, без данных на уровне клиента.

Это техническая граница автоматизации из ``usecase-final.md``: модель
получает агрегаты по сегментам, а сегменты меньше ``MIN_GROUP_SIZE``
клиентов не передаются вовсе. Вызова API здесь нет: модуль собирает
payload и текст промпта, которые можно проверить глазами и тестами.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from string import Template

import pandas as pd

from retail_kpi import config
from retail_kpi.kpi import migration_summary

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "kpi_commentary_ru.md"

LLM_KPI_FIELDS = [
    "clients", "active_share", "dormant_3m_share", "churn_rate", "avg_balance_uzs",
    "txn_volume_per_active_uzs", "products_per_client", "card_penetration",
    "deposit_penetration", "loan_penetration",
]
FORBIDDEN_KEYS = {"client_id", "display_name", "declared_income_uzs", "end_balance_uzs"}
CLIENT_ID_PATTERN = re.compile(re.escape(config.CLIENT_ID_PREFIX) + r"\d+")


def to_json_value(value):
    """Convert pandas/numpy scalars to plain JSON values; NaN and NA become ``None``."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
    return value


def kpi_snapshot(kpi: pd.DataFrame, month: str, segment: str) -> dict | None:
    """Return the LLM-visible KPI fields for one (month, segment), or None if absent."""
    row = kpi[(kpi["month"] == month) & (kpi["segment"] == segment)]
    if row.empty:
        return None
    record = row.iloc[0]
    return {field: to_json_value(record[field]) for field in LLM_KPI_FIELDS}


def kpi_delta(current: dict, previous: dict) -> dict:
    """Return current minus previous for each field; None where either side is None."""
    delta = {}
    for field in LLM_KPI_FIELDS:
        a, b = current.get(field), previous.get(field)
        delta[field] = None if a is None or b is None else round(a - b, 4)
    return delta


def assert_aggregates_only(payload: dict) -> None:
    """Raise ValueError if the payload contains client IDs, names or client-level keys."""

    def walk_keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from walk_keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk_keys(item)

    bad_keys = FORBIDDEN_KEYS.intersection(walk_keys(payload))
    if bad_keys:
        raise ValueError(f"client-level keys in LLM payload: {sorted(bad_keys)}")
    text = json.dumps(payload, ensure_ascii=False)
    if CLIENT_ID_PATTERN.search(text):
        raise ValueError("client ID found in LLM payload")
    if config.NAME_PREFIX.strip() in text:
        raise ValueError("synthetic client name found in LLM payload")


def build_llm_payload(
    kpi: pd.DataFrame,
    migration: pd.DataFrame,
    month: str | None = None,
    min_group_size: int = config.MIN_GROUP_SIZE,
) -> dict:
    """Build the aggregated LLM input for ``month`` (default: last month) vs. the previous month.

    Segments with fewer than ``min_group_size`` clients in either month are
    listed in ``suppressed_segments`` without any numbers.
    """
    months = sorted(kpi["month"].unique())
    month = month or months[-1]
    if month not in months or months.index(month) == 0:
        raise ValueError(f"month {month!r} needs a previous month in the KPI table")
    previous = months[months.index(month) - 1]

    segments, suppressed = [], []
    for segment in (config.ALL_SEGMENTS_LABEL, *config.SEGMENTS):
        current_kpi = kpi_snapshot(kpi, month, segment)
        previous_kpi = kpi_snapshot(kpi, previous, segment)
        too_small = (
            current_kpi is None
            or previous_kpi is None
            or current_kpi["clients"] < min_group_size
            or previous_kpi["clients"] < min_group_size
        )
        if too_small:
            suppressed.append(segment)
            continue
        segments.append(
            {
                "segment": segment,
                "current": current_kpi,
                "previous": previous_kpi,
                "delta": kpi_delta(current_kpi, previous_kpi),
            }
        )

    summary = migration_summary(migration)
    payload = {
        "data_origin": config.DATA_ORIGIN,
        "disclaimer": config.DISCLAIMER,
        "currency": "UZS",
        "period": {"current": month, "previous": previous},
        "segments": segments,
        "suppressed_segments": suppressed,
        "migration": summary if summary["clients"] >= min_group_size else None,
        "min_group_size": min_group_size,
    }
    assert_aggregates_only(payload)
    return payload


def load_prompt_template(path: Path = PROMPT_PATH) -> Template:
    """Read the prompt template (``$payload_json`` is the only placeholder)."""
    return Template(Path(path).read_text(encoding="utf-8"))


def render_prompt(payload: dict, template: Template | None = None) -> str:
    """Re-check the payload and substitute it into the prompt template as JSON."""
    assert_aggregates_only(payload)
    template = template or load_prompt_template()
    return template.substitute(payload_json=json.dumps(payload, ensure_ascii=False, indent=2))
