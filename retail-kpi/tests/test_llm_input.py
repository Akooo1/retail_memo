"""Граница автоматизации: в LLM уходят только агрегаты, малые сегменты скрыты."""

import copy
import re

import pytest

from retail_kpi import config
from retail_kpi.kpi import kpi_by_month_segment, segment_migration
from retail_kpi.llm_input import (
    LLM_KPI_FIELDS,
    assert_aggregates_only,
    build_llm_payload,
    kpi_delta,
    kpi_snapshot,
    load_prompt_template,
    render_prompt,
    to_json_value,
)


@pytest.fixture
def payload(small_run):
    return build_llm_payload(small_run["kpi"], small_run["migration"], min_group_size=1)


def test_payload_is_labelled_synthetic(payload):
    assert payload["data_origin"] == "SYNTHETIC"
    assert "вымышлены" in payload["disclaimer"]
    assert payload["period"] == {"current": "2025-06", "previous": "2025-05"}


def test_payload_contains_no_client_level_data(payload):
    assert_aggregates_only(payload)  # does not raise
    text = str(payload)
    assert not re.search(r"SYN-\d", text)
    assert config.NAME_PREFIX.strip() not in text


@pytest.mark.parametrize(
    "inject",
    [
        lambda p: p["segments"][0].update(client_id="x"),
        lambda p: p["segments"][0]["current"].update(display_name="x"),
        lambda p: p.update(note="top client SYN-000042"),
        lambda p: p.update(note="[SYN] Karimov Aziz"),
    ],
)
def test_guard_rejects_client_level_data(payload, inject):
    tampered = copy.deepcopy(payload)
    inject(tampered)
    with pytest.raises(ValueError):
        assert_aggregates_only(tampered)
    with pytest.raises(ValueError):
        render_prompt(tampered)


def test_payload_numbers_match_kpi_table(payload, small_run):
    kpi = small_run["kpi"]
    for seg in payload["segments"]:
        assert seg["current"] == kpi_snapshot(kpi, "2025-06", seg["segment"])
        assert seg["previous"] == kpi_snapshot(kpi, "2025-05", seg["segment"])
        assert set(seg["current"]) == set(LLM_KPI_FIELDS)


def test_small_segments_are_suppressed(tiny_mart):
    kpi = kpi_by_month_segment(tiny_mart)
    migration = segment_migration(tiny_mart)
    strict = build_llm_payload(kpi, migration)  # default MIN_GROUP_SIZE = 30
    assert strict["segments"] == []
    assert strict["suppressed_segments"] == ["ALL", "Mass", "Affluent", "Premium"]
    assert strict["migration"] is None
    loose = build_llm_payload(kpi, migration, min_group_size=1)
    # Mass has no clients in 2025-04, so it stays suppressed even at size 1.
    assert [s["segment"] for s in loose["segments"]] == ["ALL", "Affluent", "Premium"]
    assert loose["suppressed_segments"] == ["Mass"]


@pytest.mark.parametrize("small_month", ["2025-06", "2025-05"])
def test_segment_small_in_either_month_is_suppressed(small_run, small_month):
    kpi = small_run["kpi"].copy()
    kpi.loc[(kpi["month"] == small_month) & (kpi["segment"] == "Mass"), "clients"] = 29
    payload = build_llm_payload(kpi, small_run["migration"], min_group_size=30)
    assert "Mass" in payload["suppressed_segments"]
    assert "Mass" not in [s["segment"] for s in payload["segments"]]


def test_first_month_has_no_previous(small_run):
    with pytest.raises(ValueError):
        build_llm_payload(small_run["kpi"], small_run["migration"], month="2025-01")


def test_delta_and_json_values():
    assert kpi_delta({f: 1.5 for f in LLM_KPI_FIELDS}, {f: 1.0 for f in LLM_KPI_FIELDS}) == {
        f: 0.5 for f in LLM_KPI_FIELDS
    }
    assert kpi_delta({"churn_rate": None}, {"churn_rate": 0.1})["churn_rate"] is None
    assert to_json_value(float("nan")) is None


def test_prompt_template_and_render(payload):
    assert "$payload_json" in load_prompt_template().template
    prompt = render_prompt(payload)
    assert "СИНТЕТИЧЕСКИЕ ДАННЫЕ" in prompt
    assert '"data_origin": "SYNTHETIC"' in prompt
    assert "$payload_json" not in prompt


def test_pipeline_llm_files_have_no_client_data(small_run):
    for name in ("llm_payload.json", "llm_prompt.txt"):
        text = (small_run["out_dir"] / "llm" / name).read_text(encoding="utf-8")
        assert not re.search(r"SYN-\d", text), name
        assert config.NAME_PREFIX.strip() not in text, name
