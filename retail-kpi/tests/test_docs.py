"""Документация: README описывает каждую функцию, memo нельзя принять за реальную аналитику."""

import inspect
import re
from pathlib import Path

import pytest

from retail_kpi import generate, kpi, llm_input, mart, pipeline, segment

MODULES = [generate, segment, mart, kpi, llm_input, pipeline]
PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR.parent
README = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
MEMO = (REPO_DIR / "usecase-final.md").read_text(encoding="utf-8")


def module_functions():
    for module in MODULES:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if func.__module__ == module.__name__:
                yield f"{module.__name__.split('.')[-1]}.{name}"


@pytest.mark.parametrize("qualified", sorted(module_functions()))
def test_readme_documents_every_function(qualified):
    name = qualified.split(".")[-1]
    assert re.search(rf"`{name}\(", README), f"README does not document {qualified}"


def test_readme_has_reproducible_run_commands():
    for command in ("pip install -r requirements.txt", "python -m retail_kpi --out build", "pytest"):
        assert command in README
    assert "СИНТЕТИЧЕСКИЕ ДАННЫЕ" in README.split("\n## ")[0]


def test_memo_is_marked_synthetic_at_top_and_bottom():
    lines = [line for line in MEMO.splitlines() if line.strip()]
    assert "СИНТЕТИЧЕСКИЕ ДАННЫЕ" in "\n".join(lines[:3])
    assert "СИНТЕТИЧЕСКИЕ ДАННЫЕ" in "\n".join(lines[-3:])
    assert "НЕ АНАЛИТИКА РЕАЛЬНОГО БАНКА" in MEMO


@pytest.mark.parametrize(
    "heading", ["Пользователь", "Метрика", "Граница автоматизации", "Риски"]
)
def test_memo_has_required_sections(heading):
    assert re.search(rf"^## \d+\. {heading}", MEMO, re.MULTILINE)


def test_memo_contains_no_client_identifiers():
    assert not re.search(r"SYN-\d", MEMO)
    assert not re.search(r"\[SYN\] [A-ZА-Я]", MEMO)
