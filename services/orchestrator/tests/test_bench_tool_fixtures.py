"""English tool-call bench fixtures: loader, set selection, faithful translation."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bench_tool_calls.py"


def _bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_tool_calls", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_tool_calls", module)
    spec.loader.exec_module(module)
    return module


def _raw(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    return cases


def test_english_set_loads_as_english_fixtures() -> None:
    bench = _bench()
    fixtures = bench.load_fixtures(bench.FIXTURES["en"])
    assert len(fixtures) == 21
    assert {f.locale for f in fixtures} == {"en"}
    assert len({f.name for f in fixtures}) == len(fixtures)
    for f in fixtures:
        if f.arg_regex:
            re.compile(f.arg_regex)


def test_every_original_case_is_translated_once_with_the_same_tool_and_arg() -> None:
    bench = _bench()
    original = {c["name"]: c for c in _raw(bench.FIXTURES["hu"])}
    covered: list[str] = []
    for case in _raw(bench.FIXTURES["en"]):
        sources = [s.strip() for s in case["from"].split(",")]
        covered += sources
        for source in sources:
            assert case["expected_tool"] == original[source]["expected_tool"], case["name"]
            assert case.get("arg_field") == original[source].get("arg_field"), case["name"]
    assert sorted(covered) == sorted(original)


@pytest.mark.parametrize(
    ("case", "english_arg"),
    [
        ("en_lore_wolves", "wolves"),
        ("en_lore_war", "war"),
        ("en_loc_grey_forest", "Grey Forest"),
        ("en_population", "population"),
    ],
)
def test_translated_arg_regex_accepts_the_english_term(case: str, english_arg: str) -> None:
    bench = _bench()
    fixture = next(f for f in bench.load_fixtures(bench.FIXTURES["en"]) if f.name == case)
    assert bench.predicate_passes({fixture.arg_field: english_arg}, fixture)


@pytest.mark.parametrize(
    ("locale", "argv", "lang"),
    [(None, [], "en"), ("en", [], "en"), ("hu", [], "hu"), ("hu", ["--lang", "en"], "en")],
)
def test_english_set_is_the_default_unless_the_locale_is_hungarian(
    monkeypatch: pytest.MonkeyPatch, locale: str | None, argv: list[str], lang: str
) -> None:
    bench = _bench()
    monkeypatch.delenv("AIDM_DM_THINKING", raising=False)
    if locale is None:
        monkeypatch.delenv("AIDM_LOCALE", raising=False)
    else:
        monkeypatch.setenv("AIDM_LOCALE", locale)
    settings = bench.parse_settings(argv)
    assert settings.lang == lang
    assert settings.thinking is False


def test_bench_mirrors_a_thinking_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench()
    monkeypatch.setenv("AIDM_DM_THINKING", "1")
    monkeypatch.setenv("AIDM_LLM_TEMPERATURE", "1.0")
    settings = bench.parse_settings([])
    assert settings.thinking is True
    assert settings.temperature == 1.0
