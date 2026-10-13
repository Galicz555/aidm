"""Phase 7 gate: does the local LLM call vault tools reliably?

The single highest-risk question for private mode is whether the
small-quant Gemma model still emits well-formed tool calls when the
persona asks for them. We answer it before wiring anything into the
voice loop: this script talks straight to a port-forwarded llama-server
at ``http://localhost:8002/v1``, hands it the *real* persona prompt
and the *real* vault-tool schema, and counts how often it picks the
right tool with sensible arguments.

Run from the repo root::

    docker compose --profile private up -d llm-gateway       # or: services/llm-gateway/entrypoint.sh
    make bench-tools                                  # English set (default)
    make bench-tools BENCH_ARGS="--lang hu"           # historical hu/en mix
    PROFILE=qwen38-tuned make bench-tools             # a model profile's DM mode

Fixture sets: ``--lang en`` (``fixtures/tool_calls.en.yaml``) is the
default whenever ``AIDM_LOCALE`` is unset or ``en``; ``--lang hu`` runs
the original ``fixtures/tool_calls.yaml``. With ``AIDM_DM_THINKING`` on
(a tuned profile) each request mirrors the DM call's thinking mode;
``AIDM_LLM_TEMPERATURE`` overrides the default temperature.

Pass thresholds live in :func:`evaluate` — currently ≥85% tool selection
correct, ≥75% argument predicate correct, both required. The script
exits 0 on pass and non-zero on fail; PLAN.md gates Phase 7 ship on
that exit code.

This is a batch job, not a reactive loop.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

# Make the orchestrator package importable when running the script
# directly via ``uv run python scripts/bench_tool_calls.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "services" / "orchestrator"))

from aidm.config import _parse_bool  # noqa: E402
from aidm.persona import persona_for  # noqa: E402
from aidm.vault_tools import _build_tools_schema  # noqa: E402

FIXTURES = {
    "en": Path(__file__).parent / "fixtures" / "tool_calls.en.yaml",
    "hu": Path(__file__).parent / "fixtures" / "tool_calls.yaml",
}
DEFAULT_BASE_URL = "http://localhost:8002/v1"
DEFAULT_MODEL = "gemma-3-27b-it"
DEFAULT_RUNS_PER_CASE = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT_S = 60.0
# The tool decision lands in the first few tokens; a thinking model spends
# its reasoning budget (tuned profiles: 1024) before that.
MAX_TOKENS = 128
THINKING_MAX_TOKENS = 1536
# A case's runs are gathered concurrently against a PARALLEL=1 gateway, so
# the last one queues behind the others' full reasoning.
THINKING_TIMEOUT_S = 600.0

TOOL_PASS_THRESHOLD = 0.85
ARGS_PASS_THRESHOLD = 0.75


@dataclass(frozen=True)
class Fixture:
    name: str
    locale: str
    user: str
    expected_tool: str
    arg_field: str | None
    arg_regex: str | None


@dataclass
class CaseResult:
    fixture: Fixture
    tool_correct: int
    args_correct: int
    runs: int
    samples: list[dict[str, Any]]

    @property
    def tool_rate(self) -> float:
        return self.tool_correct / self.runs if self.runs else 0.0

    @property
    def args_rate(self) -> float:
        return self.args_correct / self.runs if self.runs else 0.0


async def main() -> int:
    settings = parse_settings()
    fixtures = load_fixtures(FIXTURES[settings.lang])
    schema = openai_tools_from_vault_schema()
    client = AsyncOpenAI(api_key="local-llama-cpp", base_url=settings.base_url)

    results = await run_all_cases(client, schema, fixtures, settings)
    print_report(results, settings)
    return 0 if passed(results) else 1


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    runs_per_case: int
    temperature: float
    lang: str
    thinking: bool


def parse_settings(argv: list[str] | None = None) -> Settings:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Vault-tool reliability bench against local LLM.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_CASE)
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("AIDM_LLM_TEMPERATURE") or DEFAULT_TEMPERATURE),
    )
    parser.add_argument(
        "--lang",
        choices=sorted(FIXTURES),
        default="hu" if os.environ.get("AIDM_LOCALE") == "hu" else "en",
        help="fixture set (default: en unless AIDM_LOCALE=hu)",
    )
    args = parser.parse_args(argv)
    return Settings(
        base_url=args.base_url,
        model=args.model,
        runs_per_case=args.runs,
        temperature=args.temperature,
        lang=args.lang,
        thinking=_parse_bool(os.environ.get("AIDM_DM_THINKING"), default=False),
    )


def load_fixtures(path: Path) -> list[Fixture]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixtures: list[Fixture] = []
    for raw in data["cases"]:
        fixtures.append(
            Fixture(
                name=raw["name"],
                locale=raw.get("locale", "hu"),
                user=raw["user"],
                expected_tool=raw["expected_tool"],
                arg_field=raw.get("arg_field"),
                arg_regex=raw.get("arg_regex"),
            )
        )
    return fixtures


def openai_tools_from_vault_schema() -> list[dict[str, Any]]:
    """Render the real vault ToolsSchema into OpenAI ``tools`` wire format.

    We deliberately reuse ``_build_tools_schema()`` rather than
    duplicating the definitions — the bench then drifts in lockstep
    with the orchestrator. Each FunctionSchema becomes one
    ``{"type": "function", "function": {...}}`` tool entry.
    """
    schema = _build_tools_schema()
    tools: list[dict[str, Any]] = []
    for fn in schema.standard_tools:
        body = fn.to_default_dict()
        tools.append({"type": "function", "function": body})
    return tools


async def run_all_cases(
    client: AsyncOpenAI,
    tools: list[dict[str, Any]],
    fixtures: list[Fixture],
    settings: Settings,
) -> list[CaseResult]:
    return [await run_case(client, tools, fixture, settings) for fixture in fixtures]


async def run_case(
    client: AsyncOpenAI,
    tools: list[dict[str, Any]],
    fixture: Fixture,
    settings: Settings,
) -> CaseResult:
    samples = await asyncio.gather(
        *(invoke_once(client, tools, fixture, settings) for _ in range(settings.runs_per_case))
    )
    tool_correct = sum(1 for s in samples if s["tool_correct"])
    args_correct = sum(1 for s in samples if s["args_correct"])
    return CaseResult(
        fixture=fixture,
        tool_correct=tool_correct,
        args_correct=args_correct,
        runs=settings.runs_per_case,
        samples=samples,
    )


async def invoke_once(
    client: AsyncOpenAI,
    tools: list[dict[str, Any]],
    fixture: Fixture,
    settings: Settings,
) -> dict[str, Any]:
    persona = persona_for(fixture.locale, emotion_tags=False, mode="private")
    try:
        response = await client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": fixture.user},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=settings.temperature,
            # Bench only needs to know whether a tool call was emitted —
            # that decision lands in the first few tokens. Capping
            # avoids paying for hundreds of prose tokens we discard.
            max_tokens=THINKING_MAX_TOKENS if settings.thinking else MAX_TOKENS,
            # Same thinking mode as the DM request (llm_local).
            extra_body={"chat_template_kwargs": {"enable_thinking": settings.thinking}},
            timeout=THINKING_TIMEOUT_S if settings.thinking else DEFAULT_TIMEOUT_S,
        )
    except Exception as exc:
        return {
            "tool_called": None,
            "tool_correct": False,
            "args_correct": False,
            "error": str(exc),
        }

    message = response.choices[0].message
    calls = message.tool_calls or []
    if not calls:
        return {
            "tool_called": None,
            "args": None,
            "tool_correct": False,
            "args_correct": False,
            "raw_content": message.content,
        }

    first = calls[0]
    name = first.function.name
    args = parse_arguments(first.function.arguments)
    tool_correct = name == fixture.expected_tool
    args_correct = tool_correct and predicate_passes(args, fixture)
    return {
        "tool_called": name,
        "args": args,
        "tool_correct": tool_correct,
        "args_correct": args_correct,
    }


def parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def predicate_passes(args: dict[str, Any], fixture: Fixture) -> bool:
    """Argument predicate: does the model's argument satisfy the case?

    Fixtures specify a single ``arg_field`` whose value is matched
    against ``arg_regex`` (case-insensitive). Cases that only care
    about tool selection (e.g. ``vault_list_dir``) omit both fields
    and pass automatically.
    """
    if fixture.arg_field is None or fixture.arg_regex is None:
        return True
    value = args.get(fixture.arg_field)
    if not isinstance(value, str):
        return False
    return bool(re.search(fixture.arg_regex, value, re.IGNORECASE))


def evaluate(results: list[CaseResult]) -> tuple[float, float]:
    total_runs = sum(r.runs for r in results)
    if total_runs == 0:
        return 0.0, 0.0
    tool_rate = sum(r.tool_correct for r in results) / total_runs
    args_rate = sum(r.args_correct for r in results) / total_runs
    return tool_rate, args_rate


def passed(results: list[CaseResult]) -> bool:
    tool_rate, args_rate = evaluate(results)
    return tool_rate >= TOOL_PASS_THRESHOLD and args_rate >= ARGS_PASS_THRESHOLD


def print_report(results: list[CaseResult], settings: Settings) -> None:
    print(
        f"\nBench: model={settings.model}  runs/case={settings.runs_per_case}  "
        f"temp={settings.temperature}  lang={settings.lang}  thinking={settings.thinking}\n"
    )
    print(f"{'case':<40} {'expected':<28} {'tool%':>7} {'args%':>7}")
    print("-" * 84)
    for result in results:
        print(
            f"{result.fixture.name:<40} {result.fixture.expected_tool:<28} "
            f"{result.tool_rate * 100:>6.1f}% {result.args_rate * 100:>6.1f}%"
        )
    tool_rate, args_rate = evaluate(results)
    print("-" * 84)
    print(
        f"OVERALL  tool={tool_rate * 100:.1f}%  args={args_rate * 100:.1f}%  "
        f"(thresholds tool≥{TOOL_PASS_THRESHOLD * 100:.0f}% args≥{ARGS_PASS_THRESHOLD * 100:.0f}%)"
    )
    print(f"\nFailures by case:")
    any_failure = False
    for result in results:
        if result.tool_rate < TOOL_PASS_THRESHOLD or result.args_rate < ARGS_PASS_THRESHOLD:
            any_failure = True
            for i, sample in enumerate(result.samples):
                if not sample.get("args_correct"):
                    print(
                        f"  [{result.fixture.name}#{i}] called={sample.get('tool_called')!r} "
                        f"args={sample.get('args')!r} error={sample.get('error')}"
                    )
    if not any_failure:
        print("  (none)")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
