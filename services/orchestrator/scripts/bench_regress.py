"""Per-fix regression bench runner.

Drives ``eval_llm.py`` N times against each scenario under
``eval_transcripts/regress/`` and scores the captured ``history.json``
against the assertions in the scenario's companion ``*.expect.yaml``.

Output layout::

    docs/bench/<date>/regress/
        summary.md
        summary.json
        <scenario-id>/
            replay-001.history.json
            ...
            replay-010.history.json
            scenario.md         # per-scenario detail

Two assertion kinds:

- ``deterministic`` — every replay must pass. One failure ⇒ scenario fail.
- ``behavioral`` — pass rate must clear ``scenario.pass_threshold``
  (default 0.7). Used when the assertion depends on the LLM's choice.

Exit code: 0 if every scenario passes its threshold, 1 otherwise. Stdout
shows a tight per-scenario summary; the markdown report carries the
full bucket-level breakdown.

Usage::

    uv run python services/orchestrator/scripts/bench_regress.py
    uv run python services/orchestrator/scripts/bench_regress.py --replays 3   # smoke
    uv run python services/orchestrator/scripts/bench_regress.py --scenarios 01-srd-skip 04-self-grep-prohibit

The harness shells out to ``eval_llm.py`` per replay rather than calling
it as a library — keeps eval_llm.py untouched and lets each replay reset
its own sandbox vault cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
_REGRESS_DIR = _THIS.parent / "eval_transcripts" / "regress"
_EVAL_LLM = _THIS.parent / "eval_llm.py"

# Where bench_regress drops its outputs. Mirrors docs/bench/<date>/<batch>/.
_DEFAULT_OUT_ROOT = _REPO_ROOT / "docs" / "bench"


# --- assertion engine ---------------------------------------------------


@dataclass
class AssertionResult:
    assertion_id: str
    passed: bool
    detail: str = ""


@dataclass
class ReplayResult:
    replay_index: int
    history_path: Path
    assertion_results: list[AssertionResult] = field(default_factory=list)
    crashed: bool = False
    crash_detail: str = ""


@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_path: Path
    replays: list[ReplayResult] = field(default_factory=list)
    overall_passed: bool = False
    per_assertion_pass_rate: dict[str, float] = field(default_factory=dict)


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _iter_assistant_tool_calls(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten history into a turn-stamped list of tool calls.

    Each entry: {turn_index, name, raw_args, parsed_args, result}. ``turn_index``
    is a 1-based count of user messages preceding this assistant turn.
    """
    out: list[dict[str, Any]] = []
    turn_index = 0
    # Map tool_call_id → name+args for matching the tool result message.
    pending: dict[str, dict[str, Any]] = {}
    for msg in history:
        role = msg.get("role")
        if role == "user":
            turn_index += 1
            continue
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "")
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed = None
                entry = {
                    "turn_index": turn_index,
                    "name": name,
                    "raw_args": raw_args,
                    "parsed_args": parsed,
                    "result": "",
                    "tool_call_id": tc.get("id"),
                }
                pending[tc.get("id", "")] = entry
                out.append(entry)
            continue
        if role == "tool":
            tc_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            if tc_id in pending:
                # JSON-wrapped string result: unwrap if possible for cleaner matches.
                pending[tc_id]["result"] = _unwrap_tool_result(content)
    return out


def _unwrap_tool_result(raw: Any) -> str:
    if isinstance(raw, str):
        # Tool results are JSON-encoded strings in the OpenAI tool channel.
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, str):
                return decoded
        except json.JSONDecodeError:
            pass
        return raw
    if isinstance(raw, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in raw
        )
    return str(raw)


def _iter_assistant_texts(history: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Turn-stamped list of assistant text (excluding tool-only turns)."""
    out: list[tuple[int, str]] = []
    turn_index = 0
    for msg in history:
        if msg.get("role") == "user":
            turn_index += 1
            continue
        if msg.get("role") == "assistant":
            text = msg.get("content") or ""
            if isinstance(text, list):
                text = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item) for item in text
                )
            if text and text.strip():
                out.append((turn_index, text))
    return out


# --- assertion rules ---------------------------------------------------
#
# Each rule is a function: (history, assertion_yaml) -> AssertionResult.
# The dispatcher in ``check_assertion`` picks one by ``assertion["rule"]``.


def _rule_no_grep_result_path_prefix(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    prefix = str(assertion.get("prefix", ""))
    if not prefix:
        return False, "missing 'prefix' in assertion"
    grep_calls = [tc for tc in _iter_assistant_tool_calls(history) if tc["name"] == "vault_grep"]
    if not grep_calls:
        # No grep calls means the rule is vacuously satisfied for this replay.
        return True, "no vault_grep calls fired"
    offending: list[str] = []
    for tc in grep_calls:
        if prefix in tc["result"]:
            offending.append(f"turn {tc['turn_index']}: result contains {prefix!r}")
    if offending:
        return False, "; ".join(offending)
    return True, f"{len(grep_calls)} grep call(s), none touched {prefix!r}"


def _rule_tool_paired_within_n_turns(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    leading = assertion["leading_tool"]
    following = assertion["following_tool"]
    max_gap = int(assertion.get("max_turn_gap", 2))
    slug_match = bool(assertion.get("slug_must_match", True))
    calls = _iter_assistant_tool_calls(history)
    leads = [tc for tc in calls if tc["name"] == leading]
    if not leads:
        return True, f"no {leading} calls — assertion vacuous"
    misses: list[str] = []
    for lead in leads:
        lead_slug = (lead.get("parsed_args") or {}).get("slug")
        matched = False
        for follow in calls:
            if follow["name"] != following:
                continue
            if follow["turn_index"] < lead["turn_index"]:
                continue
            if follow["turn_index"] > lead["turn_index"] + max_gap:
                continue
            if slug_match:
                fol_slug = (follow.get("parsed_args") or {}).get("slug")
                if fol_slug != lead_slug:
                    continue
            matched = True
            break
        if not matched:
            misses.append(
                f"{leading}(slug={lead_slug!r}) at turn {lead['turn_index']} "
                f"not followed by {following} within {max_gap} turns"
            )
    if misses:
        return False, "; ".join(misses)
    return True, f"all {len(leads)} {leading} call(s) paired"


def _rule_npc_name_appears_in_narration(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    after_tool = assertion["after_tool"]
    within = int(assertion.get("within_turns", 2))
    calls = _iter_assistant_tool_calls(history)
    triggers = [tc for tc in calls if tc["name"] == after_tool]
    if not triggers:
        return True, f"no {after_tool} calls — assertion vacuous"
    texts = _iter_assistant_texts(history)
    misses: list[str] = []
    for tc in triggers:
        # Try to extract a name from the tool args. Prefer explicit `name` in
        # frontmatter content; fall back to slug.
        args = tc.get("parsed_args") or {}
        name = ""
        content = args.get("content", "")
        if isinstance(content, str):
            m = re.search(r"name:\s*([^\n]+)", content)
            if m:
                name = m.group(1).strip()
        if not name:
            name = str(args.get("slug", ""))
        if not name:
            continue
        # Match either the full frontmatter name OR its first token. The
        # model routinely writes a full title in the sheet (e.g. "Eli a
        # Bárkez", "Sir Aldric of the Veil") then voices the NPC by their
        # given name only ("Eli", "Aldric"). Either form proves the named
        # NPC actually spoke — which is what this assertion measures.
        window = [t for ti, t in texts if tc["turn_index"] <= ti <= tc["turn_index"] + within]
        candidates = {name}
        first_token = name.split()[0] if name.split() else ""
        if first_token:
            candidates.add(first_token)
        if not any(c.lower() in t.lower() for c in candidates for t in window):
            misses.append(
                f"name={name!r} (and first token {first_token!r}) "
                f"not in narration within {within} turns"
            )
    if misses:
        return False, "; ".join(misses)
    return True, "all NPC names appeared in narration"


def _rule_tool_not_called_with_arg(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    tool = assertion["tool"]
    arg_name = assertion["arg_name"]
    needle = assertion.get("arg_value_matches")
    pattern = assertion.get("arg_value_matches_regex")
    offending: list[str] = []
    for tc in _iter_assistant_tool_calls(history):
        if tc["name"] != tool:
            continue
        args = tc.get("parsed_args") or {}
        val = str(args.get(arg_name, ""))
        hit = False
        if needle is not None and str(needle).lower() in val.lower():
            hit = True
        if pattern is not None and re.search(str(pattern), val):
            hit = True
        if hit:
            offending.append(f"turn {tc['turn_index']}: {tool}({arg_name}={val!r})")
    if offending:
        return False, "; ".join(offending)
    return True, f"no {tool} call matched {arg_name}={needle or pattern!r}"


def _rule_first_assistant_mentions_any(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    turn = int(assertion.get("turn", 1))
    needles = [str(n) for n in assertion.get("needles", [])]
    texts = _iter_assistant_texts(history)
    target = [t for ti, t in texts if ti == turn]
    if not target:
        return False, f"no assistant text at turn {turn}"
    body = " ".join(target).lower()
    hits = [n for n in needles if n.lower() in body]
    if hits:
        return True, f"mentioned: {hits}"
    return False, f"none of {needles} appeared at turn {turn}"


def _rule_no_grep_pattern_in_prior_assistant(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    min_words = int(assertion.get("min_pattern_words", 2))
    texts_by_turn = _iter_assistant_texts(history)
    offending: list[str] = []
    for tc in _iter_assistant_tool_calls(history):
        if tc["name"] != "vault_grep":
            continue
        pattern = str((tc.get("parsed_args") or {}).get("pattern", "")).strip()
        if not pattern:
            continue
        # Require ≥ min_words tokens; single-word greps are legitimate
        # (player named "Lyra" → grep "Lyra"). The failure mode is multi-word
        # noun phrases the DM itself wrote.
        if len(pattern.split()) < min_words:
            continue
        prior_corpus = " ".join(t for ti, t in texts_by_turn if ti < tc["turn_index"]).lower()
        if pattern.lower() in prior_corpus:
            offending.append(
                f"turn {tc['turn_index']}: grep pattern {pattern!r} "
                "is substring of prior assistant text"
            )
    if offending:
        return False, "; ".join(offending)
    return True, "no self-grep detected"


def _rule_max_tool_calls(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    tool = assertion["tool"]
    max_n = int(assertion["max_calls"])
    n = sum(1 for tc in _iter_assistant_tool_calls(history) if tc["name"] == tool)
    if n <= max_n:
        return True, f"{n} {tool} call(s) ≤ cap {max_n}"
    return False, f"{n} {tool} call(s) exceeds cap {max_n}"


def _rule_min_tool_calls(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    tool = assertion["tool"]
    min_n = int(assertion["min_calls"])
    n = sum(1 for tc in _iter_assistant_tool_calls(history) if tc["name"] == tool)
    if n >= min_n:
        return True, f"{n} {tool} call(s) ≥ floor {min_n}"
    return False, f"{n} {tool} call(s) below floor {min_n}"


def _rule_tool_called_with_arg_min_length(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    """At least one call to ``tool`` has ``arg_name`` ≥ ``min_length`` chars.

    Used to pin "the session log content must be at least a paragraph"
    type checks, where JSON truncation or stub-only outputs would emit
    a degenerate short value.
    """
    tool = assertion["tool"]
    arg_name = assertion["arg_name"]
    min_length = int(assertion["min_length"])
    qualifying = [
        tc
        for tc in _iter_assistant_tool_calls(history)
        if tc["name"] == tool
        and len(str((tc.get("parsed_args") or {}).get(arg_name, ""))) >= min_length
    ]
    if qualifying:
        return True, f"{len(qualifying)} {tool} call(s) with {arg_name}≥{min_length} chars"
    return False, f"no {tool} call had {arg_name}≥{min_length} chars"


def _rule_tool_called_with_arg_substring(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    """At least one call to ``tool`` has ``arg_name`` containing ``needle``
    (case-insensitive substring).
    """
    tool = assertion["tool"]
    arg_name = assertion["arg_name"]
    needle = str(assertion["needle"]).lower()
    for tc in _iter_assistant_tool_calls(history):
        if tc["name"] != tool:
            continue
        val = str((tc.get("parsed_args") or {}).get(arg_name, "")).lower()
        if needle in val:
            return True, f"{tool} call's {arg_name} contained {needle!r}"
    return False, f"no {tool} call had {arg_name} containing {needle!r}"


def _rule_no_assistant_text_matches(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    """No assistant turn's text matches the given regex (case-insensitive).

    Used to forbid the DM from emitting forbidden patterns in narration —
    e.g. ``\\[ROLL d`` (player-only dice markers) or assistant-voice tells.
    """
    pattern_src = assertion.get("pattern")
    if not pattern_src:
        return False, "missing 'pattern' in assertion"
    try:
        pattern = re.compile(str(pattern_src), re.IGNORECASE)
    except re.error as exc:
        return False, f"invalid regex: {exc}"
    offending: list[str] = []
    for turn_idx, text in _iter_assistant_texts(history):
        m = pattern.search(text)
        if m:
            offending.append(f"turn {turn_idx}: matched {m.group(0)!r}")
    if offending:
        return False, "; ".join(offending)
    return True, f"no assistant text matched {pattern_src!r}"


def _rule_assistant_text_matches_min_turns(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    """At least ``min_turns`` assistant turns match the given regex.

    Inverse of ``no_assistant_text_matches``. Used for "the DM SHOULD emit
    this on most turns" assertions — e.g. inline ``[state: <slug> ...]``
    directives that drift away without enforcement.
    """
    pattern_src = assertion.get("pattern")
    if not pattern_src:
        return False, "missing 'pattern' in assertion"
    try:
        pattern = re.compile(str(pattern_src), re.IGNORECASE)
    except re.error as exc:
        return False, f"invalid regex: {exc}"
    min_turns = int(assertion.get("min_turns", 1))
    matched: list[int] = []
    total = 0
    for turn_idx, text in _iter_assistant_texts(history):
        total += 1
        if pattern.search(text):
            matched.append(turn_idx)
    if len(matched) >= min_turns:
        return True, f"{len(matched)}/{total} assistant turns matched"
    return (
        False,
        f"only {len(matched)}/{total} assistant turns matched {pattern_src!r} (need {min_turns})",
    )


_STATE_DIRECTIVE_RE = re.compile(
    r"\[state:\s*([a-z0-9_-]+)\s+([^\]]+)\]",
    re.IGNORECASE,
)


def _parse_state_kv(body: str) -> dict[str, str]:
    """Pull m=/g=/k= (or long-form mood=/goal=/mask=) values out of a
    directive body. Returns lowercased canonical key → trimmed value.
    Tolerant of both ``;`` and ``,`` separators (matches the parser).
    """
    out: dict[str, str] = {}
    for raw in re.split(r"[;,]", body):
        piece = raw.strip()
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        canonical = {
            "m": "mood", "mood": "mood",
            "g": "goal", "goal": "goal", "scene_goal": "goal",
            "k": "mask", "mask": "mask", "social_mask": "mask",
        }.get(k.strip().lower())
        if canonical:
            out[canonical] = v.strip()
    return out


def _rule_state_directive_has_all_keys(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    """For the named slug, every emitted ``[state: <slug> ...]`` directive
    must contain all three keys (mood, goal, mask). Catches the degenerate
    case where the model emits ``[state: lyra m=guarded]`` and the other
    two are silently dropped — the registry then keeps stale values.

    Behavioral: ``min_turns`` of N directives meeting the all-keys bar.
    """
    slug = str(assertion.get("slug", "")).lower()
    if not slug:
        return False, "missing 'slug' in assertion"
    min_turns = int(assertion.get("min_turns", 1))
    full = []
    partial = []
    for turn_idx, text in _iter_assistant_texts(history):
        for m in _STATE_DIRECTIVE_RE.finditer(text):
            if m.group(1).lower() != slug:
                continue
            kv = _parse_state_kv(m.group(2))
            if {"mood", "goal", "mask"}.issubset(kv):
                full.append(turn_idx)
            else:
                partial.append((turn_idx, sorted(kv)))
    if len(full) >= min_turns:
        return True, f"{len(full)} directive(s) with all 3 keys"
    return (
        False,
        f"only {len(full)} all-keys directive(s); partial={partial[:3]}",
    )


def _rule_state_directive_values_evolve(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    """For the named slug, the values of at least ``key`` (default: any of
    mood/goal/mask) must take ``min_distinct`` distinct values across the
    session. Catches the live-mode failure where the model emits the same
    directive 3+ turns running — registry technically alive, scene
    technically progressing, but mood/goal/mask frozen and the NPC reads
    as a stuck record.
    """
    slug = str(assertion.get("slug", "")).lower()
    if not slug:
        return False, "missing 'slug' in assertion"
    key = str(assertion.get("key", "mood")).lower()
    if key not in ("mood", "goal", "mask"):
        return False, f"invalid key {key!r}; must be mood|goal|mask"
    min_distinct = int(assertion.get("min_distinct", 3))
    seen: list[str] = []
    for _, text in _iter_assistant_texts(history):
        for m in _STATE_DIRECTIVE_RE.finditer(text):
            if m.group(1).lower() != slug:
                continue
            kv = _parse_state_kv(m.group(2))
            v = kv.get(key)
            if v:
                seen.append(v.lower())
    distinct = set(seen)
    if len(distinct) >= min_distinct:
        return True, f"{len(distinct)} distinct {key} values: {sorted(distinct)[:5]}"
    return (
        False,
        f"only {len(distinct)} distinct {key} value(s) "
        f"across {len(seen)} emission(s): {sorted(distinct)}",
    )


def _rule_any_of_tools_called(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    """At least one call to ANY of the named tools in the full history.

    Used when multiple tool paths satisfy the same functional goal — e.g.
    "the model must fetch NPC canon" can be satisfied by either
    ``vault_grep`` (which auto-inlines NPC sheets) or ``vault_read_file``.
    """
    tools = assertion.get("tools") or []
    if not isinstance(tools, list) or not tools:
        return False, "missing 'tools' list in assertion"
    wanted = {str(t) for t in tools}
    seen = [tc["name"] for tc in _iter_assistant_tool_calls(history) if tc["name"] in wanted]
    if seen:
        return True, f"called {seen[0]} (any of {sorted(wanted)})"
    return False, f"none of {sorted(wanted)} were called"


def _rule_tool_called_in_turn_range(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    tool = assertion["tool"]
    from_turn = int(assertion.get("from_turn", 1))
    to_turn = int(assertion.get("to_turn", 9999))
    must_have = int(assertion.get("must_have_calls", 1))
    matching = [
        tc
        for tc in _iter_assistant_tool_calls(history)
        if tc["name"] == tool and from_turn <= tc["turn_index"] <= to_turn
    ]
    if len(matching) >= must_have:
        return True, f"{len(matching)} {tool} call(s) in T{from_turn}-T{to_turn}"
    return (
        False,
        f"only {len(matching)} {tool} call(s) in T{from_turn}-T{to_turn} (need {must_have})",
    )


def _rule_tool_called_after_turn(
    history: list[dict[str, Any]], assertion: dict[str, Any]
) -> tuple[bool, str]:
    tool = assertion["tool"]
    after = int(assertion["after_turn"])
    must_have = int(assertion.get("must_have_calls", 1))
    matching = [
        tc
        for tc in _iter_assistant_tool_calls(history)
        if tc["name"] == tool and tc["turn_index"] > after
    ]
    if len(matching) >= must_have:
        return True, f"{len(matching)} {tool} call(s) after turn {after}"
    return False, f"only {len(matching)} {tool} call(s) after turn {after} (need {must_have})"


_RULES: dict[str, Any] = {
    "no_grep_result_path_prefix": _rule_no_grep_result_path_prefix,
    "tool_paired_within_n_turns": _rule_tool_paired_within_n_turns,
    "npc_name_appears_in_narration": _rule_npc_name_appears_in_narration,
    "tool_not_called_with_arg": _rule_tool_not_called_with_arg,
    "first_assistant_mentions_any": _rule_first_assistant_mentions_any,
    "no_grep_pattern_in_prior_assistant": _rule_no_grep_pattern_in_prior_assistant,
    "max_tool_calls": _rule_max_tool_calls,
    "min_tool_calls": _rule_min_tool_calls,
    "tool_called_after_turn": _rule_tool_called_after_turn,
    "tool_called_in_turn_range": _rule_tool_called_in_turn_range,
    "no_assistant_text_matches": _rule_no_assistant_text_matches,
    "assistant_text_matches_min_turns": _rule_assistant_text_matches_min_turns,
    "any_of_tools_called": _rule_any_of_tools_called,
    "state_directive_has_all_keys": _rule_state_directive_has_all_keys,
    "state_directive_values_evolve": _rule_state_directive_values_evolve,
    "tool_called_with_arg_min_length": _rule_tool_called_with_arg_min_length,
    "tool_called_with_arg_substring": _rule_tool_called_with_arg_substring,
}


def check_assertion(history: list[dict[str, Any]], assertion: dict[str, Any]) -> AssertionResult:
    aid = str(assertion.get("id", "unnamed"))
    rule_name = str(assertion.get("rule", ""))
    rule = _RULES.get(rule_name)
    if rule is None:
        return AssertionResult(assertion_id=aid, passed=False, detail=f"unknown rule {rule_name!r}")
    try:
        passed, detail = rule(history, assertion)
    except Exception as exc:
        return AssertionResult(
            assertion_id=aid, passed=False, detail=f"rule crashed: {type(exc).__name__}: {exc}"
        )
    return AssertionResult(assertion_id=aid, passed=passed, detail=detail)


# --- replay driver ------------------------------------------------------


async def _run_one_replay(  # noqa: PLR0913 — replay context is naturally wide
    *,
    transcript: Path,
    scenario_id: str,
    replay_index: int,
    out_dir: Path,
    runner_cfg: dict[str, Any],
    inject_dir: Path | None,
    inject_into: str,
) -> ReplayResult:
    report_path = out_dir / f"replay-{replay_index:03d}.md"
    history_path = report_path.with_suffix(".history.json")

    if inject_dir is not None and inject_dir.is_dir():
        # eval_llm builds its own sandbox at /tmp/aidm-eval-vault-<pid>/. We
        # need the fixtures present BEFORE eval_llm builds the sandbox copy.
        # Cleanest path: drop them into the real vault temporarily for this
        # replay, then remove. Risk: parallel replays would clash; the runner
        # is serial so it's fine.
        _inject_fixtures(inject_dir, inject_into, scenario_id, replay_index)

    cmd = [
        sys.executable,
        str(_EVAL_LLM),
        str(transcript),
        "--report",
        str(report_path),
    ]
    if runner_cfg.get("resume"):
        cmd.extend(["--resume", str(runner_cfg["resume"])])

    # Per-scenario env overrides (e.g. AIDM_NPC_FAILSAFE_ENABLED=false for
    # the no-failsafe baseline scenario). We start from the parent env so
    # the LLM gateway, API keys, etc. are inherited; the scenario's
    # ``runner.env`` block then layers on top.
    subprocess_env = dict(os.environ)
    extra_env = runner_cfg.get("env") or {}
    if isinstance(extra_env, dict):
        for k, v in extra_env.items():
            subprocess_env[str(k)] = str(v)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_REPO_ROOT),
            env=subprocess_env,
        )
        _, stderr = await proc.communicate()
    finally:
        if inject_dir is not None and inject_dir.is_dir():
            _cleanup_fixtures(inject_dir, inject_into)

    if proc.returncode != 0:
        return ReplayResult(
            replay_index=replay_index,
            history_path=history_path,
            crashed=True,
            crash_detail=(
                f"eval_llm exited {proc.returncode}; stderr tail:\n"
                + stderr.decode("utf-8", errors="replace")[-1000:]
            ),
        )
    return ReplayResult(replay_index=replay_index, history_path=history_path)


def _resolve_vault_for_injection() -> Path:
    """Real vault campaign root — used by ``_inject_fixtures``."""
    from aidm.config import load_config  # noqa: PLC0415

    sys.path.insert(0, str(_THIS.parents[1]))  # services/orchestrator
    cfg = load_config()
    return cfg.vault_path  # vault_root / <campaign_slug>


_INJECTED_RECORD: list[Path] = []


def _inject_fixtures(
    fixture_dir: Path, inject_into: str, scenario_id: str, replay_index: int
) -> None:
    """Copy fixture files into the real vault before eval_llm's sandbox copy.

    The runner removes them afterward via ``_cleanup_fixtures``. We track each
    written path so we don't accidentally delete files the user actually owns.
    """
    target_root = _resolve_vault_for_injection() / inject_into
    target_root.mkdir(parents=True, exist_ok=True)
    for src in fixture_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(fixture_dir)
        dest = target_root / rel
        if dest.exists():
            # User already owns this path — don't overwrite.
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        _INJECTED_RECORD.append(dest)


def _cleanup_fixtures(fixture_dir: Path, inject_into: str) -> None:
    """Remove only the files we wrote via ``_inject_fixtures``."""
    while _INJECTED_RECORD:
        path = _INJECTED_RECORD.pop()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
    # Best-effort prune empty dirs we may have created.
    import contextlib  # noqa: PLC0415

    target_root = _resolve_vault_for_injection() / inject_into
    for d in sorted(target_root.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            with contextlib.suppress(OSError):
                d.rmdir()


# --- scenario runner ----------------------------------------------------


async def _run_scenario(  # noqa: PLR0912 — assertion dispatch is naturally branchy
    scenario_path: Path, out_root: Path, replays: int
) -> ScenarioResult:
    with scenario_path.open(encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    scn = spec.get("scenario", {})
    scenario_id = scn.get("id", scenario_path.stem)
    runner_cfg = spec.get("runner", {}) or {}
    assertions = spec.get("assertions", []) or []
    pass_threshold = float(scn.get("pass_threshold", 0.7))
    transcript = scenario_path.with_suffix("").with_suffix(".jsonl")
    if not transcript.is_file():
        # Convention is `<id>.jsonl` alongside `<id>.expect.yaml`. The double
        # `with_suffix` above strips both `.yaml` and `.expect` if needed.
        transcript = scenario_path.parent / f"{scenario_path.name.split('.')[0]}.jsonl"
    inject_dir: Path | None = None
    if runner_cfg.get("inject_dir"):
        inject_dir = scenario_path.parent / str(runner_cfg["inject_dir"])
        # Match fixture dir pattern: <scenario-stem>.fixtures/ alongside the YAML.
        if not inject_dir.is_dir():
            inject_dir = scenario_path.parent / f"{transcript.stem}.fixtures"
    inject_into = str(runner_cfg.get("inject_into", "runs/first-party"))

    scenario_out = out_root / scenario_id
    scenario_out.mkdir(parents=True, exist_ok=True)

    result = ScenarioResult(scenario_id=scenario_id, scenario_path=scenario_path)
    for i in range(1, replays + 1):
        print(f"  [{scenario_id}] replay {i}/{replays}", file=sys.stderr, flush=True)
        replay = await _run_one_replay(
            transcript=transcript,
            scenario_id=scenario_id,
            replay_index=i,
            out_dir=scenario_out,
            runner_cfg=runner_cfg,
            inject_dir=inject_dir,
            inject_into=inject_into,
        )
        if not replay.crashed:
            history = _load_history(replay.history_path)
            for a in assertions:
                replay.assertion_results.append(check_assertion(history, a))
        result.replays.append(replay)

    # Aggregate per-assertion pass rate; gate scenario by kind.
    by_id: dict[str, list[bool]] = {}
    for r in result.replays:
        if r.crashed:
            for a in assertions:
                by_id.setdefault(a["id"], []).append(False)
            continue
        for ar in r.assertion_results:
            by_id.setdefault(ar.assertion_id, []).append(ar.passed)
    overall = True
    for a in assertions:
        outcomes = by_id.get(a["id"], [])
        rate = (sum(outcomes) / len(outcomes)) if outcomes else 0.0
        result.per_assertion_pass_rate[a["id"]] = rate
        if a.get("kind", "behavioral") == "deterministic":
            if not all(outcomes):
                overall = False
        elif rate < pass_threshold:
            overall = False
    result.overall_passed = overall
    return result


# --- reporting ----------------------------------------------------------


def _render_summary(results: list[ScenarioResult], replays: int, out_root: Path) -> str:
    lines: list[str] = []
    lines.append("# Regress bench — DM-quality fixes")
    lines.append("")
    lines.append(f"- replays per scenario: **{replays}**")
    lines.append(f"- scenarios: **{len(results)}**")
    passed = sum(1 for r in results if r.overall_passed)
    lines.append(f"- overall pass: **{passed}/{len(results)}**")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Scenario | Status | Per-assertion pass rate |")
    lines.append("|---|---|---|")
    for r in results:
        status = "✅ pass" if r.overall_passed else "❌ fail"
        rates = ", ".join(
            f"`{aid}`={int(rate * 100)}%" for aid, rate in r.per_assertion_pass_rate.items()
        )
        lines.append(f"| `{r.scenario_id}` | {status} | {rates} |")
    lines.append("")
    lines.append("## Per-scenario detail")
    lines.append("")
    for r in results:
        lines.append(f"### `{r.scenario_id}`")
        lines.append("")
        crashed = sum(1 for x in r.replays if x.crashed)
        if crashed:
            lines.append(f"- crashed replays: **{crashed}/{len(r.replays)}**")
        for aid, rate in r.per_assertion_pass_rate.items():
            lines.append(f"- `{aid}` — pass rate **{int(rate * 100)}%**")
            details: dict[str, int] = {}
            for rep in r.replays:
                for ar in rep.assertion_results:
                    if ar.assertion_id == aid and not ar.passed:
                        details[ar.detail] = details.get(ar.detail, 0) + 1
            for det, count in sorted(details.items(), key=lambda kv: -kv[1]):
                lines.append(f"    - {count}x {det}")
        lines.append("")
    return "\n".join(lines)


def _dump_json(results: list[ScenarioResult], out_path: Path) -> None:
    blob = {
        "scenarios": [
            {
                "id": r.scenario_id,
                "overall_passed": r.overall_passed,
                "per_assertion_pass_rate": r.per_assertion_pass_rate,
                "replays": [
                    {
                        "index": rep.replay_index,
                        "history_path": str(rep.history_path),
                        "crashed": rep.crashed,
                        "assertion_results": [
                            {"id": ar.assertion_id, "passed": ar.passed, "detail": ar.detail}
                            for ar in rep.assertion_results
                        ],
                    }
                    for rep in r.replays
                ],
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")


# --- entry point --------------------------------------------------------


async def _main_async(args: argparse.Namespace) -> int:
    scenario_files = sorted(_REGRESS_DIR.glob("*.expect.yaml"))
    if args.scenarios:
        wanted = set(args.scenarios)
        scenario_files = [
            p
            for p in scenario_files
            if any(w in p.name for w in wanted) or p.stem.split(".")[0] in wanted
        ]
    if not scenario_files:
        print("no regress scenarios matched", file=sys.stderr)
        return 2

    out_root = args.out_dir or _DEFAULT_OUT_ROOT / date.today().isoformat() / "regress"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_root}", file=sys.stderr)

    results: list[ScenarioResult] = []
    for path in scenario_files:
        print(f"==> {path.name}", file=sys.stderr, flush=True)
        result = await _run_scenario(path, out_root, args.replays)
        results.append(result)

    summary_md = _render_summary(results, args.replays, out_root)
    (out_root / "summary.md").write_text(summary_md, encoding="utf-8")
    _dump_json(results, out_root / "summary.json")

    # Stdout: tight one-liner per scenario for grep-friendly CI logs.
    for r in results:
        status = "PASS" if r.overall_passed else "FAIL"
        print(f"{status}  {r.scenario_id}", flush=True)

    return 0 if all(r.overall_passed for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replays",
        type=int,
        default=10,
        help="Replays per scenario (default 10). Use --replays 3 for a smoke run.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[],
        help=(
            "Filter to scenarios whose name contains any of these strings "
            "(e.g. --scenarios srd-skip self-grep). Default: all."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=("Override output directory. Default: docs/bench/<today>/regress/."),
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
