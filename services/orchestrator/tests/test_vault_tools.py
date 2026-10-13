"""vault_tools argument hygiene and sheet normalisation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aidm.vault import Vault
from aidm.vault_schema import HEXACO_DIMENSIONS, hexaco_body_score
from aidm.vault_tools import (
    _handlers_for,
    _split_incoming_sheet,
    _strip_tool_markup,
    _with_hexaco_block,
    normalize_character_sheet,
)


def test_leaked_tool_markup_is_cut_from_an_argument() -> None:
    # 2026-09-27 playtest beat.md: the next XML parameter leaked into content.
    leaked = 'Tension 4/10.</content>\n<parameter name="name">The wire at dusk'
    assert _strip_tool_markup(leaked) == "Tension 4/10."
    assert _strip_tool_markup("A <b>bold</b> clue stays.") == "A <b>bold</b> clue stays."


# --- session-zero character sheets come out lint-valid -----------------------

_SESSION_ZERO_SHEET = """---
name: Rowan
class_role: Aspirant
status: alive
location: the guarded district
hexaco_honesty_humility: 4
hexaco_emotionality: 3
hexaco_extraversion: 2
hexaco_agreeableness: 4
hexaco_conscientiousness: 3
hexaco_openness: 4
---

# Rowan

A teenager in a guarded city district.

## Psychology (HEXACO)

- Honesty-Humility: 4 — no pretensions.
- Emotionality: steady on the surface.
- Extraversion: shy.
- Agreeableness: goes along.
- Conscientiousness: gets things done.

## Want and fear

- **Want:** to be free.
"""


def _lint(path: Path) -> list[str]:
    from scripts.lint_vault import _lint_file  # noqa: PLC0415

    return [issue.message for issue in _lint_file(path)]


def test_a_session_zero_sheet_is_normalised_to_pass_the_vault_lint(tmp_path: Path) -> None:
    identity, state = normalize_character_sheet(
        *_split_incoming_sheet(_SESSION_ZERO_SHEET, kind="pc"), locale="en"
    )
    assert identity is not None and state is not None
    (tmp_path / "rowan.md").write_text(identity, encoding="utf-8")
    (tmp_path / "rowan.state.md").write_text(
        state.replace("---\n", "---\nlast_updated: 2026-09-27\n", 1), encoding="utf-8"
    )

    assert _lint(tmp_path / "rowan.md") == []
    assert _lint(tmp_path / "rowan.state.md") == []
    assert "lang: en" in identity
    assert "status: active" in state
    # The model's own readings survive; its stray "4 —" does not.
    assert "- Honesty-Humility (4): no pretensions." in identity
    assert "- Openness (4)\n" in identity  # a trait the model skipped is still scored
    assert identity.index("## Psychology") < identity.index("## Want and fear")


def test_a_valid_sheet_passes_the_normaliser_untouched() -> None:
    identity, _ = normalize_character_sheet(
        *_split_incoming_sheet(_SESSION_ZERO_SHEET, kind="pc"), locale="en"
    )
    assert normalize_character_sheet(identity, None, locale="hu") == (identity, None)


@pytest.mark.asyncio
async def test_vault_update_character_writes_a_lint_valid_sheet(tmp_path: Path) -> None:
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    vault = Vault(tmp_path, run_slug="r1")
    params = MagicMock()
    params.arguments = {"slug": "rowan", "content": _SESSION_ZERO_SHEET}
    params.result_callback = AsyncMock()

    await _handlers_for(vault, locale="en")["vault_update_character"](params)

    assert _lint(tmp_path / "runs" / "r1" / "characters" / "rowan.md") == []
    assert _lint(tmp_path / "runs" / "r1" / "characters" / "rowan.state.md") == []


@pytest.mark.asyncio
async def test_the_session_log_result_asks_for_a_short_out_of_character_goodbye(
    tmp_path: Path,
) -> None:
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    vault = Vault(tmp_path, run_slug="r1")
    params = MagicMock()
    params.arguments = {"content": "---\ntype: session_log\n---\n" + "We reached the gate. " * 8}
    params.result_callback = AsyncMock()

    await _handlers_for(vault)["vault_write_session_log"](params)

    result = params.result_callback.call_args.args[0]
    assert result.startswith("session log written to sessions/")
    assert "OUT OF CHARACTER in 1-3 short sentences" in result
    assert "Do not continue the story" in result


def test_the_normaliser_patches_only_the_trait_bullets() -> None:
    # Review #9: the whole section was replaced (prose and ### subsections lost)
    # and a differently titled heading got a duplicate block appended.
    body = (
        "# Rowan\n\n"
        "### HEXACO profile\n\n"
        "Read from how they talk about home.\n\n"
        "- **Honesty-Humility**: 4 — plain.\n"
        "- Emotionality: steady.\n"
        "- Extraversion (2): shy.\n"
        "- Agreeableness: goes along.\n"
        "- Conscientiousness: drifts.\n\n"
        "#### Under pressure\n\n"
        "Freezes first, then moves.\n\n"
        "## Want and fear\n\n- **Want:** freedom.\n"
    )
    fm = {f"hexaco_{d}": s for d, s in zip(HEXACO_DIMENSIONS, (4, 3, 2, 4, 3, 4), strict=True)}

    fixed = _with_hexaco_block(body, fm)

    assert "Read from how they talk about home." in fixed
    assert "#### Under pressure\n\nFreezes first, then moves." in fixed
    assert "- Honesty-Humility (4): plain." in fixed
    assert "- Emotionality (3): steady." in fixed
    assert "- Openness (4)" in fixed  # the missing trait joins the same section
    assert fixed.index("- Openness (4)") < fixed.index("## Want and fear")
    assert fixed.count("HEXACO") == 1  # no second block
    assert all(hexaco_body_score(fixed, d) == str(fm[f"hexaco_{d}"]) for d in HEXACO_DIMENSIONS)
    assert _with_hexaco_block(fixed, fm) == fixed
