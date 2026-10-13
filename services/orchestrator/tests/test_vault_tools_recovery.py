"""LLM-facing recovery behaviour of the vault tools.

The vault tools shape errors so the LLM rarely needs a follow-up call
to recover. These tests pin the two recovery paths that previously
cost 2-3 wasted turns on every cold session: ``read_file`` against a
directory (inline listing) and ``read_file`` against a near-miss
filename (glob suggestion).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aidm.vault import Vault
from aidm.vault_tools import (
    _read_file_recovery,
    build_vault_summary,
    discover_in_scene_npcs,
    discover_pc_slug,
    has_character_sheet,
    session_zero_tools_schema,
)


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "firearms-and-psionics.md").write_text("body", encoding="utf-8")
    (tmp_path / "npcs").mkdir()
    (tmp_path / "npcs" / "elara.md").write_text("body", encoding="utf-8")
    run_root = tmp_path / "runs" / "default"
    run_root.mkdir(parents=True)
    (run_root / "run.md").write_text("---\ntype: run\n---\n", encoding="utf-8")
    return Vault(tmp_path, run_slug="default")


def test_recovery_for_directory_returns_listing(vault: Vault) -> None:
    msg = _read_file_recovery(vault, "rules", "not a file: rules")
    assert "rules/firearms-and-psionics.md" in msg
    assert "is a directory" in msg


def test_recovery_for_missing_suggests_near_match(vault: Vault) -> None:
    msg = _read_file_recovery(
        vault,
        "firearms-and-psionics.md",
        "not a file: firearms-and-psionics.md",
    )
    assert "rules/firearms-and-psionics.md" in msg
    assert "Did you mean" in msg


def test_recovery_for_unknown_path_returns_plain_error(vault: Vault) -> None:
    msg = _read_file_recovery(vault, "zzz-no-such-thing.md", "not a file: zzz-no-such-thing.md")
    assert msg.startswith("error:")
    assert "Did you mean" not in msg


def test_build_vault_summary_includes_map_and_canon(vault: Vault, tmp_path: Path) -> None:
    # campaign.md is at the vault root and should be preloaded so the DM
    # never has to fabricate the opening scene.
    (tmp_path / "campaign.md").write_text(
        "---\ntype: campaign\n---\n# Test Campaign\nOpening: the Quiet Forge.\n",
        encoding="utf-8",
    )
    summary = build_vault_summary(vault, locale="en")
    assert "rules/firearms-and-psionics.md" in summary
    # NPC filenames are deliberately masked from the vault map at startup —
    # leaking the roster causes the DM to voice unentered NPCs on turn 1.
    # The directory entry stays visible with a placeholder so the model
    # still knows the folder exists and how to surface a name.
    assert "npcs/elara.md" not in summary
    assert "npcs/" in summary
    assert "roster hidden" in summary
    assert "campaign.md" in summary
    # Canon body inlined, not just the path
    assert "Opening: the Quiet Forge" in summary


def test_has_character_sheet_detects_run_subtree(vault: Vault, tmp_path: Path) -> None:
    assert not has_character_sheet(vault)
    (tmp_path / "runs" / "default" / "characters").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / "default" / "characters" / "lyra.md").write_text(
        "---\nname: Lyra\n---\n", encoding="utf-8"
    )
    assert has_character_sheet(vault)


def test_has_character_sheet_ignores_state_only_dir(vault: Vault, tmp_path: Path) -> None:
    """A stray ``.state.md`` without its identity sheet is not a character."""
    chars = tmp_path / "runs" / "default" / "characters"
    chars.mkdir(parents=True, exist_ok=True)
    (chars / "ghost.state.md").write_text("---\ntype: character_state\n---\n", encoding="utf-8")
    assert not has_character_sheet(vault)


def test_discover_pc_slug_ignores_state_sibling(vault: Vault, tmp_path: Path) -> None:
    """Regression: the writer's ``.state.md`` sibling made the single-PC
    count 2, silently disabling PC sheet injection for the whole run."""
    chars = tmp_path / "runs" / "default" / "characters"
    chars.mkdir(parents=True, exist_ok=True)
    (chars / "rowan.md").write_text("---\nname: Rowan\n---\n", encoding="utf-8")
    (chars / "rowan.state.md").write_text("---\ntype: character_state\n---\n", encoding="utf-8")
    assert discover_pc_slug(vault) == "rowan"


def test_discover_in_scene_npcs_reads_disk_flag(vault: Vault, tmp_path: Path) -> None:
    """Only NPCs whose state file carries ``in_scene: true`` are restored."""
    npcs = tmp_path / "npcs"
    # elara.md exists from the fixture — give her an in-scene state file.
    (npcs / "elara.state.md").write_text(
        "---\ntype: npc_state\nin_scene: true\n---\n", encoding="utf-8"
    )
    (npcs / "brom.md").write_text("---\nname: Brom\n---\n", encoding="utf-8")
    (npcs / "brom.state.md").write_text(
        "---\ntype: npc_state\nin_scene: false\n---\n", encoding="utf-8"
    )
    (npcs / "silent.md").write_text("---\nname: Silent\n---\n", encoding="utf-8")  # no state
    assert discover_in_scene_npcs(vault) == ["elara"]


@pytest.mark.asyncio
async def test_set_character_finds_sheet_through_run_overlay(vault: Vault, tmp_path: Path) -> None:
    """Regression: ``_set_character`` used to check the canon path
    ``vault/characters/<slug>.md`` directly, ignoring the run overlay
    that hosts the actual file under ``runs/<run-slug>/characters/``.
    The session-zero flow would write a sheet successfully and then
    report "no character sheet" — the LLM thought the write had
    failed and retried it.
    """
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    # Write a sheet through the overlay — lands under runs/default/characters/
    vault.write_character_sheet("rowan", "---\nname: Rowan\n---\n# Rowan\n")
    assert (tmp_path / "runs" / "default" / "characters" / "rowan.md").is_file()
    # The canon path the broken check used does NOT exist:
    assert not (tmp_path / "characters" / "rowan.md").is_file()

    handler = _handlers_for(vault)["vault_set_character"]
    params = MagicMock()
    params.arguments = {"slug": "rowan"}
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()

    await handler(params)

    result = params.result_callback.call_args.args[0]
    assert not result.startswith("error:"), f"expected success, got: {result}"
    assert "rowan" in result


@pytest.mark.asyncio
async def test_set_character_unknown_slug_lists_available(vault: Vault) -> None:
    """A bad slug should come back with the real options inline, so the
    LLM can retry on the next call instead of round-tripping through
    ``vault_glob``. Regression for a real run where the model passed a
    location name as the slug and then gave up.
    """
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    vault.write_character_sheet("rowan", "---\nname: Rowan\n---\n# Rowan\n")
    vault.write_character_sheet("lyra", "---\nname: Lyra\n---\n# Lyra\n")

    handler = _handlers_for(vault)["vault_set_character"]
    params = MagicMock()
    params.arguments = {"slug": "Iron Veil Citadel"}
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()

    await handler(params)

    result = params.result_callback.call_args.args[0]
    assert result.startswith("error:")
    assert "lyra" in result
    assert "rowan" in result
    assert "vault_glob" not in result


@pytest.mark.asyncio
async def test_scene_enter_unknown_slug_lists_available(vault: Vault) -> None:
    """`npc_scene_enter("thor")` against a vault that only has `npcs/thorne.md`
    should come back with the actual NPC slugs inline, not let the LLM
    silently enter a ghost NPC into the scene. Regression from a run
    where the model entered `thor` (typo for `thorne`) and the system
    happily accepted it.
    """
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.scene_state import SceneRegistry  # noqa: PLC0415
    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    vault.write_npc_sheet("thorne", "---\nname: Thorne\n---\n# Thorne\n")
    vault.write_npc_sheet("elara", "---\nname: Elara\n---\n# Elara\n")

    registry = SceneRegistry()
    handler = _handlers_for(vault, registry)["npc_scene_enter"]
    params = MagicMock()
    params.arguments = {"slug": "thor"}
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()

    await handler(params)

    result = params.result_callback.call_args.args[0]
    assert result.startswith("error:")
    assert "thorne" in result
    assert "elara" in result
    assert "thor" not in registry.in_scene


@pytest.mark.asyncio
async def test_scene_enter_is_idempotent(vault: Vault) -> None:
    """Re-entering an already-in-scene NPC should be a no-op with a
    clear signal — not silently bump the turn counter. Stops the
    re-enter storms seen in the test run logs.
    """
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.scene_state import SceneRegistry  # noqa: PLC0415
    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    vault.write_npc_sheet("elara", "---\nname: Elara\n---\n# Elara\n")
    registry = SceneRegistry()
    registry.enter("elara")

    handler = _handlers_for(vault, registry)["npc_scene_enter"]
    params = MagicMock()
    params.arguments = {"slug": "elara"}
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()

    await handler(params)

    result = params.result_callback.call_args.args[0]
    assert "already in scene" in result
    assert registry.in_scene == {"elara"}


@pytest.mark.asyncio
async def test_update_npc_unchanged_skips_identity_write(vault: Vault, tmp_path: Path) -> None:
    """`vault_update_npc` with identity content matching the on-disk file
    must not touch the identity file. Status/location etc. route to the
    state file regardless; what we're guarding against is the identity-
    sheet rewrite churn that bloats context for the rest of the session.
    """
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    identity = "---\ntype: npc\nname: Elara\n---\n\n# Elara\n\nGentle, kind, tends the fire.\n"
    vault.write_npc_sheet("elara", identity)
    npc_path = tmp_path / "npcs" / "elara.md"
    before_mtime = npc_path.stat().st_mtime_ns

    handler = _handlers_for(vault)["vault_update_npc"]
    params = MagicMock()
    params.arguments = {"slug": "elara", "content": identity}
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()

    await handler(params)

    result = params.result_callback.call_args.args[0]
    assert "written to" in result
    assert npc_path.stat().st_mtime_ns == before_mtime  # identity untouched


@pytest.mark.asyncio
async def test_update_npc_routes_state_fields_to_state_file(vault: Vault, tmp_path: Path) -> None:
    """A status flip is a state-file change. Identity stays put; state file is created."""
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    vault.write_npc_sheet(
        "elara",
        "---\ntype: npc\nname: Elara\n---\n\n# Elara\n\nGentle.\n",
    )
    handler = _handlers_for(vault)["vault_update_npc"]
    params = MagicMock()
    params.arguments = {
        "slug": "elara",
        "content": (
            "---\ntype: npc\nname: Elara\nstatus: wounded\nlocation: forge\n---\n\n"
            "# Elara\n\nGentle.\n"
        ),
    }
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()

    await handler(params)

    result = params.result_callback.call_args.args[0]
    assert "written to" in result
    state_path = tmp_path / "runs" / "default" / "npcs" / "elara.state.md"
    assert state_path.is_file()
    state = state_path.read_text(encoding="utf-8")
    assert "status: wounded" in state
    assert "location: forge" in state


def test_session_zero_subset_excludes_play_only_tools(vault: Vault) -> None:
    schema = session_zero_tools_schema()
    names = {f.name for f in schema.standard_tools}
    # Allowed in session zero
    assert "vault_update_character" in names
    assert "vault_set_character" in names
    assert "vault_read_file" in names
    # NPC sheet writes are allowed so the interviewer can pre-seed the
    # starting party before play opens.
    assert "vault_update_npc" in names
    # Play-only tools dropped
    assert "vault_set_scene" not in names
    assert "vault_write_session_log" not in names
    assert "npc_scene_enter" not in names


@pytest.mark.asyncio
async def test_set_character_registers_the_pc_for_the_session(vault: Vault) -> None:
    """A PC created in session zero must become the session's PC: without it
    the sheet injector, the PC writer patch and Suggest action (409) all ran
    PC-less for the whole 2026-09-27 playtests."""
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.scene_state import SceneRegistry  # noqa: PLC0415
    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    vault.write_character_sheet("rowan", "---\nname: Rowan\n---\n# Rowan\n")
    registry = SceneRegistry()
    handler = _handlers_for(vault, registry)["vault_set_character"]
    params = MagicMock()
    params.arguments = {"slug": "rowan"}
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()

    await handler(params)

    assert registry.current_pc_slug == "rowan"


async def _call_update_npc(vault: Vault, slug: str, content: str) -> str:
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    from aidm.vault_tools import _handlers_for  # noqa: PLC0415

    params = MagicMock()
    params.arguments = {"slug": slug, "content": content}
    params.result_callback = AsyncMock()
    params.llm = MagicMock()
    params.llm.push_frame = AsyncMock()
    await _handlers_for(vault)["vault_update_npc"](params)
    return str(params.result_callback.call_args.args[0])


@pytest.mark.asyncio
async def test_update_npc_never_rewrites_an_authored_identity(vault: Vault, tmp_path: Path) -> None:
    """2026-09-27: a session-zero pass rewrote all six authored campaign NPC sheets
    into shorter versions. Authored canon is shared by every run — kept."""
    authored = "---\ntype: npc\nname: Elara\n---\n\n# Elara\n\nGentle.\n\n## Secret\n\nShe lies.\n"
    vault.write_npc_sheet("elara", authored)
    result = await _call_update_npc(
        vault,
        "elara",
        "---\ntype: npc\nname: Elara\nmood: wary\n---\n\n# Elara\n\nGentle.\n",
    )
    assert "authored canon" in result
    assert vault.read_file("npcs/elara.md") == authored
    state = (tmp_path / "runs" / "default" / "npcs" / "elara.state.md").read_text(encoding="utf-8")
    assert "mood: wary" in state


@pytest.mark.asyncio
async def test_update_npc_still_creates_new_npcs_and_fills_stubs(vault: Vault) -> None:
    from aidm.scene_sheet_writer import _STUB_GLOSS_MARKER  # noqa: PLC0415

    new = "---\ntype: npc\nname: Pál\n---\n\n# Pál\n\nThe tired sentry.\n"
    assert "written to" in await _call_update_npc(vault, "pal", new)
    assert "tired sentry" in vault.read_file("npcs/pal.md")

    vault.write_npc_sheet(
        "figure", f"---\ntype: npc\nname: Figure\n---\n\n**Gloss:** {_STUB_GLOSS_MARKER}\n"
    )
    filled = "---\ntype: npc\nname: Figure\n---\n\n# Figure\n\nA hooded scout.\n"
    assert "written to" in await _call_update_npc(vault, "figure", filled)
    assert "hooded scout" in vault.read_file("npcs/figure.md")
