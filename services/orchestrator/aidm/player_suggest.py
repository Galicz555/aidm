"""Generate an in-character player action from PC sheet + recent context.

Drives the text-mode "Suggest action" button. The point is sheet-driven
RP: the suggestion comes from who the PC is and what just happened, not
from what the player would type. The result lands in the text input
field for the player to edit or send as-is.

Same backend as the sheet writer — a one-shot ``AsyncOpenAI`` call to
the configured LLM (private gateway in private mode, OpenRouter in
public). Re-uses ``scene_sheet_writer.collect_npc_state`` for the
in-scene NPC summary so prompt assembly stays in one place.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from aidm.config import Config
from aidm.scene_state import SceneRegistry
from aidm.vault import Vault, VaultPathError

_PROMPT: dict[str, str] = {
    "en": (
        "You are the player of {pc_name} in a tabletop RPG session. Generate "
        "ONE short in-character action or line of dialogue the player might "
        "take next, based on the character sheet and the situation below.\n\n"
        "PC sheet:\n```\n{pc_sheet}\n```\n\n"
        "{npc_block}"
        "Recent exchange:\n{history}\n\n"
        "Rules:\n"
        '- 1-3 sentences. First-person ("I…") or quoted dialogue.\n'
        "- Drive from the sheet — HEXACO traits, recent events, inventory.\n"
        "- React to the most recent DM beat; don't ignore it.\n"
        "- No DM narration, no choice menus, no meta commentary.\n"
        "- Output ONLY the action. No preamble, no quotes around the whole thing."
    ),
    "hu": (
        "{pc_name} játékosa vagy egy szerepjáték-szessionben. Adj EGY rövid "
        "karakterben maradt akciót vagy párbeszédsort, amit a játékos "
        "tehetne, a karakterlap és az alábbi helyzet alapján.\n\n"
        "PC lap:\n```\n{pc_sheet}\n```\n\n"
        "{npc_block}"
        "Legutóbbi váltás:\n{history}\n\n"
        "Szabályok:\n"
        '- 1-3 mondat. Első személyben ("Én…") vagy idézett párbeszéd.\n'
        "- A lapból építkezz — HEXACO, recent események, leltár.\n"
        "- Reagálj a DM legutóbbi beat-jére; ne hagyd figyelmen kívül.\n"
        "- Semmi DM-narráció, semmi választékmenü, semmi meta.\n"
        "- CSAK az akciót add ki. Sem preambulum, sem körülölelő idézőjel."
    ),
}

_HISTORY_WINDOW = 6  # last 6 messages (~3 user/DM pairs)
_MAX_TOKENS = 200


class SuggestionError(Exception):
    """Raised when a suggestion can't be produced. Carries an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


async def suggest_player_action(
    config: Config,
    vault: Vault,
    registry: SceneRegistry,
    context: Any,
    locale: str,
) -> str:
    """One-shot LLM call returning a sheet-driven in-character suggestion."""
    pc_slug = registry.current_pc_slug
    if not pc_slug:
        raise SuggestionError(409, "no PC sheet active in this session")
    try:
        pc_sheet = vault.read_file(f"characters/{pc_slug}.md")
    except VaultPathError as exc:
        raise SuggestionError(404, f"PC sheet missing: {exc}") from exc
    pc_name = _pc_name_from_sheet(pc_sheet, fallback=pc_slug)

    npc_block = _npc_block(vault, registry, locale)
    history = _format_history(getattr(context, "messages", []))

    template = _PROMPT.get(locale, _PROMPT["en"])
    user_msg = template.format(
        pc_name=pc_name,
        pc_sheet=pc_sheet.rstrip(),
        npc_block=npc_block,
        history=history,
    )

    client, model = _build_client(config)
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=_MAX_TOKENS,
            temperature=0.8,
            timeout=config.npc_failsafe_timeout_s,
        )
    except Exception as exc:
        logger.warning("suggest_player_action: llm error {}: {}", type(exc).__name__, exc)
        raise SuggestionError(504, f"llm error: {type(exc).__name__}") from exc

    text = _extract_text(completion).strip()
    if not text:
        raise SuggestionError(502, "llm returned empty suggestion")
    # Strip wrapping quotes the writer sometimes adds despite the rule.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        text = text[1:-1].strip() or text
    return text


def _build_client(config: Config) -> tuple[AsyncOpenAI, str]:
    if config.mode == "private":
        return (
            AsyncOpenAI(base_url=config.local_llm_base_url, api_key="local-llama-cpp"),
            config.local_llm_model,
        )
    return (
        AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=config.openrouter_api_key),
        config.openrouter_model,
    )


def _pc_name_from_sheet(sheet: str, *, fallback: str) -> str:
    from aidm import frontmatter  # noqa: PLC0415 — avoid import cycle on cold path

    fm, _ = frontmatter.split(sheet)
    name = fm.get("name") if fm else None
    return str(name).strip() if isinstance(name, str) and name.strip() else fallback


def _npc_block(vault: Vault, registry: SceneRegistry, locale: str) -> str:
    from aidm.scene_sheet_writer import collect_npc_state  # noqa: PLC0415

    states = collect_npc_state(vault, registry.active())
    if not states:
        return ""
    header = "Jelenetben (NPC-k):\n" if locale == "hu" else "In scene (NPCs):\n"
    lines = [header]
    for s in states:
        bits = [s["slug"]]
        if s["mood"]:
            bits.append(f"mood={s['mood']}")
        if s["scene_goal"]:
            bits.append(f"goal={s['scene_goal']}")
        if s["social_mask"]:
            bits.append(f"mask={s['social_mask']}")
        lines.append("- " + ", ".join(bits))
    return "\n".join(lines) + "\n\n"


def _format_history(messages: list[Any]) -> str:
    """Render the last N user/assistant turns as plain-text dialogue."""
    if not messages:
        return "(no prior turns)"
    chosen: list[tuple[str, str]] = []
    for msg in reversed(messages):
        role = _role(msg)
        if role not in {"user", "assistant"}:
            continue
        text = _content(msg).strip()
        if not text:
            continue
        chosen.append((role, text))
        if len(chosen) >= _HISTORY_WINDOW:
            break
    chosen.reverse()
    label = {"user": "Player", "assistant": "DM"}
    return "\n".join(f"{label[r]}: {t}" for r, t in chosen) or "(no prior turns)"


def _role(message: Any) -> str:
    raw = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
    return str(raw or "").lower()


def _content(message: Any) -> str:
    raw = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_text(completion: Any) -> str:
    try:
        return str(completion.choices[0].message.content or "")
    except (AttributeError, IndexError):
        return ""
