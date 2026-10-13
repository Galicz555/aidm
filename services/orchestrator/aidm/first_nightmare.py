"""First Nightmare — the opt-in campaign feature ``first-nightmare``.

Code owns the structure, the model writes prose (docs/deterministic-
storytelling.md). A playtest left the whole trial to the DM: no Spell
greeting, no runes, no concrete conflict, no appraisal, and the dream
"ended" after six turns with nothing resolved — while the story clock
printed "*Day 1, afternoon.*" inside it. Here code decides when the trial
starts and ends, what it is about, and what the Spell says; the DM only
narrates inside the frame.

Lifecycle, all state on disk in ``runs/<run>/arcs/first-nightmare.md``
(an ``arc`` the DM may read, written only by code through ``Vault`` — so
the text regenerate journal rolls every step back):

- no arc file: not started. Each turn a logprob judge asks whether the
  PC fell asleep (``advance_judge.judge_falls_asleep``).
- ``status: dormant``: asleep, plan not written yet (the structured plan
  call failed; it is retried every turn).
- ``status: active``: the plan is on the sheet; the DM sees it every turn
  in the scene-sheets block; ``trial_turns`` counts turns inside. From
  ``MIN_TRIAL_TURNS`` on, a second judge asks whether the win or lose
  condition happened.
- ``status: resolved`` (``outcome: survived | died``): done for good.

While dormant or active the story clock is frozen (``story_events``).
The Spell's greeting and appraisal are code-written lines rendered from
the campaign's ``rules/spell-voice.md`` and shown once at the start of
the next reply (``DirectorState.spell_line`` → ``TimeShownProcessor``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger

from aidm import frontmatter
from aidm.vault import Vault, VaultPathError

FEATURE = "first-nightmare"
ARC_SLUG = "first-nightmare"
ARC_PATH = f"arcs/{ARC_SLUG}.md"
VOICE_PATH = "rules/spell-voice.md"
# Turns inside the trial before the end judge may run.
MIN_TRIAL_TURNS = 3
_PLAN_ATTEMPTS = 2
_MAX_FIELD_CHARS = 500
_MAX_NAME_CHARS = 60
_SOURCE_MAX_CHARS = 6000

# English-only by decision (docs/decisions/english-only.md). The Spell's
# player-facing wording lives in the campaign vault (rules/spell-voice.md).
_PLAN_PROMPT = (
    "You are the Nightmare Spell designing one person's First Nightmare: a trial dream that "
    "reconstructs a moment of the Dream Realm's ancient past. Build it from THIS character "
    "- their want, their fear, their skills - and from the rules and lore below.\n\n"
    "CHARACTER SHEET:\n{sheet}\n\nRULES AND LORE:\n{sources}\n\n"
    "Never reuse any of these, they belong to someone else's trial: {never}.\n\n"
    "Return a single JSON object - no prose, no code fences - with EXACTLY these keys, every "
    "string one line of plain text:\n"
    "  title - the trial's name, 2-6 words\n"
    "  setting - the reconstructed moment of the ancient past: where, when, what is happening\n"
    "  role - who the character wakes up as inside it: a weak, constrained starting position\n"
    "  conflict - ONE concrete central conflict (a siege, an escape, a hunt...) with one "
    "antagonist or threat that has a discoverable weakness\n"
    "  stages - a list of exactly 3 strings: how the conflict escalates, in order\n"
    "  win - the concrete event that resolves the conflict in the character's favour\n"
    "  lose - the concrete event that means the character dies or fails\n"
    "  aspect - the Dormant Aspect's rune name, 1-3 words, no brackets\n"
    "  aspect_description - one line in the Spell's dry, faintly mocking voice\n"
    "  aspect_ability - the ability revealed at the end: one clear effect, its cost, its limit\n"
    "  flaw - the Flaw revealed with it, forged from the same material, with a trigger anyone "
    "would recognise\n"
    '  attributes - a list of 1-3 short rune names, e.g. "[Keen Hearing]"\n\n'
    "Output JSON ONLY. Start with `{{` and end with `}}`."
)
_APPRAISAL_PROMPT = (
    "A First Nightmare trial just ended.\n\nWIN CONDITION: {win}\nLOSE CONDITION: {lose}\n\n"
    "WHAT HAPPENED (the character's running record):\n{record}\n\n"
    "FINAL TURN:\nPlayer: {player}\nGame master: {reply}\n\n"
    "Return a single JSON object - no prose, no code fences - with exactly these keys:\n"
    '  outcome - "survived" or "died"\n'
    "  appraisal - one word from this list, judging how the character handled the trial "
    "(clever or costly survival rates higher than brute force): {words}\n"
    "Output JSON ONLY."
)
_DM_HEADER = (
    "First Nightmare — the player is INSIDE their trial (turn {turn}). The Spell built it; "
    "run THIS trial: one concrete conflict, escalating through the stages, ending only on "
    "the win or lose condition. The waking world's clock is stopped. Narrate the dream "
    "concretely, never as vague symbolic prose"
)
_DM_RUNES = (
    "Runes — reveal ONLY when the player checks their runes: Name, Rank Aspirant, the Aspect "
    "with its description, the Attributes. The Aspect ability and the Flaw stay hidden; the "
    "Spell announces them at the end (code prints it, not you)"
)
_DM_DORMANT = (
    "First Nightmare — the player has fallen asleep and the Spell is preparing their trial. "
    "Keep the descent into sleep and darkness brief; do not invent the trial yet. The waking "
    "world's clock is stopped"
)


@dataclass(frozen=True)
class Plan:
    title: str
    setting: str
    role: str
    conflict: str
    stages: tuple[str, ...]
    win: str
    lose: str
    aspect: str
    aspect_description: str
    aspect_ability: str
    flaw: str
    attributes: tuple[str, ...]


@dataclass(frozen=True)
class Trial:
    status: str  # dormant | active | resolved
    turns: int
    plan: Plan | None


@dataclass(frozen=True)
class Voice:
    greeting: str
    appraisal: str
    sheet_runes: str
    words: tuple[str, ...]
    never_reuse: tuple[str, ...]
    plan_sources: tuple[str, ...]

    @property
    def default_word(self) -> str:
        return self.words[-2] if len(self.words) >= 2 else (self.words[0] if self.words else "")


# --- reading -----------------------------------------------------------------


def enabled(vault: Vault) -> bool:
    """The campaign opted in (``features: [first-nightmare]`` in campaign.md)."""
    fm = frontmatter.parse(_read(vault, "campaign.md"))
    features = fm.get("features") or []
    if isinstance(features, str):
        features = [features]
    return FEATURE in {str(f).strip() for f in features}


def read_trial(vault: Vault) -> Trial | None:
    """The run's trial, or None when it never started."""
    text = _read(vault, ARC_PATH)
    if not text:
        return None
    fm, body = frontmatter.split(text)
    status = str(fm.get("status") or "")
    if status not in ("dormant", "active", "resolved"):
        return None
    try:
        turns = int(fm.get("trial_turns") or 0)
    except (TypeError, ValueError):
        turns = 0
    return Trial(status=status, turns=turns, plan=_parse_plan(body))


def read_voice(vault: Vault) -> Voice | None:
    text = _read(vault, VOICE_PATH)
    if not text:
        return None
    fm, body = frontmatter.split(text)
    sections = _sections(body)
    sources = fm.get("plan_sources") or []
    return Voice(
        greeting=sections.get("Greeting", ""),
        appraisal=sections.get("Appraisal", ""),
        sheet_runes=sections.get("Sheet runes", ""),
        words=tuple(_bullets(sections.get("Appraisal words", ""))),
        never_reuse=tuple(_bullets(sections.get("Never reuse", ""))),
        plan_sources=tuple(str(s) for s in (sources if isinstance(sources, list) else [sources])),
    )


def dm_block(vault: Vault) -> str:
    """The scene-sheets block for a running trial; "" when none runs (or the feature is off)."""
    if not enabled(vault):
        return ""
    trial = read_trial(vault)
    if trial is None or trial.status == "resolved":
        return ""
    if trial.status == "dormant" or trial.plan is None:
        return f"[{_DM_DORMANT}]"
    p = trial.plan
    stages = "\n".join(f"{i}. {s}" for i, s in enumerate(p.stages, start=1))
    return (
        f"[{_DM_HEADER.format(turn=trial.turns + 1)}]\n\n"
        f"Trial: {p.title}\nSetting: {p.setting}\nThe player's role: {p.role}\n"
        f"Central conflict: {p.conflict}\nStages:\n{stages}\n"
        f"Win: {p.win}\nLose: {p.lose}\n\n"
        f"[{_DM_RUNES}]\n"
        f"- Aspect: [{p.aspect}] — {p.aspect_description}\n"
        f"- Attributes: {', '.join(p.attributes)}\n"
        f"- (hidden until the end) Aspect ability: {p.aspect_ability}\n"
        f"- (hidden until the end) Flaw: {p.flaw}\n\n---"
    )


# --- transitions (all writes through Vault) ------------------------------------


def start(vault: Vault) -> None:
    """The PC fell asleep: mark the trial begun (dormant until the plan lands)."""
    vault.write_arc(ARC_SLUG, _render_arc(vault, "dormant", 0, None))
    logger.info("first_nightmare: the PC fell asleep — trial started (plan pending)")


async def write_plan(
    vault: Vault, client: Any, model: str, *, pc_slug: str, timeout_s: float
) -> Plan | None:
    """One structured call (retried once) for the plan; on success the trial goes active.

    All-or-nothing validation, like stub consolidation: every key present,
    one-line strings, exactly 3 stages, 1-3 attributes, nothing from the
    campaign's never-reuse list. A failure leaves the trial dormant — the
    next turn tries again.
    """
    voice = read_voice(vault)
    prompt = _PLAN_PROMPT.format(
        sheet=_read(vault, f"characters/{pc_slug}.md") or "(no sheet)",
        sources=_plan_sources(vault, voice),
        never=", ".join(voice.never_reuse) if voice and voice.never_reuse else "(none)",
    )
    for attempt in range(1, _PLAN_ATTEMPTS + 1):
        raw = await _complete(
            client, model, prompt, max_tokens=1200, temperature=0.8, timeout_s=timeout_s
        )
        plan = _validate_plan(_parse_json(raw), voice.never_reuse if voice else ())
        if plan is not None:
            trial = read_trial(vault)
            vault.write_arc(
                ARC_SLUG, _render_arc(vault, "active", trial.turns if trial else 0, plan)
            )
            logger.info(
                "first_nightmare: plan written — {!r}, aspect [{}] (attempt {})",
                plan.title,
                plan.aspect,
                attempt,
            )
            return plan
        logger.warning("first_nightmare: plan rejected (attempt {}): {!r}", attempt, raw[:300])
    return None


def count_turn(vault: Vault, trial: Trial) -> Trial:
    """One more turn inside the trial, persisted."""
    counted = replace(trial, turns=trial.turns + 1)
    vault.write_arc(ARC_SLUG, _render_arc(vault, trial.status, counted.turns, trial.plan))
    return counted


async def finish(  # noqa: PLR0913 — keyword-only turn inputs
    vault: Vault,
    client: Any,
    model: str,
    trial: Trial,
    *,
    pc_slug: str,
    player: str,
    reply: str,
    timeout_s: float,
) -> str:
    """Resolve the trial; return the appraisal block to show ("" when the PC died).

    Death is never enforced: it is logged and left to the DM. A survival
    writes the runes into the PC identity sheet (``## Runes``). An
    unreadable verdict fails open to a survival with the default word.
    """
    plan = trial.plan
    voice = read_voice(vault)
    if plan is None or voice is None:
        logger.warning("first_nightmare: trial ended without a plan or rules/spell-voice.md")
        vault.write_arc(ARC_SLUG, _render_arc(vault, "resolved", trial.turns, plan))
        return ""
    outcome, word = await _appraise(
        vault, client, model, plan, voice, pc_slug, player, reply, timeout_s
    )
    vault.write_arc(ARC_SLUG, _render_arc(vault, "resolved", trial.turns, plan, outcome=outcome))
    if outcome == "died":
        logger.warning(
            "first_nightmare: the trial ended in the lose condition — PC not killed by code; "
            "the DM narrates the consequence"
        )
        return ""
    values = _template_values(vault, plan, pc_slug, appraisal=word)
    _write_sheet_runes(vault, pc_slug, _fill(voice.sheet_runes, values))
    logger.info("first_nightmare: trial survived — appraised {}, runes written", word)
    return _fill(voice.appraisal, values)


def greeting(vault: Vault, plan: Plan, pc_slug: str) -> str:
    """The Spell's greeting for the first reply of the trial ("" without a voice file)."""
    voice = read_voice(vault)
    if voice is None or not voice.greeting:
        logger.warning("first_nightmare: no greeting in {} — none shown", VOICE_PATH)
        return ""
    return _fill(voice.greeting, _template_values(vault, plan, pc_slug))


# --- internals -----------------------------------------------------------------


async def _appraise(  # noqa: PLR0913 — one call's inputs
    vault: Vault,
    client: Any,
    model: str,
    plan: Plan,
    voice: Voice,
    pc_slug: str,
    player: str,
    reply: str,
    timeout_s: float,
) -> tuple[str, str]:
    record = _section_text(vault.read_state("pc", pc_slug) or "", "Session so far") or "(none)"
    raw = await _complete(
        client,
        model,
        _APPRAISAL_PROMPT.format(
            win=plan.win,
            lose=plan.lose,
            record=record,
            player=player,
            reply=reply,
            words=", ".join(voice.words),
        ),
        max_tokens=60,
        temperature=0.0,
        timeout_s=timeout_s,
    )
    data = _parse_json(raw) or {}
    outcome = str(data.get("outcome") or "").strip().lower()
    word = str(data.get("appraisal") or "").strip()
    by_fold = {w.casefold(): w for w in voice.words}
    if outcome not in ("survived", "died") or word.casefold() not in by_fold:
        logger.warning(
            "first_nightmare: appraisal unreadable ({!r}) — survived, {}",
            raw[:200],
            voice.default_word,
        )
        return "survived", voice.default_word
    return outcome, by_fold[word.casefold()]


async def _complete(  # noqa: PLR0913 — one call's knobs
    client: Any, model: str, prompt: str, *, max_tokens: int, temperature: float, timeout_s: float
) -> str:
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_s,
        )
    except Exception as exc:  # fail-open: the caller treats "" as no answer
        logger.warning("first_nightmare: LLM call FAILED — {}: {}", type(exc).__name__, exc)
        return ""
    return completion.choices[0].message.content or ""


def _validate_plan(  # noqa: PLR0911 — all-or-nothing: every failed check rejects
    data: dict[str, Any] | None, never_reuse: tuple[str, ...]
) -> Plan | None:
    if data is None:
        return None
    text_keys = (
        "title", "setting", "role", "conflict", "win", "lose",
        "aspect", "aspect_description", "aspect_ability", "flaw",
    )  # fmt: skip
    values: dict[str, str] = {}
    for key in text_keys:
        value = _one_line(data.get(key))
        if not value or len(value) > _MAX_FIELD_CHARS:
            return None
        values[key] = value
    if len(values["title"]) > _MAX_NAME_CHARS or len(values["aspect"]) > _MAX_NAME_CHARS:
        return None
    values["aspect"] = values["aspect"].strip("[]").strip()
    stages = data.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        return None
    stage_lines = tuple(_one_line(s) for s in stages)
    attributes = data.get("attributes")
    if isinstance(attributes, str):
        attributes = re.findall(r"\[[^\]]+\]", attributes) or [attributes]
    if not isinstance(attributes, list) or not 1 <= len(attributes) <= 3:
        return None
    attrs = tuple(f"[{_one_line(a).strip('[]').strip()}]" for a in attributes)
    if not all(stage_lines) or any(len(a) <= 2 or len(a) > _MAX_NAME_CHARS for a in attrs):
        return None
    haystack = " ".join([*values.values(), *stage_lines, *attrs]).casefold()
    reused = [n for n in never_reuse if n.casefold() in haystack]
    if reused:
        logger.warning("first_nightmare: plan reuses {} — rejected", reused)
        return None
    return Plan(stages=stage_lines, attributes=attrs, **values)


def _render_arc(
    vault: Vault, status: str, turns: int, plan: Plan | None, *, outcome: str = ""
) -> str:
    fm: dict[str, Any] = {
        "type": "arc",
        "lang": _campaign_lang(vault),
        "name": plan.title if plan else "First Nightmare",
        "status": status,
        "trial_turns": turns,
    }
    if outcome:
        fm["outcome"] = outcome
    if plan is None:
        return frontmatter.serialize(fm, "# First Nightmare\n\nThe trial is being prepared.\n")
    fm["driving_question"] = plan.conflict
    stages = "\n".join(f"{i}. {s}" for i, s in enumerate(plan.stages, start=1))
    body = (
        f"# First Nightmare — {plan.title}\n\n"
        f"## Setting\n\n{plan.setting}\n\n"
        f"## Role\n\n{plan.role}\n\n"
        f"## Central conflict\n\n{plan.conflict}\n\n"
        f"## Stages\n\n{stages}\n\n"
        f"## Win condition\n\n{plan.win}\n\n"
        f"## Lose condition\n\n{plan.lose}\n\n"
        "## Runes\n\n"
        f"- Aspect: {plan.aspect}\n"
        f"- Aspect description: {plan.aspect_description}\n"
        f"- Aspect ability: {plan.aspect_ability}\n"
        f"- Flaw: {plan.flaw}\n"
        f"- Attributes: {', '.join(plan.attributes)}\n"
    )
    return frontmatter.serialize(fm, body)


def _parse_plan(body: str) -> Plan | None:
    sections = _sections(body)
    runes = {k: v for k, _, v in (b.partition(": ") for b in _bullets(sections.get("Runes", "")))}
    title = body.split("\n", 1)[0].removeprefix("# First Nightmare — ").strip()
    stages = tuple(
        re.sub(r"^\d+\.\s*", "", line).strip()
        for line in sections.get("Stages", "").splitlines()
        if line.strip()
    )
    try:
        plan = Plan(
            title=title,
            setting=sections["Setting"],
            role=sections["Role"],
            conflict=sections["Central conflict"],
            stages=stages,
            win=sections["Win condition"],
            lose=sections["Lose condition"],
            aspect=runes["Aspect"],
            aspect_description=runes["Aspect description"],
            aspect_ability=runes["Aspect ability"],
            flaw=runes["Flaw"],
            attributes=tuple(re.findall(r"\[[^\]]+\]", runes["Attributes"])),
        )
    except KeyError:
        return None
    return plan


def _write_sheet_runes(vault: Vault, pc_slug: str, runes: str) -> None:
    """Replace (or append) the identity sheet's ``## Runes`` section; frontmatter untouched."""
    sheet = _read(vault, f"characters/{pc_slug}.md")
    if not sheet:
        logger.warning("first_nightmare: no identity sheet for {} — runes not written", pc_slug)
        return
    block = f"## Runes\n\n{runes.strip()}\n"
    match = re.search(r"^## Runes[^\n]*\n.*?(?=^#{1,2} |\Z)", sheet, re.M | re.S)
    if match:
        tail = sheet[match.end() :]
        sheet = sheet[: match.start()] + block + ("\n" + tail if tail else "")
    else:
        sheet = f"{sheet.rstrip()}\n\n{block}"
    vault.write_character_sheet(pc_slug, sheet)


def _template_values(
    vault: Vault, plan: Plan, pc_slug: str, *, appraisal: str = ""
) -> dict[str, str]:
    fm = frontmatter.parse(_read(vault, f"characters/{pc_slug}.md"))
    return {
        "name": str(fm.get("name") or pc_slug),
        "title": plan.title,
        "appraisal": appraisal,
        "aspect": plan.aspect,
        "aspect_description": plan.aspect_description,
        "aspect_ability": plan.aspect_ability,
        "flaw": plan.flaw,
        "attributes": ", ".join(plan.attributes),
    }


class _KeepMissing(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fill(template: str, values: dict[str, str]) -> str:
    return template.format_map(_KeepMissing(values)).strip()


def _plan_sources(vault: Vault, voice: Voice | None) -> str:
    parts = []
    for path in voice.plan_sources if voice else ():
        text = frontmatter.strip(_read(vault, path)).strip()
        if text:
            parts.append(f"--- {path} ---\n{text[:_SOURCE_MAX_CHARS]}")
    return "\n\n".join(parts) or "(none)"


def _campaign_lang(vault: Vault) -> str:
    lang = str(frontmatter.parse(_read(vault, "campaign.md")).get("lang") or "en")
    return lang if lang in ("en", "hu") else "en"


def _parse_json(raw: str) -> dict[str, Any] | None:
    from aidm.scene_sheet_writer import _parse_patch  # noqa: PLC0415 — the writer imports us

    return _parse_patch(raw) if raw else None


def _one_line(value: Any) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def _sections(body: str) -> dict[str, str]:
    """``## Heading`` → its text, up to the next level-1/2 heading."""
    out: dict[str, str] = {}
    for match in re.finditer(r"^## ([^\n]+)\n(.*?)(?=^#{1,2} |\Z)", body, re.M | re.S):
        out[match.group(1).strip()] = match.group(2).strip()
    return out


def _section_text(text: str, heading: str) -> str:
    return _sections(frontmatter.strip(text)).get(heading, "")


def _bullets(section: str) -> list[str]:
    return [
        line.strip()[2:].strip() for line in section.splitlines() if line.strip().startswith("- ")
    ]


def _read(vault: Vault, path: str) -> str:
    """Raw file text without the backlink footer; "" when missing."""
    try:
        text = vault.read_file(path)
    except VaultPathError:
        return ""
    marker = "\n\n# Referenced by\n"
    return text[: text.index(marker)] if marker in text else text
