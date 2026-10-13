"""Does a DM reply move the scene forward? A one-word verdict from the local model.

Content repetition — the same beat re-told in new words — is invisible to the
text-similarity guards: on 148 blind-labelled reply pairs (2026-09-27,
docs/bench/<date>/content-repeat) the replay guard caught 8% of it and an
embedding cosine 61% at 57% precision on unselected turns. This prompt on
qwen38 scored 86% accuracy, 92% precision, ~0.9 s. 38% of unselected turns
were repeats.

``judge_scene_ends`` is the sibling verdict that drives the story clock:
the DM almost never calls vault_set_scene, so the clock froze (playtest 3,
2026-09-27). On 155 blind-labelled turns (docs/bench/<date>/scene-end)
this departure/arrival prompt at P(YES) >= 0.2 caught 8 of 19 scene ends
with no false positive; the plain ENDS/CONTINUES wording never fired.
Precision over recall: a false tick is a forced time skip mid-scene.
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger

# English-only by decision (docs/decisions/english-only.md). Wording is the
# measured prompt — change it and re-run content-repeat/qwen_judge.py.
_PROMPT = (
    "Two consecutive replies from a game master in a text roleplaying game.\n\n"
    "PREVIOUS REPLY:\n{previous}\n\nPLAYER'S TURN:\n{player}\n\nCURRENT REPLY:\n{current}\n\n"
    "Does the CURRENT reply add meaningful new development - a new event, new information, "
    "a decision, the outcome of the player's action, or an NPC action that changes the "
    "situation? Or does it mostly re-narrate the same beat as the PREVIOUS reply (the same "
    "hesitation, standoff, description or action, even in different words)?\n"
    "Answer with exactly one word: ADVANCES or REPEATS."
)
_TIMEOUT_S = 10.0


async def judge_repeats(
    client: Any, model: str, *, previous: str, player: str, current: str
) -> bool | None:
    """True = ``current`` re-tells ``previous``; None = no verdict (error, empty, unparsed)."""
    if not previous.strip() or not current.strip():
        return None
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": _PROMPT.format(previous=previous, player=player, current=current),
                }
            ],
            max_tokens=4,
            temperature=0,
            # No enable_thinking kwarg: side calls stay on the gateway's
            # non-thinking default, like the sheet writer and the critic.
            timeout=_TIMEOUT_S,
        )
    except Exception as exc:  # fail-open: no verdict never blocks a turn
        logger.warning("advance_judge: FAILED — {}: {}", type(exc).__name__, exc)
        return None
    answer = (completion.choices[0].message.content or "").strip().upper()
    if answer.startswith("REPEAT"):
        return True
    if answer.startswith("ADVANC"):
        return False
    logger.info("advance_judge: unparsed answer {!r}", answer[:20])
    return None


_SCENE_END_PROMPT = (
    "A turn from a text roleplaying game.\n\nPLAYER'S TURN:\n{player}\n\n"
    "GAME MASTER'S REPLY:\n{current}\n\n"
    "In this turn, does the player's character leave, set off from, or arrive at a place "
    "(a room, building, bridge, camp, district - not a step or two within the same spot), "
    "or does the reply skip time forward (later, next morning, hours pass)?\n"
    "Answer YES or NO."
)
# Measured cut on P(YES) / (P(YES) + P(NO)); the model rarely says YES outright.
_SCENE_END_MIN_P = 0.2


async def judge_scene_ends(client: Any, model: str, *, player: str, current: str) -> bool | None:
    """True = the turn moved the scene to another place or time; None = no verdict."""
    if not current.strip():
        return None
    p_yes = await _p_yes(
        client,
        model,
        _SCENE_END_PROMPT.format(player=player, current=current),
        "scene_end_judge",
    )
    return None if p_yes is None else p_yes >= _SCENE_END_MIN_P


# First Nightmare (campaign feature ``first-nightmare``, aidm.first_nightmare).
# Unmeasured on labelled turns yet — the cuts are conservative guesses and
# every verdict logs its P(YES) for calibration. A false start freezes the
# clock and opens a trial the player never entered; a false end cuts it short.
_FALLS_ASLEEP_PROMPT = (
    "A turn from a text roleplaying game.\n\nPLAYER'S TURN:\n{player}\n\n"
    "GAME MASTER'S REPLY:\n{current}\n\n"
    "In this turn, does the player's character actually fall asleep or lose consciousness "
    "(not merely feel tired, lie down, or think about sleeping)?\n"
    "Answer YES or NO."
)
FALLS_ASLEEP_MIN_P = 0.5
_TRIAL_RESOLVED_PROMPT = (
    "A turn from a text roleplaying game. The player's character is inside a trial dream "
    "with one central conflict.\n\n"
    "The trial is WON when: {win}\nThe trial is LOST when: {lose}\n\n"
    "PLAYER'S TURN:\n{player}\n\nGAME MASTER'S REPLY:\n{current}\n\n"
    "In this turn, is the trial's central conflict finally resolved - has the win or the "
    "lose condition above actually happened (not merely approached, threatened, or partly "
    "achieved)?\nAnswer YES or NO."
)
TRIAL_RESOLVED_MIN_P = 0.5


async def judge_falls_asleep(client: Any, model: str, *, player: str, current: str) -> bool | None:
    """True = the PC falls asleep or passes out this turn; None = no verdict."""
    if not current.strip():
        return None
    p_yes = await _p_yes(
        client,
        model,
        _FALLS_ASLEEP_PROMPT.format(player=player, current=current),
        "nightmare_start_judge",
    )
    return None if p_yes is None else p_yes >= FALLS_ASLEEP_MIN_P


async def judge_trial_resolved(  # noqa: PLR0913 — keyword-only turn + plan inputs
    client: Any, model: str, *, win: str, lose: str, player: str, current: str
) -> bool | None:
    """True = the trial's win or lose condition happened this turn; None = no verdict."""
    if not current.strip():
        return None
    p_yes = await _p_yes(
        client,
        model,
        _TRIAL_RESOLVED_PROMPT.format(win=win, lose=lose, player=player, current=current),
        "nightmare_end_judge",
    )
    return None if p_yes is None else p_yes >= TRIAL_RESOLVED_MIN_P


async def _p_yes(client: Any, model: str, prompt: str, name: str) -> float | None:
    """P(YES) / (P(YES) + P(NO)) of a one-token answer; None on any failure (fail-open)."""
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=10,
            timeout=_TIMEOUT_S,
        )
        top = completion.choices[0].logprobs.content[0].top_logprobs
    except Exception as exc:  # fail-open: no verdict
        logger.warning("{}: FAILED — {}: {}", name, type(exc).__name__, exc)
        return None
    yes = sum(math.exp(t.logprob) for t in top if t.token.strip().upper().startswith("YES"))
    no = sum(math.exp(t.logprob) for t in top if t.token.strip().upper().startswith("NO"))
    if yes + no == 0:
        logger.info("{}: neither YES nor NO in the top tokens", name)
        return None
    p_yes = yes / (yes + no)
    logger.info("{}: P(YES)={:.3f}", name, p_yes)
    return p_yes
