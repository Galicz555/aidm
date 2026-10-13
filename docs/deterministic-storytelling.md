# Deterministic storytelling — how time, events, and goals actually work

The design rule for this layer is the same one the rest of the
orchestrator follows: **code detects, code decides, the model only
writes prose.** The DM-LLM narrates the world; it is never allowed to
decide what the world does. This doc says plainly which of the three
ingredients — time, events, goals — is actually machine-driven, and
which is not.

The short version:

| Ingredient | Machine-driven? | Where |
|---|---|---|
| **Time** | **Yes, fully.** | `runs/<slug>/clock.md` + `aidm/story_events.py` |
| **Events** | **Yes, fully** — but only *authored* events. | `runs/<slug>/events/*.md` + `aidm/story_events.py` |
| **Goals** | **No — not at runtime.** Goals steer authoring and the DM's prose; code never interprets them. | NPC sheets (`## Goals`), event chains authored from them |

## Time — code-owned, model-hinted

Every run that carries a `clock.md` has an in-fiction clock: a day
number and one of four segments (`morning / afternoon / evening /
night`). The model never sets it. It advances by at most one of two
deterministic signals per turn, checked in this order:

1. **A clamped hint from the sheet writer.** The end-of-turn PC patch
   (the same structured-JSON call that updates HP and location) may
   include `time_advance: next_segment | next_day` when the narration
   clearly moved time — "we camp for the night". Code validates the
   enum, clamps it to at most one day, strips it before anything
   touches disk. The model can *suggest* time passed; it cannot own the
   value, and it cannot move time backward. Hints are also rate-limited
   to one honored hint per `turns_per_segment` turns (`last_hint_turn`
   in `clock.md`) — an eager writer once pushed a 50-turn run to day 12
   against day 7. A suppressed hint is simply dropped.
2. **A scene change.** A `vault_set_scene` tool call, or the scene-end
   judge (`advance_judge.judge_scene_ends`, private mode, runs beside the
   sheet patches) saying the turn left/arrived somewhere or skipped time,
   ticks one segment. The DM almost never calls `vault_set_scene`, so
   without the judge the clock froze (playtest 3). The judge is tuned for
   precision — 8/19 scene ends caught, 0 false positives on 155 labelled
   turns (`docs/bench/<date>/scene-end/`). Scene-change ticks are
   rate-limited to one per `turns_per_segment` turns since the last
   advance: gate → hall → yard is one evening, not three.

There is **no baseline tick** (removed 2026-09-27): a quiet-turns tick
advanced the clock — and printed the new time — in the middle of scenes
still playing out. Time never jumps before a scene ends; the price is
that a run that never changes scene and never narrates time passing
keeps its clock (and its time-triggered events) where they are.

**New runs** created from the UI copy the campaign's
`runs/_template/` (`clock.md` + `events/`), so the engine is on from the
first turn. `runs/` is invisible to the DM's vault tools, so the
template leaks no plot. An arrival event can carry
`if_subject_present: satisfied`: when its subject is already with the player
it is recorded as fired (`narrated: pre-empted`) without re-staging it. The campaign template carries the eval-30t events
(Lyra at the gate … the Ash Circle raid), starting on day 1, evening.

The DM sees time as a `[Time] Day 2, evening.` line at the top of the
**scene-sheets block** every turn — the channel the 14B obeys (it used
to ride the turn contract, which it ignored; the 2026-09-25 marathon
collapsed days 1–7 into one continuous night). The passage of time is
**shown by code**, not coaxed from the model: after every advance, the
first bot reply opens with a code-written line — *"\*Day 3, morning.\*"*
— that reaches the transcript and the model's own context (so the DM
sees that time moved). An earlier design asked the DM to narrate each
transition and held the clock until it did; unguarded, the local 14B
narrated 1 of ~15, and the hold only slowed the clock and starved
authored events, so both the directive and the hold are gone. Fired event directives
carry their in-fiction time — *"(Day 5, night) …"* — so the DM stages
them instead of pasting them into the current moment. The clock file
itself is hidden from the DM's vault tools (see spoiler guard below).

## Events — the plot is authored files, fired by code

The campaign's plot does not live in the model. It lives in small
markdown files under `runs/<slug>/events/`, written at world-building
time. Each one carries:

- **a trigger**: a day/segment plus optional state conditions
  (subject NPC's `status` / `location`, the PC's location) read fresh
  from the `.state.md` sheets. Locations match by containment — every
  content word of the authored location must appear in the writer's
  free-text one ("ridgeway fallout shelter" matches "hall of ridgeway fallout
  shelter"); `dead` and `deceased` are the same status; an event with no
  status requirement only needs its subject not to be gone;
- **a directive**: prose describing *what has already happened*, which
  the DM must weave into narration;
- **mutations** (`## On fire`): deterministic world changes — NPC state
  fields, beat appends, notes — applied through the normal vault write
  paths.

Every turn, *after* the sheet writer has landed the turn's patches and
auto-exits, the engine evaluates pending events against the clock and
the fresh sheets. This ordering is the whole player-agency story: by
the time a trigger is checked, the player's consequences are already
on disk. Kill the scout → his chain cancels. Escort the medic → the
"she went alone" step finds her location wrong and expires. Let the
stranger in → the "she gives up and leaves" event dies on its
condition. **No LLM judgment anywhere in that pipeline** — deferral,
expiry, cancellation, and branch exclusivity (two mutually exclusive
day-5 raids, keyed on whether the scout is alive) are all plain code
rules over sheet state.

Fired directives ride the **scene-sheets block** (max two per turn;
overflow persists on disk and carries forward, restart-safe), with a
binding pointer in the `<!--aidm:turn-contract-->` note — the first
full 30-turn run proved the prompt-tail channel gets ignored while the
sheets block gets obeyed. The wording is deliberate: *"these facts have
already happened … they never replace or veto the player's declared
action."* And delivery is verified, not assumed: a rendered directive
stays live and re-renders every turn until code confirms the DM's reply
actually contains it (content-word overlap), or the render cap (3)
retires it as `narrated: abandoned` — a loud, on-disk failure the
scorer and judge both read. The model narrates the world's motion; it
cannot veto it, reorder it, or quietly forget it.

A second enforcement layer runs inside the local LLM provider
(`providers/llm_local.py`): the reply is **buffered before the player
sees it** and regenerated once if it misses an active **event** directive
(time-transition directives are not guarded here — they rely on the
end-of-turn confirm/hold/abandon; guarding them cost a failed retry on
most transition turns). The retry note names the uncovered events
explicitly and is appended to the context then removed in `finally` — it
never persists into history. Cross-reply prose tics (4-gram shingles
recurring in 3+ of the last 10 replies) and verbatim paragraph loops
(8-gram shingles recurring across 2+ of the last 10 replies) are **not
regenerated — they are excised**: the draft's sentences containing them
are dropped (quote-safe splitting), tiered so the cut stays ≤ 40% (all
repeated 8-grams first, else only the director-flagged phrases), with a
single retry only if even that would cut more. The retry mostly failed
for these (18 of 30 tic drafts, 8 of 13 loop drafts still violated) at
~10 s each. Whatever a retry releases is excised the same way.
**Whole-reply replays** still regenerate: a draft ≥ 0.8 similar
(`turn_director.similarity`, difflib without autojunk) to one of the
last two replies fails, and the retry is made genuinely different — every copied prior reply is hidden from that
one call (restored in `finally`), the retry note quotes the player's
latest message ("respond to THIS"), and temperature rises by 0.25. If a
retry triggered by a lesser guard (banned phrase, missed directive)
comes back as a replay while the draft was fresh, the draft is released
instead — a whole repeated reply is the worse failure. Replaying the
2026-09-26 marathon's 15 looping turns with live guards
(`scripts/eval_replay_guard.py --live-guards`): every replay-guard
retry cleared (0.98–1.00 → 0.18–0.32); the only replays still released
came from lesser-guard retries (3 of 15), which the draft fallback now
catches. Every guard the draft failed contributes a
section to the single retry note — still max one retry (two
completions) per turn.

These four guards were tuned against Qwen3-14B's failure modes, so each
has a per-model switch (`AIDM_GUARD_DIRECTIVES` / `_REPLAY` / `_TICS` /
`_PARAGRAPH`, default on; set by the model profiles in
`services/llm-gateway/profiles/`). Switching off the directive guard
drops only the release retry — the directive is still injected, still
confirmed at end of turn, and still counted as a miss by
`eval_replay_guard.py --summary-json`.

**Draft→critic pass** (`AIDM_CRITIC_PASS_ENABLED`, default off): on
turns with 2+ NPC sheets in scene and no other guard failure, one
extra short completion checks the draft against the sheets actually
spliced into context this turn for three kinds of problem in a single
call: (1) sheet contradiction or invented facts (a relationship, an
event, a possession with no basis in the sheets), (2) self-contradiction
inside the draft itself (a location/time/object stated one way then
another — no other guard compares a reply against itself, they're all
cross-turn), (3) voice drift against a sheet's `speech_register`. A hit
folds into the same regenerate-once note as the other guards —
selective and bounded to at most one extra completion pair, never
stacked on a guard-triggered retry. Fails open: a critic timeout or
bad JSON is logged and skipped, never blocks the turn.

**Directive logit bias** (`AIDM_DIRECTIVE_LOGIT_BIAS_ENABLED`, default
off): while a directive is live, a soft positive bias
(`AIDM_DIRECTIVE_LOGIT_BIAS_STRENGTH`, default 6) nudges generation
toward the directive's proper-noun anchors (e.g. "Marta") at the
sampling layer, via llama.cpp's `logit_bias` on the first leading-space
token of each name (one `/tokenize` round-trip per anchor, milliseconds
against the turn's 10-30s completion). Deliberately NOT a grammar or
`tool_choice` constraint — that's what caused the same-day agentic-recon
runaway (a forced completion that can't converge just rambles to the
token ceiling); a logit bias only adds to the logits, it never forces
or excludes a token. Fails open: a tokenize error drops that one word,
never the turn.

**Spoiler guard:** `events/` and `clock.md` are invisible to the DM's
vault tools (`list_dir` / `read_file` / `grep` / `glob` / the vault
map). A DM that can read day 5's raid file will foreshadow it to
death. Only the engine's dedicated accessors reach these files.

## Campaign features — code-owned set pieces (First Nightmare)

A campaign can opt into a mechanic whose *structure* code owns, with
`features: [...]` in `campaign.md`. The first is `first-nightmare`
(`aidm/first_nightmare.py`, private mode; no shipped campaign enables it). A playtest
left the trial to the DM: no Spell greeting, no runes, no conflict, no
appraisal, vague dream prose, "over" after six turns — and a
`*Day 1, afternoon.*` inside the dream, because the scene-end judge read
falling asleep as leaving a place.

- **Start** — a logprob YES/NO judge beside the scene-end judge ("does
  the character fall asleep or lose consciousness?", P(YES) ≥ 0.5, every
  P logged) runs until a trial exists, never in session zero. On YES one
  structured-JSON call writes the plan to `runs/<run>/arcs/first-nightmare.md`
  (setting in the Dream Realm's past, role, one central conflict, 3
  stages, win/lose, Dormant Aspect + description, ability, Flaw, 1–3
  bracketed Attributes), validated all-or-nothing and rejected if it names
  anything on the campaign's never-reuse list. A failed
  plan leaves the arc `dormant` and is retried next turn.
- **During** — the plan rides the scene-sheets block every turn (runes
  marked "reveal only on a rune check; ability and Flaw at the end"); the
  clock is **frozen**: no scene-change tick, no writer hint, no event
  fires or renders, no `[Time]` line.
- **End** — from the 3rd trial turn a second judge, shown the win and
  lose conditions, asks whether the conflict resolved. On YES a small
  JSON call picks `outcome` and a one-word appraisal from the campaign's
  list; a survival writes a `## Runes` section into the PC identity sheet.
  A death is logged, never enforced — the DM narrates it.
- **The Spell's words are code-written**, like `*Day N*`: the greeting
  opens the first reply after the plan lands, the appraisal block the
  reply after the end — rendered from the campaign's `rules/spell-voice.md`,
  never from Python strings.

Every step writes through `Vault`, so the text regenerate journal rolls
a start or end back; the pending Spell line lives on `DirectorState`,
which the regenerate snapshot restores. The judges' first probe is in
`docs/bench/<date>/first-nightmare/`.

## Goals — load-bearing for authoring, invisible to the engine

Honesty section. **The runtime does not interpret goals.** There is no
code that reads an NPC's `## Goals` section, reasons about it, or
generates events from it. What goals actually do, in decreasing order
of machinery:

1. **They are the authoring source of event chains.** Every NPC sheet
   carries a goal with an *object, an obstacle, and a deadline* (Marta:
   antibiotics for Tomi / horde ground / day 4). The event chains under
   `events/` are those goals hand-compiled into steps: if the player
   never engages, the NPC pursues the goal off-screen and the world
   changes anyway. The goal → event translation happens at world-
   building time, by a human (or an authoring agent) — not at runtime.
   Runtime generation of events from goals is the explicitly flagged
   upgrade path, not something the code does today.
2. **The per-turn `scene_goal` is measured, not understood.** The
   sheet writer records each in-scene NPC's momentary `scene_goal`; the
   turn director only checks whether it has been *textually static for
   3+ turns* (difflib similarity) and, if so, injects a "change tack"
   note. That is stagnation detection, not goal reasoning.
3. **Goals steer the DM as prose.** The `## Goals` section rides into
   the prompt with the rest of the identity sheet every turn, so the
   model plays the NPC as someone who wants something. That is prompt
   content, not mechanism.

So: **time and events are deterministic machinery; goals are the
human-legible spec those machines were built from.** If someone asks
"does the engine know Marta wants antibiotics?" — no. It knows an
event file says her location changes to `pharmacy district` on day 3
morning if she's still at the shelter, and that everything in her
chain dies with her. The *why* lives on her sheet and in the DM's
prose.

## What is deliberately not here

- No runtime event generation from goals (flagged upgrade path).
- No LLM-in-the-loop for triggers, cancellation, or scheduling.
- No model-emitted state syntax of any kind — the retired inline
  `[state:]` machinery is the cautionary tale.
- No clock or plot visibility for the DM beyond the `[Time]` line,
  time-transition directives, and fired directives.

## Running the tests

Everything below is plain pytest — no GPU, no network, no API keys.

```bash
# the whole gate (ruff + mypy --strict + all tests + vault lint + web build)
make check

# just the python tests
make test

# the storytelling layer, targeted:
uv run pytest services/orchestrator/tests/test_story_events.py -q          # clock rules + trigger/defer/cancel/branch matrix
uv run pytest services/orchestrator/tests/test_vault_story_namespace.py -q # spoiler guard incl. path-traversal bypass shapes
uv run pytest services/orchestrator/tests/test_lint_vault_events.py -q     # event/clock schema + mutation grammar lint
uv run pytest services/orchestrator/tests/test_turn_director.py -q         # world-update pointer (clock line now in the sheets block) + phrase-tic + para-shingle
uv run pytest services/orchestrator/tests/test_scene_sheet_writer.py -q    # time_advance extraction, engine hook, stale-directive clearing
uv run pytest services/orchestrator/tests/test_llm_local_guard.py -q       # guarded-completion retry paths (phrase tic, contrast tic, para loop, directive miss, critic, logit bias)
uv run pytest services/orchestrator/tests/test_first_nightmare.py -q       # judges' cut + fail-open, plan, start/end, frozen clock, Spell lines, rollback

# the eval tooling around it:
uv run pytest services/orchestrator/tests/test_loop_sim.py -q    # harness pure functions (scenario, beats, anti-dup, SSE parse)
uv run pytest services/orchestrator/tests/test_loop_score.py -q  # deterministic scorer buckets
uv run pytest services/orchestrator/tests/test_judge_run.py -q   # judge schema + median math
```

The *live* closed-loop runs are not pytest — they need the llama.cpp
gateway and `ANTHROPIC_API_KEY`:

```bash
docker compose --profile private up -d llm-gateway
ANTHROPIC_API_KEY=… make loop-sim-smoke              # 5 turns, ~10-15 min
make judge-run REPLAY=docs/bench/<date>/loop/smoke-5t/replay-001
```

Vault content (the authored events, clocks, sheets) is validated by
`make vault-lint`, which runs inside `make check`.
