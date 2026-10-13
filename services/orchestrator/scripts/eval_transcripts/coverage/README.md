# Coverage corpus — broader DM-quality probe

Sister to `regress/`. Where the regress suite pins specific landed
fixes against degradation, the coverage suite probes the DM's behavior
across orthogonal scene shapes — the failure-mode taxonomy distilled
from the 2026-06-25 test-session analysis.

Driver: same `services/orchestrator/scripts/bench_regress.py` discovers
both `regress/*.expect.yaml` and `coverage/*.expect.yaml`. Filter to
one or the other with `--scenarios <substring>`.

## Taxonomy — 10 categories

| Category | Probes | Phase 2a status |
|---|---|---|
| `01-resume-continuity` | rich resume log → opening references loaded canon | included |
| `02-npc-lifecycle` | first-time NPC naming triggers full triad | regress/02 covers it |
| `03-stale-state` | long-quiet-stretch with failsafe | regress/05+05b cover it |
| `04-multi-npc` | two NPCs active, both maintained separately | included |
| `05-vault-grounding` | player names canon → grep/read before narration | regress/01 covers near-cases |
| `06-self-grep` | extended narrative with self-grep temptation | regress/04 covers it |
| `07-dice` | `[ROLL d20+N = M]` markers interpreted by intensity range | included |
| `08-pure-dialogue` | "I say:" turns trigger zero tool calls | TODO |
| `09-scene-change` | named location → `vault_set_scene` trigger | TODO |
| `10-session-end` | "we're done" → `vault_write_session_log` with 200+ words | included |

Phase 2a ships **4 scenarios** — the ones orthogonal to existing
regress coverage. The other six either overlap heavily with regress
scenarios already in the suite, or need a different transcript shape
that's worth its own design pass (08, 09).

## Per-scenario expectations

Same YAML shape as `regress/`. Pass thresholds across coverage default
to **0.6** — coverage is meant to surface behavioral signal across many
shapes, not gate merges. A drop from 90% to 50% on a coverage scenario
is a "look here" pointer; a drop on a regress scenario is "stop the
ship". Coverage scenarios with deterministic assertions (e.g.
scene-change must emit `vault_set_scene`) use the per-assertion
`kind: deterministic` to override the soft default.

## Running the corpus

```
make bench-regress                              # runs both regress + coverage
uv run python services/orchestrator/scripts/bench_regress.py \
    --scenarios coverage                        # only the coverage suite
```

The cost is meaningful at 10 scenarios × 10 replays ≈ 1.5-2 hours
overnight on Qwen3-14B private mode. For quicker smoke,
`--replays 3` collapses it to ~10-15 minutes.
