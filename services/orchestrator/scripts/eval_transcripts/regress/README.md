# Regress scenarios — per-fix benchmarks

One JSONL transcript + companion `.expect.yaml` per landed DM-quality fix.
Each scenario is replayed N times by `scripts/bench_regress.py` against the
configured local LLM (private mode) and scored against the YAML's assertion
list. The summary is written under `docs/bench/<date>/regress/`.

| Scenario | Fix it covers | Assertion focus |
|---|---|---|
| `01-srd-skip` | `Vault.grep` skips `rules/srd/` subdirs by default | grep results contain no `rules/srd/` paths |
| `02-npc-pair-enforce` | empty-state hint + scene_enter result hint | `npc_update_state` within 2 turns of `npc_scene_enter` |
| `03-resume-no-relist` | resume-wrap directive + persona Continuity sections | no `vault_list_dir("sessions")` or session-log re-read on opening |
| `04-self-grep-prohibit` | persona "don't grep on phrases YOU wrote" rule | no `vault_grep` pattern is substring of prior assistant text |
| `05-stale-state-nudge` | `SessionNPCRegistry` inject-turn counter + stale marker | `npc_update_state` fires after a 5+ turn quiet stretch + beat shift |

Fix #1 (`AIDM_LOCALE` default `en`) is covered by a Python unit test, not a
transcript replay — it's a config-loading concern with no model behavior to
measure.

## Assertion kinds

- **deterministic** — a single replay failing fails the scenario. Used when
  the assertion is a fact about the harness/code path (e.g. SRD paths never
  reach the model because `Vault.grep` filters them).
- **behavioral** — pass rate must clear `scenario.pass_threshold` (default
  0.7). Used when the LLM's choice is involved.

## Fixtures

Scenarios that need extra vault content (existing session log, sheets) carry
a sibling `<scenario>.fixtures/` directory. The runner copies its contents
into the sandbox vault under the path named by `runner.inject_into` before
replay starts.
