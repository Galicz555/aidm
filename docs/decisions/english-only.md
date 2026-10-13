# Decision: English-only for new development

- **Date:** 2026-07-11
- **Status:** accepted

## Decision

All new AIDM work — event engine, world content, prompts, tests, bench
tooling — is authored in English only. Campaign vaults are
`lang: en` (`campaign.md`).

## What this does NOT mean

Hungarian support is **frozen, not removed**. Removal is invasive and buys
nothing:

- Existing locale dicts stay as they are (`turn_director.py`,
  `scene_sheet_writer.py`, `scene_sheet_injector.py`, `player_suggest.py`,
  `vault_tools.py`, `prompts/dm.hu*.md`).
- Existing `hu` tests keep running and must keep passing.
- The `AIDM_LOCALE=hu` runtime path remains functional; it just stops
  receiving new strings.

## Rules for new code and content

1. New user-facing strings are plain English constants — do **not** add a
   `{"en": ..., "hu": ...}` dict for new features.
2. New vault content (campaign canon, eval runs, events) is `lang: en`.
3. New prompt files get no `.hu.md` sibling.
4. Do not delete or "clean up" existing `hu` plumbing while touching
   adjacent code; freezing means leaving it alone.

## Why

The closed-loop eval + judge stack (2026-07) measures English output; a
bilingual content matrix doubles authoring and calibration cost for a
locale not under active development. Revisit if Hungarian play resumes.
