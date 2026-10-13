# vault — campaign knowledge bases

This directory is the **vault root** — it holds one self-contained
Obsidian vault per campaign. The orchestrator enumerates every campaign
at runtime; the web picker switches between them without a restart.
`VAULT_PATH` only names the *default* campaign (for the local CLI and
the initial pick in the browser when localStorage is empty); a separate
`AIDM_VAULT_ROOT` env var can point the discovery scan elsewhere when
the campaigns don't live next to the default.

## Layout

```
vault/                         (the vault root — AIDM_VAULT_ROOT, default ./vault/)
├── README.md                  (this file)
└── <campaign-slug>/           (one directory per campaign — a full Obsidian vault)
    ├── campaign.md            (campaign metadata: title, status, tone, lang, player_guide?)
    ├── npcs/                  world-canon: non-player characters
    ├── locations/             world-canon: places (subfolder by region if useful)
    ├── factions/              world-canon: organized groups
    ├── items/                 world-canon: notable objects (magic items, relics)
    ├── lore/                  world-canon: history, religions, magic system
    ├── rules/                 world-canon: house rules, mechanic clarifications
    └── runs/                  (one subdirectory per playthrough)
        └── <run-slug>/
            ├── run.md         run metadata: title, status, created, parent_campaign
            ├── characters/    active PCs and party members
            ├── arcs/          ongoing storylines
            ├── notes/         DM-authored discoveries (vault_append_note target)
            ├── discarded/     DM replies the player regenerated away (hidden from the DM, not linted)
            └── sessions/      per-session logs, timestamped, append-only
```

**World-canon vs run-scoped.** The split is the point of the hierarchy.
NPCs, lore, locations, factions, items and rules describe the world
itself — they're shared across every playthrough of the same campaign,
so two parties exploring the same world see the same Iron Veil
Citadel. Characters, arcs, sessions and notes describe *this party's*
journey — they evolve per run, so replaying the campaign with a new
party doesn't disturb the old timeline.

**The DM sees a flat view.** Run-scoped paths route through an overlay:
`vault_read_file("characters/lyra.md")` resolves to
`runs/<active-run>/characters/lyra.md` automatically. The persona prompt
doesn't have to know about runs.

## Rulebook content under `rules/`

`rules/` is the only world-canon folder allowed to be **multi-layered**:

```
<campaign>/rules/
├── *.md              campaign layer — house rules + the system being
│                     played (e.g. OSR mechanics, hand-authored or
│                     PDF-converted)
└── <system>/         base-system layer — the ruleset the campaign
    ├── *.md          builds on (e.g. `srd/` for D&D 5e SRD 5.2)
    └── index.md
```

On conflict the campaign layer wins (the persona prompt teaches the
DM this precedence). Subdir files are typically large chapter dumps
from a rulebook PDF, navigated by `vault_grep`, never read in full.

Source PDFs / DOCX of rulebooks live **outside the vault**, under
`rulebooks/<campaign>/`. They're gitignored (usually copyrighted) and
never seen by the DM-LLM. `scripts/import_rulebook.py` converts them
into markdown under `rules/` or `rules/<system>/` (run it with
`--help` for the options).

## Karpathy-RAG, not vector DB

The AI DM has no embedding pipeline. It navigates the vault with
filesystem tools (`vault_list_dir`, `vault_read_file`, `vault_grep`,
`vault_glob`). For this to work the **layout must be predictable** —
the persona prompt teaches the DM exactly the folder names above. Don't
reorganize without updating the prompt files in
`services/orchestrator/prompts/`.

## Frontmatter schema

- **Keys and enum values are English** regardless of vault language.
  Free-text values (descriptions, names, prose) follow the file's
  `lang:` tag.
- The single source of truth is
  `services/orchestrator/aidm/vault_schema.py`. Lint
  (`services/orchestrator/scripts/lint_vault.py`, wired into
  `make check`) enforces it.
- `character` and `npc` files require a HEXACO psychology block (six
  1-5 scores in frontmatter + one-line gloss per dimension in the
  body). Without it the DM has no consistent psychological anchor
  across sessions.
- **NPC voice & color (optional, opt-in per character):**
  - `voice_id:` — ElevenLabs library voice ID used when this NPC
    speaks. Falls back to the session default DM voice if absent.
  - `speech_register:` — free-text description of how they talk
    (clipped, formal, profession-specific jargon, regional dialect).
    Fed to the DM via the NPC file; helps shape dialogue tone.
  - `physical_state:` — free-text body condition that affects how
    they speak today (arm in sling, drunk, hoarse from shouting).
    Distinct from `status:` which is the world-level flag
    (active/wounded/dead/missing).
- **NPC/PC volatile state lives in a `.state.md` sibling file.**
  Each `npcs/<slug>.md` / `characters/<slug>.md` identity sheet has a
  `<slug>.state.md` sibling (`type: npc_state` / `character_state`)
  carrying mood, scene goal, social mask, location, `in_scene`,
  `## Recent (last 5 events)` and `## Session so far`. The post-turn
  sheet writer patches it every turn; the identity sheet only changes
  on real canon shifts (or stub consolidation). `in_scene: true` is
  how the next session restores who was present — no session-log
  flush needed. `npc_scene_enter` / `npc_scene_exit` toggle scene
  membership.
- **Relationships and knowledge** that evolve across sessions belong
  in the NPC body as `## Relationships` and `## Knowledge` sections —
  one disk hit reads them with the rest of the NPC's profile.

## File conventions

- **One subject per file.** Don't combine NPCs, locations, or events.
- **Wikilinks** (`[[elara]]`) over plain text references. The
  orchestrator's `vault_read_file` appends a `# Referenced by` footer
  listing other files that link to the one being read.
- **Aliases** in frontmatter for nicknames and alternate spellings, so
  `vault_grep` finds the file regardless of how the player said the
  name.
- **Auto-fill.** Server-side writes set `last_updated:` automatically;
  don't trust LLM-supplied dates.

## Session lifecycle

1. **Session start**: the web picker (or `RUN_SLUG` env for the local
   CLI) chooses a campaign + run. The DM lists `sessions/` and reads
   the most recent log under the active run to recover continuity.
   Reads `campaign.md` for top-level world state. Lists `rules/`
   (small files read eagerly). The orchestrator restores the PC slug
   and re-enters every NPC whose `.state.md` says `in_scene: true` —
   no DM tool calls needed for that.
2. **During play**: chat history lives in the LLM context window;
   per-turn NPC/PC state lives on the `.state.md` sheets, patched by
   the post-turn writer. `vault_update_character`, `vault_set_scene`,
   `vault_set_character`, `vault_append_note` write the durable bits —
   all routed under the active run via the overlay, so the world canon
   stays untouched.
3. **Session end**: `vault_write_session_log` drops a 200-400 word
   narrative summary into `runs/<run-slug>/sessions/YYYY-MM-DD-HHMMSS.md`.
   Filenames are timestamped so logs never overwrite. Humans can edit
   them afterward.

Run-scoped subtrees are first-class vault content — committed alongside
the canon. They are NOT gitignored.

## Authoring outside the orchestrator

Open `vault/<campaign-slug>/` directly in Obsidian. The orchestrator and
Obsidian share the same files; concurrent edits are last-write-wins (no
locking). For run-specific edits (a character sheet, a session log)
navigate into the relevant `runs/<run-slug>/` subdirectory.

The browser client exposes an edit form for the **active character
only** as a convenience during play; for everything else, use Obsidian.

## Adding a new campaign

1. `mkdir vault/<new-campaign-slug>/`
2. Create `vault/<new-campaign-slug>/campaign.md` with the required
   frontmatter (`type: campaign`, `title`, `status: active`, `lang`).
   Optionally add `player_guide: rules/<filename>.md` to point the
   session-zero DM at the system's character-creation guide — without it
   the DM falls back to D&D 5e defaults.
3. The web picker enumerates the vault root on every load, so the new
   campaign appears in the dropdown after a refresh. The first run is
   created from the UI via the "+ Run" button.

## Adding a new run to an existing campaign

The web picker handles this end-to-end via `POST /api/campaigns/{slug}/runs`.
For scripted creation, drop a `run.md` under
`runs/<new-run-slug>/` with frontmatter `type: run`, `title`,
`status: active`, `created: YYYY-MM-DD`, `parent_campaign: <slug>`, and
create the four standard subdirectories (`characters/`, `arcs/`,
`sessions/`, `notes/`).
