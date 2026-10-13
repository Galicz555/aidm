"""Per-session scene presence — who is currently in scene.

The full NPC/PC state lives on disk as running sheets (HEXACO, status,
running memory). The only thing this in-process registry tracks is
*who's in the scene right now*, so the per-turn sheet injector knows
which sheets to read and the post-turn writer knows which sheets to
update.

Replaces the previous ``SessionNPCRegistry`` which held volatile
mood/scene_goal/social_mask in memory — that signal now lives on the
sheet itself, rewritten each turn by the sheet writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words that carry no place identity in the writer's free-text locations
# ("the gate, her side of the wire", "approaching the shelter").
_PLACE_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "at",
        "on",
        "near",
        "by",
        "to",
        "from",
        "toward",
        "towards",
        "into",
        "inside",
        "outside",
        "with",
        "her",
        "his",
        "their",
        "its",
        "s",
        "now",
        "just",
        "still",
        "side",
        "edge",
        "area",
        "entrance",
        "exit",
        "door",
        "approaching",
        "leading",
        "standing",
        "facing",
    ]
)


def place_words(location: str) -> frozenset[str]:
    """Content words of a free-text location, casefolded."""
    words = re.findall(r"\w+", location.casefold())
    return frozenset(w for w in words if w not in _PLACE_STOPWORDS and not w.isdigit())


def same_place(a: str, b: str) -> bool:
    """Lenient: two locations are the same place if they share a content word.

    Simplification: word overlap, not a place hierarchy — "infirmary" vs "ridgeway fallout
    shelter" still reads as different. Canonical location slugs if that bites.
    """
    return bool(place_words(a) & place_words(b))


def place_matches(required: str, actual: str) -> bool:
    """Strict: every content word of an authored location appears in the actual one."""
    need = place_words(required)
    return bool(need) and need <= place_words(actual)


# Words that legitimately follow "I'm" / "I am" capitalized (sentence
# rhythm, emphasis) and are never names. Casefolded before comparison.
NAME_STOPWORDS = frozenset(
    {
        "afraid",
        "done",
        "fine",
        "glad",
        "good",
        "here",
        "just",
        "listening",
        "no",
        "not",
        "nothing",
        "okay",
        "serious",
        "sorry",
        "sure",
        "the",
        "yes",
    }
)


def npc_named(text: str, name: str, slug: str) -> bool:
    """``text`` names this NPC: full name, first name, or slug head.

    Shared by the writer's auto-exit guard and the story engine's
    subject-named hold. ``name`` is the identity sheet's ``name:``
    (empty falls back to the slug).
    """
    name = name or slug
    keys = {name, name.split(maxsplit=1)[0], slug.split("-", 1)[0]}
    return any(
        re.search(rf"\b{re.escape(k)}\b", text, re.IGNORECASE)
        for k in keys
        if len(k) >= 3 and k.casefold() not in NAME_STOPWORDS
    )


@dataclass
class SceneRegistry:
    """Mutable per-session container: which slugs are in the active scene."""

    in_scene: set[str] = field(default_factory=set)
    current_pc_slug: str | None = None
    # Count of ``vault_set_scene`` calls since the last end-of-turn
    # engine pass. The story clock (aidm.story_events) consumes this as
    # one of its advancement inputs.
    scene_changes: int = 0

    def note_scene_change(self) -> None:
        self.scene_changes += 1

    def consume_scene_changes(self) -> int:
        """Return and reset the scene-change count for this turn."""
        count = self.scene_changes
        self.scene_changes = 0
        return count

    def enter(self, slug: str) -> None:
        clean = slug.strip()
        if clean:
            self.in_scene.add(clean)

    def exit(self, slug: str) -> None:
        self.in_scene.discard(slug.strip())

    def active(self) -> list[str]:
        """Sorted snapshot of NPC slugs currently in scene."""
        return sorted(self.in_scene)

    def set_pc(self, slug: str | None) -> None:
        self.current_pc_slug = slug.strip() if slug else None
