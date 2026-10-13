"""Turn director: detectors, thresholds, note assembly."""

from __future__ import annotations

from aidm.turn_director import (
    NEAR_DUPLICATE_THRESHOLD,
    PHRASE_WINDOW_REPLIES,
    REGISTER_REMINDER_AFTER,
    REPLY_SIMILARITY_THRESHOLD,
    STALL_ESCALATE_AFTER,
    TURN_CONTRACT_TAG,
    DirectorState,
    build_notes,
    player_passes,
    similarity,
)

_REPLY_A = (
    "Lira's breath catches in her throat as you pull her closer, your lips "
    "pressing against hers with deliberate intensity. She swallows hard, "
    "still holding onto your shoulders for a moment longer."
)
_REPLY_B = (
    "Lira's breath catches in her throat as you pull her closer, your lips "
    "pressing against hers with deliberate intensity. She swallows hard, "
    "still holding onto your shoulders for just a moment longer before "
    "stepping back slightly."
)
_REPLY_FRESH = (
    "The generator coughs twice and dies. In the sudden dark, Lira's hand "
    "finds your wrist — not tender this time, urgent. Something is scraping "
    "against the outer door."
)


def test_similarity_bounds() -> None:
    assert similarity("abc", "abc") == 1.0
    assert similarity("", "anything") == 0.0
    assert similarity("  spaced   out  TEXT ", "spaced out text") == 1.0


def test_similarity_scores_long_near_replays_high() -> None:
    # difflib's autojunk (on for inputs >= 200 chars) scored real ~0.9
    # replays at ~0.4-0.6; long texts differing in one clause must stay high.
    base = " ".join(
        f"Kerek's jaw tightens as you head for door {n}, his eyes flicking between you and the bench."
        for n in range(8)
    )
    assert len(base) > 600
    assert similarity(base, base.replace("door 3", "gate 3")) > 0.95


def test_observe_turn_flags_verbatim_loop() -> None:
    state = DirectorState()
    state.observe_turn(
        user_text="I kiss her",
        prev_user_text="I say hi",
        assistant_text=_REPLY_B,
        prev_assistant_text=_REPLY_A,
    )
    assert state.reply_similarity > REPLY_SIMILARITY_THRESHOLD
    assert not state.user_repeated


def test_observe_turn_fresh_reply_stays_below_threshold() -> None:
    state = DirectorState()
    state.observe_turn(
        user_text="I look around",
        prev_user_text="I wait",
        assistant_text=_REPLY_FRESH,
        prev_assistant_text=_REPLY_A,
    )
    assert state.reply_similarity < REPLY_SIMILARITY_THRESHOLD


def test_observe_turn_detects_repeated_user_input() -> None:
    state = DirectorState()
    state.observe_turn(
        user_text="I take out my map ",
        prev_user_text="i take out  my map",
        assistant_text="x",
        prev_assistant_text="",
    )
    assert state.user_repeated
    assert state.reply_similarity == 0.0  # no previous reply → no loop signal


def test_goal_streak_flags_after_three_static_turns() -> None:
    state = DirectorState()
    state.observe_goal("lira", "extract information about the shelter")
    assert state.static_goal_slugs == []
    state.observe_goal("lira", "extract information about the shelter")
    assert state.static_goal_slugs == []
    state.observe_goal("lira", "extract information about the shelter")
    assert state.static_goal_slugs == ["lira"]
    # A genuinely new goal resets the streak.
    state.observe_goal("lira", "get the player to open the armory")
    assert state.static_goal_slugs == []


def test_goal_streak_ignores_cosmetic_rewording() -> None:
    state = DirectorState()
    state.observe_goal("lira", "navigate the emotional aftermath of escalating intimacy")
    state.observe_goal("lira", "navigate the emotional aftermath of the escalating intimacy")
    state.observe_goal("lira", "navigate the emotional aftermath of escalating intimacy")
    assert state.static_goal_slugs == ["lira"]
    assert (
        similarity(
            "navigate the emotional aftermath of escalating intimacy",
            "navigate the emotional aftermath of the escalating intimacy",
        )
        > NEAR_DUPLICATE_THRESHOLD
    )


def test_prune_drops_exited_npcs() -> None:
    state = DirectorState()
    for _ in range(3):
        state.observe_goal("lira", "same goal forever")
    assert state.static_goal_slugs == ["lira"]
    state.prune(keep=[])
    assert state.static_goal_slugs == []
    # Re-entering starts a fresh streak, not a resumed one.
    state.observe_goal("lira", "same goal forever")
    assert state.static_goal_slugs == []


def test_build_notes_always_carries_agency_contract() -> None:
    payload = build_notes(DirectorState(), "en")
    assert payload.startswith(TURN_CONTRACT_TAG)
    assert "[Turn plan — non-negotiable this turn]" in payload
    assert "it is canon" in payload
    # No triggered signals → Must with exactly the agency bullet, no other
    # section, no Scene line (empty roster).
    assert "Must:" in payload
    assert "Must not:" not in payload
    assert "Voice:" not in payload
    assert "Scene:" not in payload
    assert payload.count("- ") == 1


def test_build_notes_buckets_notes_into_sections() -> None:
    state = DirectorState()
    state.user_repeated = True  # Must
    state.player_echo = True  # Must not
    state.reply_similarity = 0.9  # Must not
    state.observe_scene_tenure("marta", "clipped, wry")
    for _ in range(REGISTER_REMINDER_AFTER):
        state.observe_scene_tenure("marta", "clipped, wry")  # Voice
    payload = build_notes(state, "en")
    must, _, rest = payload.partition("Must not:")
    must_not, _, voice = rest.partition("Voice:")
    assert "repeated their previous input" in must
    assert "Do not repeat or restate" in must_not
    assert "nearly identical" in must_not
    assert "Voice check — marta" in voice


def test_build_notes_scene_roster_line() -> None:
    state = DirectorState()
    assert "Scene: lyra, marta + pc rowan" in build_notes(
        state, "en", npc_slugs=["lyra", "marta"], pc_slug="rowan"
    )
    assert "Scene: pc rowan" in build_notes(state, "en", pc_slug="rowan")
    assert "Scene:" not in build_notes(state, "en")


def test_build_notes_stall_order_fires_and_resets() -> None:
    state = DirectorState()
    for _ in range(STALL_ESCALATE_AFTER - 1):
        state.observe_movement(False)
    assert "has not moved" not in build_notes(state, "en")  # not yet
    state.observe_movement(False)  # reaches threshold
    assert "has not moved for 3 turns" in build_notes(state, "en")
    state.observe_movement(True)  # movement resets
    assert "has not moved" not in build_notes(state, "en")


def test_build_notes_conditional_sections() -> None:
    state = DirectorState()
    state.user_repeated = True
    state.reply_similarity = 0.9
    state.static_goal_slugs = ["lira"]
    payload = build_notes(state, "en")
    assert "repeated their previous input" in payload
    assert "nearly identical" in payload
    assert "lira" in payload


def test_build_notes_hungarian_and_fallback() -> None:
    state = DirectorState()
    state.user_repeated = True
    hu = build_notes(state, "hu")
    assert "kánon" in hu
    assert "megismételte" in hu
    fallback = build_notes(state, "de")
    assert "it is canon" in fallback


def test_build_notes_leaves_clock_line_to_the_sheets_block() -> None:
    # The 14B ignores the contract for world info (2026-09-25 replay-002);
    # the time rides the scene-sheets block instead (SceneSheetInjector).
    state = DirectorState()
    state.clock_line = "Day 2, evening."
    payload = build_notes(state, "en")
    assert "Day 2, evening." not in payload
    # No directives → no world-update block.
    assert "World update" not in payload


def test_build_notes_points_at_the_world_events_block() -> None:
    # The full directive text rides the scene-sheets block (the channel
    # the model obeys); the contract carries only the binding pointer.
    state = DirectorState()
    state.event_directives = ["Marta is missing.", "The purifier died."]
    payload = build_notes(state, "en")
    assert "World events" in payload
    assert "part of this contract" in payload
    assert "Marta is missing." not in payload  # text lives on the sheets block
    assert payload.index("it is canon") < payload.index("World events")


def test_build_notes_without_engine_state_is_unchanged_shape() -> None:
    payload = build_notes(DirectorState(), "en")
    assert "World events" not in payload
    assert "It is day" not in payload


# ---- voice check (long-scene register drift) ----------------------------


def test_voice_check_appears_after_long_tenure_only() -> None:
    state = DirectorState()
    register = "calm triage voice — short declaratives, medical precision"
    for _ in range(REGISTER_REMINDER_AFTER - 1):
        state.observe_scene_tenure("marta", register)
    assert state.voice_checks() == []
    state.observe_scene_tenure("marta", register)
    assert state.voice_checks() == [("marta", register)]
    payload = build_notes(state, "en")
    assert "Voice check — marta" in payload
    assert "short declaratives" in payload


def test_voice_check_needs_a_register_and_resets_on_exit() -> None:
    state = DirectorState()
    for _ in range(REGISTER_REMINDER_AFTER):
        state.observe_scene_tenure("stub-npc", "")  # no register on the sheet
    assert state.voice_checks() == []
    for _ in range(REGISTER_REMINDER_AFTER):
        state.observe_scene_tenure("marta", "triage voice")
    state.prune(keep=[])  # left the scene → tenure resets
    assert state.voice_checks() == []
    state.observe_scene_tenure("marta", "triage voice")  # fresh scene, tenure 1
    assert state.voice_checks() == []


# ---- banned phrases (cross-reply prose tics) -----------------------------


def _observe_reply(state: DirectorState, text: str) -> None:
    state.observe_turn(
        user_text="x", prev_user_text="y", assistant_text=text, prev_assistant_text=""
    )


def test_phrase_recycled_across_three_replies_gets_banned() -> None:
    state = DirectorState()
    tic = "your boots crunching against gravel"
    _observe_reply(state, f"You walk on, {tic}, toward the ridge line ahead.")
    assert state.banned_phrases == []
    _observe_reply(state, f"The path bends south, {tic} in the dark silence.")
    assert state.banned_phrases == []  # two replies is coincidence
    _observe_reply(state, f"Dawn comes grey, {tic} as the wire hums nearby.")
    assert any("crunching" in p for p in state.banned_phrases)
    payload = build_notes(state, "en")
    assert "Do not use them or close variants" in payload
    assert "crunching" in payload


def test_phrase_repeated_inside_one_reply_does_not_count() -> None:
    state = DirectorState()
    tic = "boots crunching against gravel"
    _observe_reply(state, f"{tic}. Again {tic}. And again {tic}.")
    _observe_reply(state, "Something completely different happens in the morning light.")
    assert state.banned_phrases == []


def test_one_long_tic_fills_one_slot_not_all() -> None:
    state = DirectorState()
    tic = "the shelter hums around you warm and steady and alive tonight"
    for filler in ("The rain begins.", "A door creaks.", "Nothing moves outside."):
        _observe_reply(state, f"{tic}. {filler}")
    # Overlapping 4-word windows of the same tic dedupe to one entry.
    assert len(state.banned_phrases) == 1


def test_function_word_shingles_are_ignored() -> None:
    state = DirectorState()
    for filler in ("Go on now.", "The rain falls.", "A door creaks open."):
        _observe_reply(state, f"you don't have to do that. {filler}")
    # "you don't have to" carries no ≥5-char word — never banned.
    assert all("don't have" not in p for p in state.banned_phrases)


def test_paragraph_loop_detected_after_two_replies() -> None:
    state = DirectorState()
    para = (
        "The dust stirs faintly beneath the collapsed shelter, settling again "
        "as silence reclaims the corridor."
    )
    _observe_reply(state, f"You move forward cautiously. {para} Nothing stirs.")
    assert state.paragraph_banned == []  # one reply is never a loop
    _observe_reply(state, f"The air is still. {para} Your boots find dust.")
    # An 8-gram from the shared paragraph should appear in paragraph_banned.
    assert state.paragraph_banned  # at least one phrase detected


def test_paragraph_repeated_inside_one_reply_does_not_count() -> None:
    state = DirectorState()
    para = "The collapsed wall breathes dust into the afternoon light slowly"
    _observe_reply(state, f"{para}. You stop. {para}.")
    # Same 8-gram twice in one reply: set-based, counts as 1.
    assert state.paragraph_banned == []


def test_paragraph_ban_does_not_pollute_four_gram_ban() -> None:
    state = DirectorState()
    long_phrase = (
        "darkness settles over the rubble like a second skin pressing every "
        "thought into the ground beneath your feet slowly"
    )
    _observe_reply(state, long_phrase)
    _observe_reply(state, f"You push on. {long_phrase} Still nothing moves.")
    # paragraph_banned fires at 2 replies; banned_phrases still needs 3.
    assert state.paragraph_banned  # 8-gram hit
    assert state.banned_phrases == []  # 4-gram not yet at threshold


def test_player_echo_detected_and_noted() -> None:
    # Kimi K2.6 marathon replay-004 T43-45: the DM reply was a verbatim
    # copy of the player's input.
    echo = "I let her words hang — smells like before — because I know that feeling."
    state = DirectorState()
    state.observe_turn(
        user_text=echo,
        prev_user_text="",
        assistant_text=echo,
        prev_assistant_text="",
    )
    assert state.player_echo
    assert "Do not repeat or restate the player's own message" in build_notes(state, "en")


def test_player_echo_not_flagged_on_a_real_reply() -> None:
    state = DirectorState()
    state.observe_turn(
        user_text="I ask her name and where she came from.",
        prev_user_text="",
        assistant_text=(
            'She studies you for a long moment before answering. "Lyra," '
            'she says finally, "from the river camps — what\'s left of them."'
        ),
        prev_assistant_text="",
    )
    assert not state.player_echo
    assert "Do not repeat or restate the player's own message" not in build_notes(state, "en")


def test_player_echo_catches_prefix_echo_with_long_narration_tail() -> None:
    # The real replay-004 shape: verbatim player text as the reply's
    # opening, followed by thousands of chars of narration.
    player = "I let her words hang — smells like before — because I know that feeling."
    state = DirectorState()
    state.observe_turn(
        user_text=player,
        prev_user_text="",
        assistant_text=player + " " + ("The corridor stretches on. " * 150),
        prev_assistant_text="",
    )
    assert state.player_echo


def test_phrase_ban_decays_once_the_tic_leaves_the_recent_window() -> None:
    # 16 real transcripts: top-3 ban slots held by phrases unseen for 10+
    # turns (all-time counts never decay). A ban survives the reply that
    # obeys it — no flicker — and lapses once the tic ages out.
    state = DirectorState()
    tic = "your boots crunching against gravel"
    for place in ("ridge", "wire", "gate"):
        _observe_reply(state, f"You walk on, {tic}, toward the {place}.")
    banned = list(state.banned_phrases)
    assert banned
    _observe_reply(state, "Quiet.")
    assert state.banned_phrases == banned
    for _ in range(PHRASE_WINDOW_REPLIES):
        _observe_reply(state, "Quiet.")
    assert state.banned_phrases == []


def test_a_stalled_reply_asks_for_movement_within_the_scene_not_a_time_skip() -> None:
    notes = build_notes(DirectorState(reply_stalled=True), "en")
    assert "re-told the same beat" in notes
    assert "WITHIN the scene" in notes and "Do not skip time" in notes
    assert "re-told the same beat" not in build_notes(DirectorState(), "en")


def test_pass_turns_ask_for_a_full_exchange_without_a_time_skip() -> None:
    # 2026-09-27 playtest: "Continue" / "I listen patiently…" crawled.
    for text in ("Continue", "...", "I listen patiently. Who knows where this goes....", "go on"):
        assert player_passes(text), text
    for text in (
        "I unzip my bag and hand her the map.",
        "Continue the attack on the gate now please",
    ):
        assert not player_passes(text), text
    notes = build_notes(DirectorState(), "en", player_passes=True)
    assert "letting the scene run" in notes and "Do not skip time" in notes
    assert "letting the scene run" not in build_notes(DirectorState(), "en")
