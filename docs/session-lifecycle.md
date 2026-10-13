# Session lifecycle

How a session starts, ends, and (mis)behaves under disconnect. Decisions
recorded here so the WebRTC wiring and the Phase 4 session-log code
agree on a single state machine instead of inferring one each.

## Definition

A **session** is one continuous AI-DM ↔ player exchange. It corresponds
to one `SessionContext` (locale + session id), one `LLMContext`
(conversation history + tool state), and — under WebRTC — one
`RTCPeerConnection` between the browser and the orchestrator.

Sessions are not persistent across processes. Restarting the
orchestrator drops the in-memory `LLMContext`; the only durable record
is the session log written under `runs/<active-run>/sessions/` inside the active campaign vault.

## Transport ↔ lifecycle mapping

### `AIDM_TRANSPORT=local` (CLI)

| event                  | action                                                    |
| ---------------------- | --------------------------------------------------------- |
| process start          | build `SessionContext` from `Config.locale`, run pipeline |
| `Ctrl+C` / SIGTERM     | flush + write session log, then exit                      |
| crash                  | no session log; tail the loguru output for forensics      |

One process = one session. Local-mode keeps the original Phase 4
semantics; nothing changes here.

### `AIDM_TRANSPORT=webrtc` (browser)

| event                                      | action                                                                  |
| ------------------------------------------ | ----------------------------------------------------------------------- |
| browser opens page                         | no session yet                                                          |
| browser submits start-session form         | server builds `SessionContext` (locale from form), creates `LLMContext` |
| `RTCPeerConnection` reaches `connected`    | pipeline runs                                                           |
| user clicks **End session** in UI          | DM's farewell turn runs to completion, then the server ends the session |
| peer-connection drops without End          | context stashed, 5s grace for reconnect; on timeout, tear down          |
| browser reconnects within grace period     | resume same `LLMContext`, no log write                                  |
| browser reconnects after grace period      | new session, new `LLMContext`                                           |
| orchestrator restart with active session   | session lost (no resumable state on disk by design)                     |

Peer-connection lifecycle is the source of truth, **not** browser tab
visibility. A backgrounded tab keeps the peer alive; closing the tab
drops it.

## Mid-turn behavior

A "turn" is the span between `UserStartedSpeakingFrame` and the
matching `BotStoppedSpeakingFrame`. If the peer-connection drops
mid-turn:

* **STT in flight** — drop the partial audio buffer; do not retry on
  reconnect. STT context (Whisper priming, Scribe keyterms) is rebuilt
  fresh.
* **LLM in flight** — cancel the in-flight request. Pipecat's
  `InterruptionFrame` handles this; we don't add a separate cancel
  path. No replay on reconnect.
* **TTS in flight** — discard buffered audio; the player hears silence
  on reconnect, not the tail of a previous response.

Rationale: replay would be confusing (the player heard half the
response already, the orchestrator doesn't know how much), and a
half-finished turn is more confusing in transcript form than a clean
restart. Trade clarity for completeness here.

## Session log write

The session log is written by the DM itself on its farewell turn
(`vault_write_session_log`, prompted by the persona). The server does
**not** write a log on grace-period timeout — the durable record of an
abandoned session is the rolling summary note
(`notes/session-<id>.md`, written every few turns by the
`SessionSummarizer`), which the resume picker offers as a
"running" source. We do **not** post-process the DM's log: if it didn't
summarize, the log is short — that's a persona problem, not a transport
problem.

## Text transport (`AIDM_TRANSPORT=text`)

Different window, different mechanism: the pipeline stays **live** while
no SSE stream is attached, and is torn down once the window passes. The
window is `AIDM_TEXT_RECONNECT_GRACE_S`, default **30 minutes** — a phone
lock drops the SSE stream and outlasts 5 s every time (2026-09-27 playtest
lost its session that way), and an idle text pipeline holds no audio or
peer connection, only its `LLMContext`. A
reattach inside the window re-arms it and resumes the same pipeline
(`resumed_from: "live"`, the only reconnect the web client accepts).
`/end` lets the farewell turn finish, then the pipeline ends itself.
`/history` serves a per-session transcript kept outside `LLMContext`,
so a refresh repaints every turn even after summaries compacted the
model's context. If the window has passed, the client starts a fresh
session on the same run (the vault — sheets, summary notes, clock,
events — carries the state) and keeps the old transcript on screen.

## Reconnect semantics (WebRTC)

Resume window: **5 seconds** from peer-connection drop. Picked because:

* Wi-Fi handoffs (AP roaming, network switch) typically resolve under
  3s; 5s covers the common case.
* Longer windows hold `LLMContext` (potentially with tool state)
  indefinitely on bad network, leaking memory.
* The browser shows a "Reconnecting…" indicator during the window.

The browser identifies the resuming session via a server-issued
`session_id` cookie or query parameter (decision deferred to the
WebRTC signaling commit). The server matches `session_id` to the
in-memory `LLMContext` and re-attaches.

## Not in scope (yet)

* Multi-user shared sessions.
* Resume across orchestrator restarts (requires `LLMContext`
  serialization).
* Background-tab muting / mic auto-mute.
* Push-to-talk vs. continuous-listen toggle.
