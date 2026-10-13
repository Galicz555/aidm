# web

SvelteKit (Svelte 5, TypeScript) browser client for AIDM. WebRTC mic +
speaker against the orchestrator's `/api/offer` endpoint.

## Run

One command from the repo root runs both the orchestrator (in
`AIDM_TRANSPORT=webrtc` mode) and the Vite dev server in parallel:

```
make web-install            # first run only — npm install in web/
make start-public-voice     # starts both; Ctrl+C tears down both
# variants: start-public-voice-debug, start-private-text, start-private-text-debug
```

Open <http://localhost:5173>. Pick a locale, click **Start session**, talk.

Manually if you want them in separate terminals:

```
# terminal 1
AIDM_TRANSPORT=webrtc uv run aidm

# terminal 2
cd web && npm run dev
```

Vite proxies `/api/*` and `/health` to `http://127.0.0.1:8000` (override
with `AIDM_ORCHESTRATOR_URL`).

## Build

```
npm run build       # produces web/build/ — static SPA bundle
npm run preview     # serve the bundle locally
```

The orchestrator will serve `web/build/` from `/` in a later commit; the
proxy goes away in production because everything is same-origin.

## Layout

```
src/
  app.html              SvelteKit shell
  app.d.ts              ambient types
  routes/
    +layout.ts          SSR/prerender disabled (SPA)
    +page.svelte        single-page session UI
  lib/
    webrtc.ts           AidmSession — peer connection + data channel + reconnect
```

## What the UI shows

* **Transcript panel** — final user transcripts and streaming bot
  deltas, fed by `transcript.user` / `transcript.bot_delta` /
  `transcript.bot_end` data-channel events. A blinking caret marks the
  in-flight bot turn.
* **Scene panel** — name + short description, updated on `scene.change`
  data-channel events when the DM calls `vault_set_scene`.
* **Character panel** — slug header + the full sheet, fetched from
  `/api/vault/characters/{slug}` and rendered via `marked` when the DM
  calls `vault_set_character`.
* **Locale dropdown** — HU (default) / EN, sent with the offer; the
  orchestrator builds the SessionContext with that locale.
* **Start/End session button** — End triggers a polite end (DM writes
  the session log + says goodbye, page auto-disconnects on the next
  `transcript.bot_end`).
* **Reconnect** — if the peer drops unexpectedly within 5s, the page
  transparently resumes with the stored `session_id` and the
  conversation continues.

See `docs/session-lifecycle.md` for the underlying state machine.

## Notes

* SPA only — no SSR. WebRTC, `getUserMedia`, and the audio element all
  need a real browser, and there's no SEO surface to render.
* Mic constraints enable browser-side echo cancellation, noise
  suppression, and AGC. These are the single biggest audio-quality
  lever and they cost nothing.
* No STUN/TURN configured: dev is same-host. Remote use needs a STUN
  server in `RTCPeerConnection({ iceServers: [...] })` — deploy-time
  decision, not a code one.
* `marked` renders the character sheet markdown into HTML. Vault
  content is fully under our control (we author the markdown), so we
  inject via `{@html ...}` without a sanitizer. If we ever ingest
  external content the trust boundary moves and DOMPurify lands.
