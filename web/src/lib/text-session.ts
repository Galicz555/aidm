// Browser-side text-chat session against the orchestrator's
// /api/text/* endpoints.
//
// Flow:
//   1. POST /api/text/session { locale, session_id? } → { session_id }.
//      A fresh session_id starts new; a known one resumes within the
//      5s grace window (same as WebRTC).
//   2. EventSource('/api/text/session/{id}/stream') opens an SSE
//      connection. The server pushes the same JSON payloads the
//      WebRTC data channel does (transcript.bot_delta, transcript.bot_end,
//      scene.change, character.change), so the page's onMessage
//      callbacks are unchanged.
//   3. POST /api/text/session/{id}/turn { text } pushes a player line.
//      No echo: the page renders the player's own input locally.
//   4. POST /api/text/session/{id}/end on the End-session button — the
//      DM writes the session log and says goodbye; the page waits for
//      the first transcript.bot_end after farewell text, then stop().
//   5. POST /api/text/session/{id}/regenerate on the ↻ of the last DM
//      reply — the server rolls that turn back and re-runs the player line.

import type { DataChannelMessage } from './webrtc';

export type TextSessionState =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'ending'
  | 'disconnecting'
  | 'error';

export interface TextSessionCallbacks {
  onStateChange: (state: TextSessionState) => void;
  onError: (message: string) => void;
  onMessage: (message: DataChannelMessage) => void;
  /** Fires after a successful reattach when the server projected a
   *  fresh transcript history. The page repaints from this so any
   *  events the browser missed while the tab was hidden / suspended
   *  (mobile lock) re-sync from the live LLMContext. */
  onHistorySync?: (history: HistoryResponse) => void;
  /** Fires when the server had already ended the session (reconnect
   *  window passed) and a fresh session now continues the same run,
   *  seeded from the old session's last summary when one exists. */
  onSessionContinued?: (previousSessionId: string) => void;
}

/** The server no longer knows the session (its reconnect window passed). */
class SessionGoneError extends Error {}

export interface TextStartOptions {
  locale: 'hu' | 'en';
  resumeFrom?: { source: 'log' | 'running'; id: string } | null;
  /** Active campaign slug; required server-side, defaults to VAULT_PATH for legacy clients. */
  campaignId: string;
  /** Active run slug; required server-side unless RUN_SLUG env supplies a default. */
  runId: string;
}

interface StartResponse {
  session_id: string;
  locale: 'hu' | 'en';
  resumed_from?: string;
}

export interface HistoryTurn {
  role: 'user' | 'assistant';
  text: string;
}

export interface HistoryResponse {
  session_id: string;
  locale: 'hu' | 'en';
  turns: HistoryTurn[];
}

export class AidmTextSession {
  private state: TextSessionState = 'idle';
  private sessionId: string | null = null;
  private lastLocale: 'hu' | 'en' | null = null;
  private lastCampaignId: string = '';
  private lastRunId: string = '';
  private events: EventSource | null = null;
  // stop() / end() set this so an unsolicited stream error doesn't
  // bounce us back into reconnect after the user told us to close.
  private userInitiatedDisconnect = false;
  // Guards against overlapping resume attempts when both the SSE
  // error and a visibilitychange wake-up fire near-simultaneously
  // (typical mobile-unlock pattern).
  private resumeInFlight = false;

  constructor(private readonly callbacks: TextSessionCallbacks) {}

  async start(options: TextStartOptions): Promise<void> {
    if (
      this.state !== 'idle' &&
      this.state !== 'error' &&
      this.state !== 'reconnecting'
    ) {
      this.callbacks.onError(`cannot start: session is ${this.state}`);
      return;
    }

    this.userInitiatedDisconnect = false;
    this.lastLocale = options.locale;
    this.lastCampaignId = options.campaignId;
    this.lastRunId = options.runId;

    try {
      if (this.state !== 'reconnecting') this.transition('connecting');

      // body is intentionally Record<string, unknown> — resume_from is
      // an object, not a string, so the value type can't be string-only.
      const body: Record<string, unknown> = {
        locale: options.locale,
        campaign_id: options.campaignId,
        run_id: options.runId
      };
      if (this.state === 'reconnecting' && this.sessionId) {
        body.session_id = this.sessionId;
      }
      // Reconnect (within 5s grace) keeps the live LLMContext; we
      // only pass resume_from on a *fresh* start so we don't
      // accidentally re-seed a session that's already underway.
      if (options.resumeFrom && this.state !== 'reconnecting') {
        body.resume_from = options.resumeFrom;
      }
      const response = await fetch('/api/text/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`session start rejected (${response.status}): ${detail}`);
      }
      const result = (await response.json()) as StartResponse;
      this.sessionId = result.session_id;

      this.openStream();
      this.transition('connected');
    } catch (err) {
      this.handleFailure(err);
    }
  }

  /** Attach to a still-running session on the server.
   *
   * Use after a page refresh: the browser kept the session_id in
   * localStorage, the server's pipeline is still alive, we want to
   * reattach without spawning a new one. Returns the projected
   * conversation so the caller can repaint the transcript. Throws if
   * the server doesn't know the id (process restart, expired grace).
   */
  async attach(
    sessionId: string,
    run?: { campaignId: string; runId: string }
  ): Promise<HistoryResponse> {
    if (this.state !== 'idle' && this.state !== 'error') {
      throw new Error(`cannot attach: session is ${this.state}`);
    }
    this.userInitiatedDisconnect = false;
    // Remember the run so a later lost session can be continued on it.
    if (run) {
      this.lastCampaignId = run.campaignId;
      this.lastRunId = run.runId;
    }
    this.transition('connecting');
    // Hot-reattach via the start route — the server returns the live
    // session metadata without spawning a new pipeline because the
    // session_id is in its live registry.
    const startResponse = await fetch('/api/text/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    if (!startResponse.ok) {
      this.transition('idle');
      throw new Error(`session not found (${startResponse.status})`);
    }
    const startBody = (await startResponse.json()) as StartResponse;
    if (startBody.resumed_from !== 'live') {
      // Server returned 200 but didn't recognize the session as live —
      // it must have started a fresh one. Abort and let the caller
      // decide whether to use the fresh session or bail.
      this.sessionId = startBody.session_id;
      this.lastLocale = startBody.locale;
      this.openStream();
      this.transition('connected');
      throw new Error('session expired; started fresh');
    }
    this.sessionId = startBody.session_id;
    this.lastLocale = startBody.locale;

    const historyResponse = await fetch(
      `/api/text/session/${this.sessionId}/history`
    );
    if (!historyResponse.ok) {
      this.transition('idle');
      throw new Error(`history fetch failed (${historyResponse.status})`);
    }
    const history = (await historyResponse.json()) as HistoryResponse;
    this.openStream();
    this.transition('connected');
    return history;
  }

  async send(text: string): Promise<void> {
    if (this.state !== 'connected' || !this.sessionId) {
      this.callbacks.onError(`cannot send: session is ${this.state}`);
      return;
    }
    const trimmed = text.trim();
    if (!trimmed) return;
    try {
      const response = await fetch(
        `/api/text/session/${this.sessionId}/turn`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: trimmed })
        }
      );
      if (!response.ok) {
        const detail = await response.text();
        this.callbacks.onError(`turn rejected (${response.status}): ${detail}`);
      }
    } catch (err) {
      this.callbacks.onError(err instanceof Error ? err.message : String(err));
    }
  }

  /** Discard the last DM reply and re-run the same player line.
   *
   *  The server rolls the world back to before that turn (history,
   *  scene, sheets on disk) and archives the discarded reply; the new
   *  reply streams over SSE as usual. Resolves false when the server
   *  refused (409: a turn is in flight, or nothing to regenerate).
   */
  async regenerate(): Promise<boolean> {
    if (this.state !== 'connected' || !this.sessionId) return false;
    try {
      const response = await fetch(
        `/api/text/session/${this.sessionId}/regenerate`,
        { method: 'POST' }
      );
      if (!response.ok) {
        const detail = await response.text();
        this.callbacks.onError(`regenerate rejected (${response.status}): ${detail}`);
        return false;
      }
      return true;
    } catch (err) {
      this.callbacks.onError(err instanceof Error ? err.message : String(err));
      return false;
    }
  }

  /** Politely end the session: persona writes the log + says goodbye.
   *
   *  The page is expected to call ``stop()`` once the farewell text has
   *  landed — the first ``transcript.bot_end`` after non-empty text. The
   *  session-log tool leg ends with an empty ``bot_end`` of its own.
   */
  async end(): Promise<void> {
    if (this.state !== 'connected' || !this.sessionId) return;
    this.userInitiatedDisconnect = true;
    this.transition('ending');
    try {
      const response = await fetch(
        `/api/text/session/${this.sessionId}/end`,
        { method: 'POST' }
      );
      if (!response.ok) {
        this.callbacks.onError(`end-session rejected (${response.status})`);
        await this.stop();
      }
    } catch (err) {
      this.callbacks.onError(err instanceof Error ? err.message : String(err));
      await this.stop();
    }
  }

  async stop(): Promise<void> {
    if (this.state === 'idle' || this.state === 'disconnecting') return;
    this.userInitiatedDisconnect = true;
    this.transition('disconnecting');
    this.teardownStream();
    this.sessionId = null;
    this.lastLocale = null;
    this.transition('idle');
  }

  get isEnding(): boolean {
    return this.state === 'ending';
  }

  /** The active server-side session id. Null when idle. */
  get currentSessionId(): string | null {
    return this.sessionId;
  }

  private openStream(): void {
    if (!this.sessionId) return;
    this.teardownStream();
    const source = new EventSource(
      `/api/text/session/${this.sessionId}/stream`
    );
    source.onmessage = (event) => {
      if (typeof event.data !== 'string') return;
      try {
        const payload = JSON.parse(event.data) as DataChannelMessage;
        this.callbacks.onMessage(payload);
      } catch {
        /* malformed event — ignore */
      }
    };
    source.onerror = () => {
      if (this.userInitiatedDisconnect) return;
      if (this.state === 'connected' && this.sessionId && this.lastLocale) {
        // SSE drops are usually network-level. The server keeps the
        // session warm for 5s; reopen with the same session_id.
        void this.attemptResume();
      }
    };
    this.events = source;
  }

  private teardownStream(): void {
    if (this.events) {
      this.events.onmessage = null;
      this.events.onerror = null;
      this.events.close();
      this.events = null;
    }
  }

  /** External nudge to re-sync after the tab was hidden (mobile lock,
   *  tab switch). No-op if we're not in a state that wants reconnection. */
  async resumeIfStale(): Promise<void> {
    if (this.userInitiatedDisconnect) return;
    if (!this.sessionId) return;
    if (
      this.state !== 'connected' &&
      this.state !== 'error' &&
      this.state !== 'reconnecting'
    ) {
      return;
    }
    await this.attemptResume();
  }

  private async attemptResume(): Promise<void> {
    if (this.resumeInFlight) return;
    if (!this.sessionId) return;
    this.resumeInFlight = true;
    this.transition('reconnecting');
    this.teardownStream();

    // Mobile wake-up: the JS event loop resumes before the radio is
    // back. Retry with a short backoff so a transient ECONNREFUSED on
    // the first ping doesn't kill the session.
    const backoffsMs = [0, 500, 1500, 3500];
    let lastError: unknown = null;
    try {
      for (const delay of backoffsMs) {
        if (this.userInitiatedDisconnect) return;
        if (delay > 0) await new Promise((r) => setTimeout(r, delay));
        try {
          const ok = await this.reattachOnce();
          if (ok) return;
        } catch (err) {
          if (err instanceof SessionGoneError && this.lastRunId && this.sessionId) {
            await this.continueOnSameRun(this.sessionId);
            return;
          }
          lastError = err;
        }
      }
      this.handleFailure(
        lastError ?? new Error('reconnect failed: server lost the session')
      );
    } finally {
      this.resumeInFlight = false;
    }
  }

  /** One attempt to re-bind to the live server-side session.
   *
   *  Returns true on success (SSE reopened, history re-synced). Returns
   *  false if the server responded but did NOT recognize the session as
   *  live (5s grace expired, process restart) — caller decides whether
   *  to retry or surface an error. Throws on transport errors.
   */
  private async reattachOnce(): Promise<boolean> {
    if (!this.sessionId) return false;
    const response = await fetch('/api/text/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: this.sessionId })
    });
    if (!response.ok) {
      // 4xx: the server ended the session (reconnect window passed).
      if (response.status >= 400 && response.status < 500) {
        throw new SessionGoneError(`session not recoverable (${response.status})`);
      }
      return false;
    }
    const body = (await response.json()) as StartResponse;
    if (body.resumed_from !== 'live') {
      // Server spawned a fresh session — that's not a reconnect, that's
      // a brand-new chat with the player's history thrown away. Treat
      // as terminal so the page surfaces an error instead of silently
      // losing the transcript.
      throw new Error('session expired; would lose transcript');
    }
    // Live session confirmed. Pull the projected history so any events
    // missed during sleep re-paint, then reopen the SSE stream.
    try {
      const historyResponse = await fetch(
        `/api/text/session/${this.sessionId}/history`
      );
      if (historyResponse.ok) {
        const history = (await historyResponse.json()) as HistoryResponse;
        if (history.locale === 'hu' || history.locale === 'en') {
          this.lastLocale = history.locale;
        }
        this.callbacks.onHistorySync?.(history);
      }
    } catch {
      /* history is best-effort: a missing one still lets the SSE
         reopen and the player keep typing. */
    }
    this.openStream();
    this.transition('connected');
    return true;
  }

  /** Start a fresh session on the same run after the old one ended.
   *
   *  The vault carries the run's state (sheets, clock, events); the old
   *  session's rolling summary (notes/session-<id>.md) seeds the new
   *  context. No summary yet (short session) → 404 → start without it.
   *  2026-09-27: a phone lock outlasted the window and the page was left
   *  with a 400 and no way back into the run. */
  private async continueOnSameRun(previousId: string): Promise<void> {
    const base = {
      locale: this.lastLocale,
      campaign_id: this.lastCampaignId,
      run_id: this.lastRunId
    };
    const post = (body: Record<string, unknown>) =>
      fetch('/api/text/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
    try {
      let response = await post({ ...base, resume_from: { source: 'running', id: previousId } });
      if (response.status === 404) response = await post(base);
      if (!response.ok) {
        throw new Error(`session lost and could not be continued (${response.status})`);
      }
      const result = (await response.json()) as StartResponse;
      this.sessionId = result.session_id;
      this.openStream();
      this.transition('connected');
      this.callbacks.onSessionContinued?.(previousId);
    } catch (err) {
      this.handleFailure(err);
    }
  }

  private handleFailure(err: unknown): void {
    const message = err instanceof Error ? err.message : String(err);
    this.callbacks.onError(message);
    this.transition('error');
    this.teardownStream();
  }

  private transition(next: TextSessionState): void {
    this.state = next;
    this.callbacks.onStateChange(next);
  }
}
