<svelte:options runes={true} />

<script lang="ts">
  // Text-adventure page, mobile-first.
  //
  // Layout:
  //   - Mobile (default): single column that fills the viewport (dvh).
  //     Sticky app bar on top, transcript scrolls in the middle,
  //     composer sticks to the bottom with safe-area padding. Scene +
  //     character live in a slide-up sheet you open from the app bar.
  //   - Tablet/desktop (>= 960px): the scene/character sheet is a
  //     persistent right-hand aside instead.
  //
  // Tone + Beat editors live in TuningPanel.
  // Character sheet view/edit + roleplay-prompt dialog live in CharacterPanel.
  // The 3-pane resume picker lives in ResumeSheet.

  import { onDestroy, onMount } from 'svelte';
  import { renderMarkdown } from '$lib/markdown';
  import { toast } from 'svelte-sonner';
  import type { DataChannelMessage } from '$lib/webrtc';
  import { AidmTextSession, type TextSessionState } from '$lib/text-session';
  import { replaceRollsInline } from '$lib/dice';
  // The run picker uses different shapes than the resume-grouping
  // "Campaign" interface defined below — keep the imports aliased so
  // the names don't clash.
  import {
    listCampaigns as listPickerCampaigns,
    listRuns as listPickerRuns,
    type Campaign as PickerCampaign,
    type Run as PickerRun,
  } from '$lib/runs';
  // createRun is invoked through the shared RunPicker component now;
  // keep the alias only where the page calls it directly.
  import { createRun as createPickerRun } from '$lib/runs';
  import type { Campaign, ResumeSession } from '$lib/types/aidm';

  // shadcn-svelte UI primitives
  import * as Card from '$lib/components/ui/card';
  import * as Sheet from '$lib/components/ui/sheet';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from '$lib/components/ui/badge';
  import ModeBadge from '$lib/components/aidm/ModeBadge.svelte';
  import RunPicker from '$lib/components/aidm/RunPicker.svelte';
  import StatusPill from '$lib/components/aidm/StatusPill.svelte';
  import TuningPanel from '$lib/components/aidm/TuningPanel.svelte';
  import CharacterPanel from '$lib/components/aidm/CharacterPanel.svelte';
  import ResumeSheet from '$lib/components/aidm/ResumeSheet.svelte';

  type Speaker = { kind: 'user' } | { kind: 'dm' } | { kind: 'npc'; slug: string };

  interface TranscriptEntry {
    role: 'user' | 'assistant';
    speaker: Speaker;
    text: string;
  }

  function parseSpeaker(raw: unknown): Speaker {
    if (typeof raw !== 'string') return { kind: 'dm' };
    if (raw === 'dm') return { kind: 'dm' };
    if (raw === 'user') return { kind: 'user' };
    if (raw.startsWith('npc:')) {
      const slug = raw.slice(4).trim();
      if (slug) return { kind: 'npc', slug };
    }
    return { kind: 'dm' };
  }

  function speakerLabel(speaker: Speaker): string {
    if (speaker.kind === 'user') return 'You';
    if (speaker.kind === 'dm') return 'DM';
    return speaker.slug;
  }

  function sameSpeaker(a: Speaker, b: Speaker): boolean {
    if (a.kind !== b.kind) return false;
    if (a.kind === 'npc' && b.kind === 'npc') return a.slug === b.slug;
    return true;
  }

  let locale = $state<'hu' | 'en'>('en');
  // Active run for character endpoint calls + start_session payload.
  // localStorage carries the picker selection across reloads; if absent,
  // we fall back to "default" (the migration's slug for legacy vaults).
  let selectedRun = $state<string>(
    (typeof localStorage !== 'undefined' && localStorage.getItem('aidm:run')) || 'default'
  );

  // Run picker state — parallel to the resume-grouping campaign concept
  // (which lives in ``campaigns`` below). These are the campaigns the
  // process is bound to; runs hang under the selected campaign.
  let pickerCampaigns = $state<PickerCampaign[]>([]);
  let selectedPickerCampaign = $state<string>(
    (typeof localStorage !== 'undefined' && localStorage.getItem('aidm:campaign')) || ''
  );
  let pickerRuns = $state<PickerRun[]>([]);
  let creatingRun = $state<boolean>(false);

  async function loadPickerCampaigns(): Promise<void> {
    try {
      pickerCampaigns = await listPickerCampaigns();
      const initial =
        pickerCampaigns.find((c) => c.slug === selectedPickerCampaign)?.slug ??
        pickerCampaigns[0]?.slug ??
        '';
      selectedPickerCampaign = initial;
      if (initial) await loadPickerRuns(initial);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadPickerRuns(campaignSlug: string): Promise<void> {
    try {
      pickerRuns = await listPickerRuns(campaignSlug);
      const initial =
        pickerRuns.find((r) => r.slug === selectedRun)?.slug ?? pickerRuns[0]?.slug ?? '';
      selectedRun = initial;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    }
  }

  async function onPickerCampaignChange(value: string): Promise<void> {
    selectedPickerCampaign = value;
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('aidm:campaign', value);
    }
    if (value) await loadPickerRuns(value);
  }

  function onPickerRunChange(value: string): void {
    selectedRun = value;
    if (typeof localStorage !== 'undefined' && value) {
      localStorage.setItem('aidm:run', value);
    }
  }

  async function submitNewRun(title: string): Promise<void> {
    if (!selectedPickerCampaign) return;
    creatingRun = true;
    try {
      const created = await createPickerRun(selectedPickerCampaign, { title });
      await loadPickerRuns(selectedPickerCampaign);
      selectedRun = created.slug;
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('aidm:run', selectedRun);
      }
      toast.success(`Run created: ${created.title}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      creatingRun = false;
    }
  }

  let sessionState = $state<TextSessionState>('idle');
  let errorMessage = $state<string | null>(null);
  let transcript = $state<TranscriptEntry[]>([]);
  let pendingBotText = $state<string>('');
  let pendingBotSpeaker = $state<Speaker>({ kind: 'dm' });
  let inputText = $state<string>('');
  let sceneName = $state<string | null>(null);
  let sceneDescription = $state<string>('');
  // characterSlug is the single source of truth for the active character;
  // CharacterPanel watches this prop and fetches/clears its own sheet state.
  let characterSlug = $state<string | null>(null);
  // toneText is kept here only for passing to ModeBadge; TuningPanel is the
  // source of truth and notifies us via the ontoneloaded callback.
  let toneText = $state<string>('');
  // externalBeat is a signal channel: the page sets it when SSE beat.change
  // arrives so TuningPanel can sync its internal state.
  let externalBeat = $state<string | null>(null);

  interface NpcState {
    slug: string;
    mood: string;
    scene_goal: string;
    social_mask: string;
  }
  let npcStates = $state<NpcState[]>([]);
  const npcMoodBySlug = $derived.by(() => {
    const m: Record<string, string> = {};
    for (const n of npcStates) if (n.mood) m[n.slug] = n.mood;
    return m;
  });

  interface DebugEntry {
    turn: number;
    context: { role: string; text: string }[];
    reply: string;
  }
  let debugEnabled = $state<boolean>(false);
  let debugTurns = $state<DebugEntry[]>([]);

  interface MemoryEvent {
    at: Date;
    summarized: number;
    kept: number;
    summary: string;
  }
  let memoryEvents = $state<MemoryEvent[]>([]);

  let serverMode = $state<'public' | 'private' | 'unknown'>('unknown');

  // Composite key "source:id" so the resume start request can rebuild
  // the {source, id} pair. Empty string == start fresh (no resume).
  let resumeChoice = $state<string>('');
  const resumeFrom = $derived((): { source: 'log' | 'running'; id: string } | null => {
    if (!resumeChoice) return null;
    const [source, ...rest] = resumeChoice.split(':');
    if (source !== 'log' && source !== 'running') return null;
    return { source, id: rest.join(':') };
  });

  let resumeSheetOpen = $state<boolean>(false);
  let campaigns = $state<Campaign[]>([]);
  let orphanSessions = $state<ResumeSession[]>([]);
  let infoOpen = $state<boolean>(false);

  const resumeAvailable = $derived(
    campaigns.some((c) => c.sessions.length > 0) || orphanSessions.length > 0
  );

  let transcriptEl = $state<HTMLOListElement | null>(null);
  // True from turn send until the post-turn sheet writer finishes (the
  // server-side turn.status: idle event). With local Gemma the first-token
  // latency is several seconds AND the writer takes 25-70s more, so without
  // a status that spans both the chat looks done long before the server is.
  let isGenerating = $state<boolean>(false);
  // 'streaming' = DM tokens flowing; 'writing_sheets' = post-turn writer
  // running; 'idle' = ready for next turn. Drives the typing-indicator label.
  let writerPhase = $state<'streaming' | 'writing_sheets' | 'idle'>('idle');
  // Turn submitted while the post-turn sheet writer was still running.
  // Dispatch must wait for turn.status: idle (the next turn's injector
  // reads the state files the writer is mid-rewrite on), but the human
  // shouldn't have to watch the writer: typing and submit stay open,
  // and the queued turn fires the moment the server reports idle.
  let queuedTurn = $state<string>('');
  // /end in flight: the session-log tool leg ends with an empty bot_end of
  // its own, so the page closes only after the farewell text has streamed.
  let farewellHeard = false;

  // localStorage key for the active session id. Set on a successful
  // Start, cleared on Stop/End. Read on page mount to detect a refresh
  // while a session was running — the page tries to reattach to the
  // live pipeline before falling back to fresh start.
  const SESSION_STORAGE_KEY = 'aidm.text.sessionId';

  function persistSessionId(id: string | null): void {
    if (typeof window === 'undefined') return;
    try {
      if (id) window.localStorage.setItem(SESSION_STORAGE_KEY, id);
      else window.localStorage.removeItem(SESSION_STORAGE_KEY);
    } catch {
      /* private-mode storage / quota — accept the data loss */
    }
  }

  function loadPersistedSessionId(): string | null {
    if (typeof window === 'undefined') return null;
    try {
      return window.localStorage.getItem(SESSION_STORAGE_KEY);
    } catch {
      return null;
    }
  }

  onMount(async () => {
    try {
      const response = await fetch('/api/info');
      if (!response.ok) return;
      const body = (await response.json()) as { mode?: string };
      if (body.mode === 'public' || body.mode === 'private') {
        serverMode = body.mode;
      }
    } catch {
      /* server may not be up — leave as unknown */
    }
    await loadCampaigns();
    await loadPickerCampaigns();
    // TuningPanel loads scene tuning automatically via its own $effect.

    // After info is populated, attempt a hot reattach if the previous
    // page-load persisted a session_id.
    const persisted = loadPersistedSessionId();
    if (persisted) {
      try {
        const history = await session.attach(persisted, {
          campaignId: selectedPickerCampaign,
          runId: selectedRun
        });
        // Repopulate the transcript from server-side projection so the
        // refresh lands exactly where the player left off.
        transcript = history.turns.map((t) => ({
          role: t.role,
          speaker: t.role === 'user' ? { kind: 'user' as const } : { kind: 'dm' as const },
          text: t.text
        }));
        if (history.locale === 'hu' || history.locale === 'en') {
          locale = history.locale;
        }
      } catch {
        // The persisted id is stale (process restart, expired session).
        // Continue the same run in a fresh session seeded from that
        // session's last summary instead of dropping the player at an
        // empty picker (2026-09-27 phone-lock loss).
        persistSessionId(null);
        if (selectedRun) {
          await session.start({
            locale,
            campaignId: selectedPickerCampaign,
            runId: selectedRun,
            resumeFrom: { source: 'running', id: persisted }
          });
        }
      }
    }

    // Mobile lock / unlock fires visibilitychange before the SSE
    // EventSource notices its socket is dead. Treat the visible+online
    // transition as the canonical "we're back" signal and proactively
    // re-sync — beats waiting for an onerror that may arrive seconds
    // later, or never if the browser silently dropped the stream.
    document.addEventListener('visibilitychange', handleVisibility);
    window.addEventListener('online', handleOnline);
  });

  function handleVisibility(): void {
    if (document.visibilityState !== 'visible') return;
    if (!navigator.onLine) return;
    void session.resumeIfStale();
  }

  function handleOnline(): void {
    if (document.visibilityState !== 'visible') return;
    void session.resumeIfStale();
  }

  function parseSession(e: Record<string, unknown>): ResumeSession {
    return {
      id: typeof e.id === 'string' ? e.id : '',
      source: e.source === 'running' ? 'running' : 'log',
      title: typeof e.title === 'string' ? e.title : '(untitled)',
      mtime: typeof e.mtime === 'number' ? e.mtime : 0,
      snippet: typeof e.snippet === 'string' ? e.snippet : '',
      campaign: typeof e.campaign === 'string' ? e.campaign : null,
      date: typeof e.date === 'string' ? e.date : null,
      characters: Array.isArray(e.characters)
        ? e.characters.filter((c): c is string => typeof c === 'string')
        : [],
      locations: Array.isArray(e.locations)
        ? e.locations.filter((l): l is string => typeof l === 'string')
        : []
    };
  }

  async function loadCampaigns(): Promise<void> {
    try {
      // /api/campaigns is the picker's flat list — session grouping
      // moved to /api/sessions/grouping with the same payload shape.
      // Pass campaign + run query so multi-campaign resume works.
      const params = new URLSearchParams();
      if (selectedPickerCampaign) params.set('campaign', selectedPickerCampaign);
      if (selectedRun) params.set('run', selectedRun);
      const response = await fetch(
        `/api/sessions/grouping${params.toString() ? '?' + params : ''}`
      );
      if (!response.ok) return;
      const body = (await response.json()) as {
        campaigns?: unknown;
        orphans?: unknown;
      };
      const newCampaigns: Campaign[] = Array.isArray(body.campaigns)
        ? body.campaigns
            .filter((c): c is Record<string, unknown> => typeof c === 'object' && c !== null)
            .map((c): Campaign => ({
              slug: typeof c.slug === 'string' ? c.slug : '',
              title: typeof c.title === 'string' ? c.title : '(untitled)',
              status: typeof c.status === 'string' ? c.status : '',
              tone: typeof c.tone === 'string' ? c.tone : '',
              lang: typeof c.lang === 'string' ? c.lang : '',
              sessions: Array.isArray(c.sessions)
                ? c.sessions
                    .filter((s): s is Record<string, unknown> => typeof s === 'object' && s !== null)
                    .map(parseSession)
                    .filter((s) => s.id)
                : []
            }))
            .filter((c) => c.slug)
        : [];
      const newOrphans: ResumeSession[] = Array.isArray(body.orphans)
        ? body.orphans
            .filter((s): s is Record<string, unknown> => typeof s === 'object' && s !== null)
            .map(parseSession)
            .filter((s) => s.id)
        : [];
      campaigns = newCampaigns;
      orphanSessions = newOrphans;
    } catch {
      /* server unreachable — keep prior state */
    }
  }

  function openResumeSheet(): void {
    resumeSheetOpen = true;
  }

  async function handleResumePick(choice: string): Promise<void> {
    resumeChoice = choice;
    // resumeSheetOpen is already false — ResumeSheet sets open=false before
    // firing the callback so the dialog closes immediately on click.
    await session.start({
      locale,
      resumeFrom: resumeFrom(),
      campaignId: selectedPickerCampaign,
      runId: selectedRun
    });
    resumeChoice = '';
    await loadCampaigns();
  }

  // Render DM markdown only on completed bubbles. Half-open ``*`` or
  // ``**`` during streaming would otherwise flicker between formatted
  // and unformatted states until the closing pair lands.
  function renderBubbleHtml(text: string): string {
    return renderMarkdown(text);
  }

  // Pull `[ROLL …]` markers out of the player's text and replace them
  // with a styled <span> so they jump visually. We escape the
  // surrounding text manually (no marked here — user bubbles are not
  // markdown) and only mark up the roll segments.
  const ROLL_MARKER_RE = /\[ROLL [^\]]+\]/g;
  function renderUserText(text: string): string {
    const escaped = escapeHtml(text);
    return escaped.replace(ROLL_MARKER_RE, (m) => `<span class="roll-chip">${m}</span>`);
  }
  function escapeHtml(s: string): string {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Auto-scroll only if the user is already pinned near the bottom.
  // Otherwise they're reading older turns and a streaming reply would
  // rip them back to the bottom mid-read.
  const SCROLL_PIN_THRESHOLD_PX = 80;
  $effect(() => {
    void transcript.length;
    void pendingBotText;
    const el = transcriptEl;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom <= SCROLL_PIN_THRESHOLD_PX) {
      el.scrollTop = el.scrollHeight;
    }
  });

  const session = new AidmTextSession({
    onStateChange: (next) => {
      sessionState = next;
      if (next !== 'error') errorMessage = null;
      if (next === 'ending') farewellHeard = false;
      if (next === 'idle') {
        transcript = [];
        pendingBotText = '';
        pendingBotSpeaker = { kind: 'dm' };
        inputText = '';
        sceneName = null;
        sceneDescription = '';
        // Clearing the slug cascades into CharacterPanel, which clears
        // its sheet / edit-buffer / prompt-modal state on its own.
        characterSlug = null;
        npcStates = [];
        debugTurns = [];
        memoryEvents = [];
        isGenerating = false;
        writerPhase = 'idle';
        queuedTurn = '';
        // Session ended (user pressed End or session.stop()); clear
        // the persisted id so the next page load doesn't try to
        // reattach to a dead pipeline.
        persistSessionId(null);
      } else if (next === 'error') {
        // Keep transcript + scene + character visible. A mobile lock
        // can drop the SSE and exhaust the backoff before the radio
        // is back — but the server pipeline is usually still alive,
        // so a visibilitychange wake-up or manual Start can recover.
        // Clearing context here would needlessly amputate the chat.
        pendingBotText = '';
        isGenerating = false;
        writerPhase = 'idle';
        // Deliberately NOT queuing across an error: the pipeline may
        // have died mid-writer, and firing a stale turn into a
        // recovered session out of context is worse than retyping it.
        queuedTurn = '';
      } else if (next === 'connected') {
        // Just connected — capture the live session id for refresh
        // survival. session.attach() also lands here; the id was
        // already in storage so this is effectively a no-op then.
        const liveId = session.currentSessionId;
        if (liveId) persistSessionId(liveId);
      }
    },
    onError: (message) => {
      errorMessage = message;
    },
    onMessage: handleMessage,
    onSessionContinued: () => {
      // The old session had ended while the phone was locked; a new one
      // continues the run. Keep the transcript, mark the seam.
      transcript = [
        ...transcript,
        {
          role: 'assistant',
          speaker: { kind: 'dm' },
          text: '*The connection was gone too long — this run continues in a new session from the last summary.*'
        }
      ];
      pendingBotText = '';
      isGenerating = false;
      writerPhase = 'idle';
      queuedTurn = '';
    },
    onHistorySync: (history) => {
      // Mobile-unlock / tab-resume path: server's LLMContext is the
      // authoritative transcript. Repaint from it so any events that
      // landed while the EventSource was severed (DM finished a turn
      // while phone was locked) show up after reconnect.
      transcript = history.turns.map((t) => ({
        role: t.role,
        speaker: t.role === 'user' ? { kind: 'user' as const } : { kind: 'dm' as const },
        text: t.text
      }));
      pendingBotText = '';
      pendingBotSpeaker = { kind: 'dm' };
      isGenerating = false;
      writerPhase = 'idle';
      queuedTurn = '';
    }
  });

  function handleMessage(message: DataChannelMessage): void {
    switch (message.type) {
      case 'transcript.user': {
        flushPendingBot();
        const text = typeof message.text === 'string' ? message.text.trim() : '';
        if (text && transcript.at(-1)?.text !== text) {
          transcript = [
            ...transcript,
            { role: 'user', speaker: { kind: 'user' }, text }
          ];
        }
        break;
      }
      case 'transcript.segment': {
        const chunk = typeof message.text === 'string' ? message.text : '';
        if (!chunk) break;
        const speaker = parseSpeaker((message as Record<string, unknown>).speaker);
        if (pendingBotText && !sameSpeaker(pendingBotSpeaker, speaker)) {
          flushPendingBot();
        }
        if (!pendingBotText) pendingBotSpeaker = speaker;
        pendingBotText += chunk;
        if (sessionState === 'ending' && chunk.trim()) farewellHeard = true;
        break;
      }
      case 'transcript.bot_end': {
        flushPendingBot();
        // Don't clear isGenerating here — the post-turn sheet writer is
        // still running. We wait for turn.status: idle so the user can't
        // fire the next turn while the server is mid-rewrite (the
        // pipeline would queue it but the UI would falsely look ready).
        if (sessionState === 'ending' && farewellHeard) {
          void session.stop();
        }
        break;
      }
      case 'turn.status': {
        const phase = typeof message.phase === 'string' ? message.phase : '';
        if (phase === 'writing_sheets') {
          writerPhase = 'writing_sheets';
        } else if (phase === 'idle') {
          writerPhase = 'idle';
          isGenerating = false;
          if (queuedTurn) {
            const text = queuedTurn;
            queuedTurn = '';
            dispatchTurn(text);
          }
        }
        break;
      }
      case 'scene.change': {
        const name = typeof message.name === 'string' ? message.name.trim() : '';
        const description =
          typeof message.description === 'string' ? message.description.trim() : '';
        if (name) {
          sceneName = name;
          sceneDescription = description;
        }
        break;
      }
      case 'character.change': {
        const slug = typeof message.slug === 'string' ? message.slug.trim() : '';
        // CharacterPanel watches characterSlug and fetches its own sheet.
        if (slug) characterSlug = slug;
        break;
      }
      case 'npc.state': {
        const raw = (message as { npcs?: unknown }).npcs;
        if (!Array.isArray(raw)) break;
        npcStates = raw
          .filter((n): n is Record<string, unknown> => typeof n === 'object' && n !== null)
          .map((n) => ({
            slug: typeof n.slug === 'string' ? n.slug : '',
            mood: typeof n.mood === 'string' ? n.mood : '',
            scene_goal: typeof n.scene_goal === 'string' ? n.scene_goal : '',
            social_mask: typeof n.social_mask === 'string' ? n.social_mask : ''
          }))
          .filter((n) => n.slug);
        break;
      }
      case 'beat.change': {
        const content = typeof message.content === 'string' ? message.content : '';
        // The DM rewrote the beat via vault_set_beat. Signal TuningPanel
        // to sync its internal state.
        externalBeat = content;
        break;
      }
      case 'debug.context': {
        const raw = (message as { messages?: unknown }).messages;
        const messages = Array.isArray(raw)
          ? (raw as { role?: unknown; text?: unknown }[]).map((m) => ({
              role: typeof m.role === 'string' ? m.role : '?',
              text: typeof m.text === 'string' ? m.text : ''
            }))
          : [];
        debugTurns = [
          ...debugTurns,
          { turn: debugTurns.length + 1, context: messages, reply: '' }
        ];
        break;
      }
      case 'debug.summary': {
        const m = message as {
          summary?: unknown;
          summarized_turns?: unknown;
          kept_turns?: unknown;
        };
        const summary = typeof m.summary === 'string' ? m.summary : '';
        const summarized = typeof m.summarized_turns === 'number' ? m.summarized_turns : 0;
        const kept = typeof m.kept_turns === 'number' ? m.kept_turns : 0;
        memoryEvents = [
          ...memoryEvents,
          { at: new Date(), summarized, kept, summary }
        ];
        break;
      }
      case 'debug.bot': {
        const raw = (message as { text?: unknown }).text;
        const text = typeof raw === 'string' ? raw : '';
        if (debugTurns.length === 0) {
          debugTurns = [...debugTurns, { turn: 1, context: [], reply: text }];
        } else {
          const updated = debugTurns.slice();
          updated[updated.length - 1] = { ...updated[updated.length - 1], reply: text };
          debugTurns = updated;
        }
        break;
      }
      default:
        break;
    }
  }

  function flushPendingBot(): void {
    const text = pendingBotText.trim();
    if (text)
      transcript = [
        ...transcript,
        { role: 'assistant', speaker: pendingBotSpeaker, text }
      ];
    pendingBotText = '';
    pendingBotSpeaker = { kind: 'dm' };
  }

  const isBusy = $derived(sessionState === 'connecting' || sessionState === 'disconnecting');
  const isConnected = $derived(sessionState === 'connected');
  const isEnding = $derived(sessionState === 'ending');

  // Pill tone — coarse classification so the StatusPill doesn't have to
  // enumerate transport-specific labels. ``generating`` overlays the
  // normal state when the LLM is mid-stream, so it gets ``busy`` too.
  const statusLabel = $derived(
    !isGenerating
      ? sessionState
      : writerPhase === 'writing_sheets'
        ? 'updating sheets'
        : 'generating'
  );
  const statusTone = $derived(
    isGenerating
      ? ('busy' as const)
      : sessionState === 'connected'
        ? ('connected' as const)
        : sessionState === 'connecting' ||
            sessionState === 'reconnecting' ||
            sessionState === 'ending'
          ? ('busy' as const)
          : sessionState === 'error'
            ? ('error' as const)
            : ('idle' as const)
  );

  // Primary action label — pre-resolve so the template stays legible.
  function primaryButtonLabel(): string {
    if (isConnected) return 'End';
    if (isEnding) return 'Wrapping up…';
    if (sessionState === 'reconnecting') return 'Reconnecting…';
    if (sessionState === 'connecting') return 'Connecting…';
    if (sessionState === 'disconnecting') return 'Disconnecting…';
    return 'Start';
  }
  const hasInfo = $derived(sceneName !== null || characterSlug !== null);

  async function toggle() {
    if (isConnected) {
      await session.end();
      return;
    }
    if (isEnding) {
      await session.stop();
      return;
    }
    if (sessionState === 'connecting') {
      await session.stop();
      return;
    }
    await session.start({
      locale,
      resumeFrom: resumeFrom(),
      campaignId: selectedPickerCampaign,
      runId: selectedRun
    });
    // Clear resume after a successful start — the next Start should
    // not re-seed unless the user explicitly picks again.
    resumeChoice = '';
    // Refresh the list so the just-started session shows up in the
    // picker as soon as its first summary lands.
    await loadCampaigns();
  }

  let isSuggesting = $state<boolean>(false);
  let suggestError = $state<string>('');

  async function suggestPlayerTurn() {
    if (!isConnected || isGenerating || isSuggesting) return;
    const sid = session.currentSessionId;
    if (!sid) return;
    isSuggesting = true;
    suggestError = '';
    try {
      const res = await fetch(`/api/text/session/${sid}/suggest-player-turn`, {
        method: 'POST'
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as { text?: string };
      const text = (data.text ?? '').trim();
      if (!text) throw new Error('empty suggestion');
      inputText = text;
    } catch (err) {
      suggestError = err instanceof Error ? err.message : 'suggestion failed';
    } finally {
      isSuggesting = false;
    }
  }

  // Submit is open while the sheet writer runs (the turn queues); it is
  // closed while DM tokens are streaming or a turn is already queued.
  const canSend = $derived(
    isConnected && !queuedTurn && (!isGenerating || writerPhase === 'writing_sheets')
  );

  // ↻ on the last DM reply: only once that turn has fully landed (writer
  // idle), never while ending — the farewell turn is not regenerated.
  const canRegenerate = $derived(
    isConnected && !isGenerating && writerPhase === 'idle' && !queuedTurn && !pendingBotText
  );

  async function regenerateLast(): Promise<void> {
    if (!canRegenerate) return;
    // One DM reply may be split into several entries by speaker: drop
    // everything after the player line it answered.
    let lastUser = transcript.length - 1;
    while (lastUser >= 0 && transcript[lastUser].role !== 'user') lastUser--;
    if (lastUser < 0) return;
    const previous = transcript;
    transcript = transcript.slice(0, lastUser + 1);
    isGenerating = true;
    writerPhase = 'streaming';
    if (!(await session.regenerate())) {
      transcript = previous;
      isGenerating = false;
      writerPhase = 'idle';
    }
  }

  function dispatchTurn(text: string): void {
    isGenerating = true;
    writerPhase = 'streaming';
    void session.send(text);
  }

  async function sendInput(event?: SubmitEvent) {
    event?.preventDefault();
    const raw = inputText.trim();
    if (!raw || !canSend) return;
    // Roll any `/roll dXX+Y` specs inline before send. The substituted
    // text (with `[ROLL ...]` markers) is what both the player sees in
    // the transcript and what the DM receives — same string, no
    // server-side roll engine to keep in sync.
    const { text: rolled } = replaceRollsInline(raw);
    transcript = [
      ...transcript,
      { role: 'user', speaker: { kind: 'user' }, text: rolled }
    ];
    inputText = '';
    if (writerPhase === 'writing_sheets') {
      queuedTurn = rolled;
      return;
    }
    dispatchTurn(rolled);
  }

  onDestroy(() => {
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', handleVisibility);
    }
    if (typeof window !== 'undefined') {
      window.removeEventListener('online', handleOnline);
    }
    void session.stop();
  });
</script>

<svelte:head>
  <title>AIDM — text DM</title>
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover" />
</svelte:head>

<div class="bg-background text-foreground flex min-h-dvh flex-col">
  <!-- Sticky app bar: brand, mode chip, transport nav, view toggles. -->
  <header class="border-border bg-background/95 sticky top-0 z-30 flex flex-wrap items-center gap-3 border-b px-4 py-2 backdrop-blur">
    <h1 class="font-mono text-sm font-semibold tracking-widest uppercase">AIDM</h1>
    <ModeBadge mode={serverMode} tone={toneText} />

    <nav class="ml-auto flex items-center gap-2 font-mono text-[0.65rem] uppercase tracking-widest" aria-label="Transport mode">
      <a href="/" class="text-muted-foreground hover:text-primary transition-colors">voice</a>
      <span class="text-primary" aria-current="page">text</span>
    </nav>

    <Button
      variant="outline"
      size="xs"
      onclick={() => (infoOpen = !infoOpen)}
      aria-pressed={infoOpen}
      aria-label="Toggle scene and character"
      class="font-mono"
    >
      <span aria-hidden="true">{hasInfo ? '◑' : '○'}</span>
      <span class="ml-1">Info</span>
    </Button>

    <label class="border-border hover:bg-muted/40 flex h-6 cursor-pointer items-center gap-1 rounded-md border px-2 font-mono text-[0.65rem]">
      <input type="checkbox" bind:checked={debugEnabled} class="size-3" />
      <span aria-hidden="true">{debugEnabled ? '⏺' : '◯'}</span>
      <span>Debug</span>
    </label>
  </header>

  <!-- Session bar: picker + locale + resume + status + start/end. -->
  <section class="border-border flex flex-wrap items-end gap-2 border-b px-4 py-2" aria-label="Session controls">
    <RunPicker
      campaigns={pickerCampaigns}
      runs={pickerRuns}
      selectedCampaign={selectedPickerCampaign}
      {selectedRun}
      disabled={isBusy || isConnected}
      {creatingRun}
      onCampaignChange={onPickerCampaignChange}
      onRunChange={onPickerRunChange}
      onSubmitNewRun={submitNewRun}
    />

    {#if resumeAvailable && !isConnected}
      <Button variant="outline" size="sm" onclick={openResumeSheet} disabled={isBusy}>
        📚 Resume
      </Button>
    {/if}

    <StatusPill label={statusLabel} tone={statusTone} />

    {#if memoryEvents.length > 0}
      <Badge
        variant="secondary"
        class="font-mono text-[0.65rem]"
        title="Auto session-summary {memoryEvents.length} ×"
      >
        📜 {memoryEvents.length}
      </Badge>
    {/if}

    <Button
      class="ml-auto rounded-pill px-5"
      variant={isConnected || isEnding ? 'destructive' : 'default'}
      onclick={toggle}
      disabled={sessionState === 'disconnecting'}
    >
      {primaryButtonLabel()}
    </Button>
  </section>

  {#if errorMessage}
    <div
      class="border-destructive/40 bg-destructive/10 text-destructive mx-4 mt-2 rounded-md border px-3 py-2 font-mono text-sm"
      role="alert"
    >
      {errorMessage}
    </div>
  {/if}

  <!-- Main layout: transcript + composer (full width on mobile) + side info (sheet on mobile). -->
  <main class="grid flex-1 min-h-0 grid-cols-1 md:grid-cols-[1fr_320px]">
    <section class="flex min-h-0 flex-col">
      <ol class="flex-1 space-y-3 overflow-y-auto px-4 py-4" bind:this={transcriptEl}>
        {#if transcript.length === 0 && !pendingBotText}
          <li class="text-muted-foreground text-center italic text-sm py-8">
            {#if isConnected}Type to begin…{:else}Start a session to see the transcript.{/if}
          </li>
        {:else}
          {#each transcript as entry, i (i)}
            <li
              class="grid grid-cols-[3rem_1fr] items-baseline gap-3 text-[0.95rem] leading-relaxed"
            >
              <span
                class={'pt-0.5 text-right font-mono text-[0.65rem] uppercase tracking-wider ' +
                  (entry.speaker.kind === 'npc' ? 'text-primary' : 'text-muted-foreground')}
              >
                {speakerLabel(entry.speaker)}
              </span>
              {#if entry.role === 'assistant'}
                <div
                  class={'markdown ' +
                    (entry.speaker.kind === 'npc' ? 'text-primary' : 'text-ink italic')}
                >
                  {#if entry.speaker.kind === 'npc' && npcMoodBySlug[entry.speaker.slug]}
                    <span class="text-muted-foreground mb-1 block font-mono text-[0.65rem] uppercase tracking-wider italic md:hidden">
                      mood · {npcMoodBySlug[entry.speaker.slug]}
                    </span>
                  {/if}
                  {@html renderBubbleHtml(entry.text)}
                  {#if i === transcript.length - 1 && !pendingBotText && isConnected}
                    <button
                      type="button"
                      class="text-muted-foreground hover:text-foreground not-italic mt-1 block rounded px-1 font-mono text-xs disabled:opacity-40"
                      aria-label="Regenerate the last DM reply"
                      title="Regenerate this reply"
                      disabled={!canRegenerate}
                      onclick={() => void regenerateLast()}
                    >↻</button>
                  {/if}
                </div>
              {:else}
                <div class="text-foreground">{@html renderUserText(entry.text)}</div>
              {/if}
            </li>
          {/each}
          {#if pendingBotText}
            <li class="grid grid-cols-[3rem_1fr] items-baseline gap-3 text-[0.95rem] leading-relaxed">
              <span
                class={'pt-0.5 text-right font-mono text-[0.65rem] uppercase tracking-wider ' +
                  (pendingBotSpeaker.kind === 'npc' ? 'text-primary' : 'text-muted-foreground')}
              >
                {speakerLabel(pendingBotSpeaker)}
              </span>
              <div class={pendingBotSpeaker.kind === 'npc' ? 'text-primary' : 'text-ink italic'}>
                {#if pendingBotSpeaker.kind === 'npc' && npcMoodBySlug[pendingBotSpeaker.slug]}
                  <span class="text-muted-foreground mb-1 block font-mono text-[0.65rem] uppercase tracking-wider italic md:hidden">
                    mood · {npcMoodBySlug[pendingBotSpeaker.slug]}
                  </span>
                {/if}
                {pendingBotText}<span class="text-primary ml-0.5 animate-[blink_0.85s_steps(1)_infinite]">▍</span>
              </div>
            </li>
          {:else if isGenerating}
            <li
              class="grid grid-cols-[3rem_1fr] items-baseline gap-3 text-[0.95rem] leading-relaxed"
              aria-live="polite"
            >
              <span class="text-muted-foreground pt-0.5 text-right font-mono text-[0.65rem] uppercase tracking-wider">
                {writerPhase === 'writing_sheets' ? 'sys' : 'DM'}
              </span>
              <span
                class="typing inline-flex items-center gap-1"
                aria-label={writerPhase === 'writing_sheets' ? 'Updating scene sheets' : 'DM is thinking'}
              >
                <span></span><span></span><span></span>
                {#if writerPhase === 'writing_sheets'}
                  <span class="text-muted-foreground ml-2 font-mono text-[0.65rem] uppercase tracking-wider italic">
                    updating sheets…
                  </span>
                {/if}
              </span>
            </li>
          {/if}
        {/if}
      </ol>

      <form
        class="border-border bg-background flex flex-col gap-1 border-t px-4 py-3"
        onsubmit={sendInput}
      >
        <div class="flex items-stretch gap-2">
          <textarea
            class="bg-card border-border text-foreground placeholder:text-muted-foreground flex-1 resize-none rounded-md border px-3 py-2 font-sans text-sm focus-visible:outline-none disabled:opacity-50"
            bind:value={inputText}
            placeholder={isConnected ? 'Your move… (try /roll d20+3 inline)' : 'Start a session first'}
            disabled={!isConnected}
            rows="1"
            onkeydown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void sendInput();
              }
            }}
          ></textarea>
          <Button
            type="button"
            variant="outline"
            disabled={!isConnected || isGenerating || isSuggesting}
            onclick={() => void suggestPlayerTurn()}
            aria-label="Suggest in-character action from sheet"
            title="Suggest in-character action from sheet"
            class="h-auto"
          >
            {isSuggesting ? '…' : '✨'}
          </Button>
          <Button
            type="submit"
            disabled={!canSend || !inputText.trim()}
            aria-label="Send"
            class="h-auto"
          >
            ➤
          </Button>
        </div>
        {#if suggestError}
          <p class="text-destructive px-1 text-xs" role="alert">
            Suggest failed: {suggestError}
            <button
              type="button"
              class="ml-2 underline"
              onclick={() => (suggestError = '')}
            >dismiss</button>
          </p>
        {/if}
      </form>
    </section>

    <!-- Desktop side panel — Sheet handles mobile separately below. -->
    <aside class="border-border hidden border-l md:flex md:flex-col">
      <div class="flex h-full flex-col overflow-y-auto">
        <Card.Root class="m-3">
          <Card.Header>
            <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
              Scene
            </Card.Title>
          </Card.Header>
          <Card.Content>
            {#if sceneName}
              <p class="text-foreground text-base font-semibold mb-2">{sceneName}</p>
              {#if sceneDescription}
                <p class="text-ink italic text-sm leading-relaxed">{sceneDescription}</p>
              {/if}
            {:else}
              <p class="text-muted-foreground italic text-sm">No scene yet.</p>
            {/if}
          </Card.Content>
        </Card.Root>

        <Card.Root class="m-3 mt-0">
          <Card.Header>
            <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
              NPCs in scene
            </Card.Title>
          </Card.Header>
          <Card.Content>
            {#if npcStates.length === 0}
              <p class="text-muted-foreground italic text-sm">None active.</p>
            {:else}
              <ul class="space-y-2">
                {#each npcStates as npc (npc.slug)}
                  <li class="text-sm">
                    <p class="text-primary font-mono">{npc.slug}</p>
                    {#if npc.mood}
                      <p class="text-foreground text-xs"><span class="text-muted-foreground">mood:</span> {npc.mood}</p>
                    {/if}
                    {#if npc.scene_goal}
                      <p class="text-foreground text-xs"><span class="text-muted-foreground">goal:</span> {npc.scene_goal}</p>
                    {/if}
                    {#if npc.social_mask}
                      <p class="text-foreground text-xs"><span class="text-muted-foreground">mask:</span> {npc.social_mask}</p>
                    {/if}
                  </li>
                {/each}
              </ul>
            {/if}
          </Card.Content>
        </Card.Root>

        <CharacterPanel
          {characterSlug}
          {selectedPickerCampaign}
          {selectedRun}
          {isConnected}
        />

        <details class="border-border m-3 mt-0 rounded-md border bg-card">
          <summary class="text-muted-foreground cursor-pointer px-3 py-2 font-mono text-xs uppercase tracking-wider">
            Tone &amp; beat
          </summary>
          <TuningPanel
            {selectedPickerCampaign}
            {selectedRun}
            {externalBeat}
            {isConnected}
            ontoneloaded={(text) => (toneText = text)}
          />
        </details>
      </div>
    </aside>
  </main>

  <!-- Mobile-only side sheet that mirrors the desktop aside. -->
  <Sheet.Root bind:open={infoOpen}>
    <Sheet.Content side="right" class="w-[min(92vw,420px)] md:hidden">
      <Sheet.Header>
        <Sheet.Title>Scene &amp; Character</Sheet.Title>
      </Sheet.Header>
      <div class="space-y-4 overflow-y-auto p-4">
        <Card.Root>
          <Card.Header>
            <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
              Scene
            </Card.Title>
          </Card.Header>
          <Card.Content>
            {#if sceneName}
              <p class="text-foreground text-base font-semibold mb-2">{sceneName}</p>
              {#if sceneDescription}
                <p class="text-ink italic text-sm leading-relaxed">{sceneDescription}</p>
              {/if}
            {:else}
              <p class="text-muted-foreground italic text-sm">No scene yet.</p>
            {/if}
          </Card.Content>
        </Card.Root>

        <CharacterPanel
          {characterSlug}
          {selectedPickerCampaign}
          {selectedRun}
          {isConnected}
        />

        <details class="border-border rounded-md border bg-card">
          <summary class="text-muted-foreground cursor-pointer px-3 py-2 font-mono text-xs uppercase tracking-wider">
            Tone &amp; beat
          </summary>
          <TuningPanel
            {selectedPickerCampaign}
            {selectedRun}
            {externalBeat}
            {isConnected}
            ontoneloaded={(text) => (toneText = text)}
          />
        </details>
      </div>
    </Sheet.Content>
  </Sheet.Root>

  <!-- Debug panel: collapsible by the appbar toggle. Memory consolidations
       + per-turn context/reply dumps. -->
  {#if debugEnabled}
    <section class="border-border border-t" aria-label="Debug">
      <Card.Root class="m-3">
        <Card.Header class="flex flex-row items-center gap-3">
          <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono flex-1">
            Debug
          </Card.Title>
          <span class="text-muted-foreground font-mono text-xs">
            {debugTurns.length} turn{debugTurns.length === 1 ? '' : 's'}
          </span>
          {#if debugTurns.length > 0}
            <Button size="xs" variant="ghost" onclick={() => (debugTurns = [])}>clear</Button>
          {/if}
        </Card.Header>
        <Card.Content class="space-y-3">
          {#if memoryEvents.length > 0}
            <h3 class="text-muted-foreground font-mono text-[0.65rem] uppercase tracking-wider">
              📜 Memory consolidations
            </h3>
            <ol class="space-y-2">
              {#each memoryEvents as ev, i (i)}
                <li>
                  <details class="bg-card border-border rounded-md border p-2">
                    <summary class="cursor-pointer text-xs">
                      #{i + 1} — compressed {ev.summarized} turn{ev.summarized === 1 ? '' : 's'},
                      kept last {ev.kept} · {ev.at.toLocaleTimeString()}
                    </summary>
                    <pre class="text-muted-foreground mt-2 overflow-x-auto whitespace-pre-wrap font-mono text-xs">{ev.summary}</pre>
                  </details>
                </li>
              {/each}
            </ol>
          {/if}
          {#if debugTurns.length === 0}
            <p class="text-muted-foreground italic text-sm">No turns yet.</p>
          {:else}
            <ol class="space-y-2">
              {#each debugTurns as entry (entry.turn)}
                <li>
                  <details class="bg-card border-border rounded-md border p-2">
                    <summary class="cursor-pointer text-xs">
                      Turn {entry.turn} — {entry.context.length} ctx · {entry.reply.length} reply
                    </summary>
                    <h4 class="text-muted-foreground mt-2 font-mono text-[0.65rem] uppercase tracking-wider">
                      Context
                    </h4>
                    <ol class="mt-1 space-y-1">
                      {#each entry.context as msg, i (i)}
                        <li class="bg-background rounded p-2">
                          <span class="text-primary font-mono text-[0.65rem] uppercase">{msg.role}</span>
                          <pre class="text-muted-foreground mt-1 overflow-x-auto whitespace-pre-wrap font-mono text-xs">{msg.text}</pre>
                        </li>
                      {/each}
                    </ol>
                    <h4 class="text-muted-foreground mt-2 font-mono text-[0.65rem] uppercase tracking-wider">
                      Reply
                    </h4>
                    <pre class="text-ink mt-1 overflow-x-auto whitespace-pre-wrap font-mono text-xs">{entry.reply || '(streaming…)'}</pre>
                  </details>
                </li>
              {/each}
            </ol>
          {/if}
        </Card.Content>
      </Card.Root>
    </section>
  {/if}
</div>

<!-- Resume picker dialog — campaigns + sessions + preview in three panes. -->
<ResumeSheet
  bind:open={resumeSheetOpen}
  {campaigns}
  {orphanSessions}
  onresumepick={handleResumePick}
/>

<style>
  /* Local: blink keyframe used by the streaming caret. */
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  /* Three-dot "DM is thinking" indicator. Each dot pulses with a 0.4s
   * stagger so the indicator reads as motion, not three discrete blobs. */
  .typing span {
    display: inline-block;
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 9999px;
    background: var(--muted-foreground);
    animation: typing-pulse 1.2s ease-in-out infinite;
  }
  .typing span:nth-child(2) {
    animation-delay: 0.2s;
  }
  .typing span:nth-child(3) {
    animation-delay: 0.4s;
  }
  @keyframes typing-pulse {
    0%, 80%, 100% {
      transform: scale(0.6);
      opacity: 0.4;
    }
    40% {
      transform: scale(1);
      opacity: 1;
    }
  }

  /* Marked-rendered assistant-bubble markdown. DM output is server-authored,
   * so :global() selectors here only style markup we control. */
  .markdown {
    color: var(--foreground);
    font-size: 0.9rem;
    line-height: 1.55;
  }
  .markdown :global(h1),
  .markdown :global(h2),
  .markdown :global(h3) {
    margin: 0.5rem 0 0.4rem;
    color: var(--primary);
    font-size: 0.95rem;
    letter-spacing: 0.02em;
  }
  .markdown :global(h1):first-child,
  .markdown :global(h2):first-child {
    margin-top: 0;
  }
  .markdown :global(p) {
    margin: 0.35rem 0;
  }
  .markdown :global(ul),
  .markdown :global(ol) {
    margin: 0.35rem 0;
    padding-left: 1.25rem;
  }
  .markdown :global(li) {
    margin: 0.1rem 0;
  }
  .markdown :global(code) {
    background: var(--card);
    border-radius: 3px;
    padding: 0.05rem 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }
  .markdown :global(hr) {
    border: 0;
    border-top: 1px solid var(--border);
    margin: 0.6rem 0;
  }
  .markdown :global(strong) {
    color: var(--foreground);
  }
  .markdown :global(em) {
    color: var(--ink);
  }

  /* Dice-roll marker — rendered inline via dice.ts when /roll appears. */
  .markdown :global(.roll) {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.85em;
    color: var(--primary);
  }
</style>
