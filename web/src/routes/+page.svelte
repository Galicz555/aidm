<svelte:options runes={true} />

<script lang="ts">
  // Voice UI: campaign/run picker, locale, Start/End, status pill, hidden
  // <audio> for DM voice, live transcript + scene + character side panel.
  //
  // Visual layer: Tailwind v4 + shadcn-svelte primitives. The session
  // lifecycle, vault fetch, and run-create flows live in $lib (webrtc.ts,
  // character.ts, runs.ts) — this file is mostly state + template.

  import { onDestroy, onMount } from 'svelte';
  import { renderMarkdown } from '$lib/markdown';
  import { toast } from 'svelte-sonner';
  import { AidmSession, type DataChannelMessage, type SessionState } from '$lib/webrtc';
  import {
    fetchCharacter,
    updateCharacter,
    fetchRoleplayPrompt,
  } from '$lib/character';
  import {
    listCampaigns,
    listRuns,
    createRun,
    type Campaign,
    type Run,
  } from '$lib/runs';

  import * as Card from '$lib/components/ui/card';
  import * as Dialog from '$lib/components/ui/dialog';
  import * as Select from '$lib/components/ui/select';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import { Skeleton } from '$lib/components/ui/skeleton';
  import ModeBadge from '$lib/components/aidm/ModeBadge.svelte';
  import RunPicker from '$lib/components/aidm/RunPicker.svelte';
  import StatusPill from '$lib/components/aidm/StatusPill.svelte';

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
  let sessionState = $state<SessionState>('idle');
  let errorMessage = $state<string | null>(null);
  let audioEl = $state<HTMLAudioElement | null>(null);
  let remoteStream = $state<MediaStream | null>(null);
  let transcript = $state<TranscriptEntry[]>([]);
  let pendingBotText = $state<string>('');
  let pendingBotSpeaker = $state<Speaker>({ kind: 'dm' });
  let sceneName = $state<string | null>(null);
  let sceneDescription = $state<string>('');
  let characterSlug = $state<string | null>(null);
  let characterSheet = $state<string | null>(null);
  let characterLastModified = $state<string | null>(null);
  let characterError = $state<string | null>(null);
  let isEditingCharacter = $state<boolean>(false);
  let editBuffer = $state<string>('');
  let editError = $state<string | null>(null);
  let editSaving = $state<boolean>(false);
  let promptModalOpen = $state<boolean>(false);
  let promptText = $state<string>('');
  let promptError = $state<string | null>(null);
  let promptCopied = $state<boolean>(false);

  let serverMode = $state<'public' | 'private' | 'unknown'>('unknown');

  // Campaign + run picker. Selection survives reload via localStorage so
  // the player doesn't re-pick on every page open.
  let campaigns = $state<Campaign[]>([]);
  let selectedCampaign = $state<string>('');
  let runs = $state<Run[]>([]);
  let selectedRun = $state<string>('');
  let creatingRun = $state<boolean>(false);

  const LS_CAMPAIGN = 'aidm:campaign';
  const LS_RUN = 'aidm:run';

  async function loadCampaigns(): Promise<void> {
    try {
      campaigns = await listCampaigns();
      const stored = localStorage.getItem(LS_CAMPAIGN) ?? '';
      const initial =
        campaigns.find((c) => c.slug === stored)?.slug ?? campaigns[0]?.slug ?? '';
      selectedCampaign = initial;
      if (initial) await loadRuns(initial);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadRuns(campaignSlug: string): Promise<void> {
    try {
      runs = await listRuns(campaignSlug);
      const stored = localStorage.getItem(LS_RUN) ?? '';
      const initial =
        runs.find((r) => r.slug === stored)?.slug ?? runs[0]?.slug ?? '';
      selectedRun = initial;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    }
  }

  async function onCampaignChange(value: string): Promise<void> {
    selectedCampaign = value;
    localStorage.setItem(LS_CAMPAIGN, value);
    if (value) await loadRuns(value);
  }

  function onRunChange(value: string): void {
    selectedRun = value;
    if (value) localStorage.setItem(LS_RUN, value);
  }

  async function submitNewRun(title: string): Promise<void> {
    if (!selectedCampaign) return;
    creatingRun = true;
    try {
      const created = await createRun(selectedCampaign, { title });
      await loadRuns(selectedCampaign);
      selectedRun = created.slug;
      localStorage.setItem(LS_RUN, selectedRun);
      toast.success(`Run created: ${created.title}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      creatingRun = false;
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
      /* server may not be up — leave unknown */
    }
    await loadCampaigns();
  });

  // The sheet markdown is rendered to HTML once per change. The text can
  // come from the LLM or a PUT on the vault, so renderMarkdown sanitises it
  // before it is injected via {@html ...}.
  const characterSheetHtml = $derived(characterSheet ? renderMarkdown(characterSheet) : '');

  const session = new AidmSession({
    onStateChange: (next) => {
      sessionState = next;
      if (next !== 'error') errorMessage = null;
      if (next === 'idle' || next === 'error') {
        transcript = [];
        pendingBotText = '';
        pendingBotSpeaker = { kind: 'dm' };
        sceneName = null;
        sceneDescription = '';
        characterSlug = null;
        characterSheet = null;
        characterLastModified = null;
        characterError = null;
        isEditingCharacter = false;
        editBuffer = '';
        editError = null;
        promptModalOpen = false;
        promptText = '';
      }
    },
    onError: (message) => {
      errorMessage = message;
    },
    onRemoteTrack: (stream) => {
      remoteStream = stream;
    },
    onMessage: handleMessage
  });

  function handleMessage(message: DataChannelMessage): void {
    switch (message.type) {
      case 'transcript.user': {
        flushPendingBot();
        const text = typeof message.text === 'string' ? message.text.trim() : '';
        if (text)
          transcript = [
            ...transcript,
            { role: 'user', speaker: { kind: 'user' }, text }
          ];
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
        break;
      }
      case 'transcript.bot_end': {
        flushPendingBot();
        if (sessionState === 'ending') {
          void session.stop();
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
        if (slug) void loadCharacter(slug);
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

  async function loadCharacter(slug: string): Promise<void> {
    characterSlug = slug;
    characterError = null;
    characterSheet = null;
    characterLastModified = null;
    isEditingCharacter = false;
    editError = null;
    try {
      const { markdown, lastModified } = await fetchCharacter(
        slug,
        selectedCampaign,
        selectedRun
      );
      characterSheet = markdown;
      characterLastModified = lastModified;
    } catch (err) {
      characterError = err instanceof Error ? err.message : String(err);
    }
  }

  function startEditCharacter(): void {
    if (!characterSheet) return;
    editBuffer = characterSheet;
    editError = null;
    isEditingCharacter = true;
  }

  function cancelEditCharacter(): void {
    isEditingCharacter = false;
    editBuffer = '';
    editError = null;
  }

  async function saveEditCharacter(): Promise<void> {
    if (!characterSlug) return;
    editSaving = true;
    editError = null;
    try {
      const result = await updateCharacter(
        characterSlug,
        selectedCampaign,
        selectedRun,
        editBuffer,
        characterLastModified
      );
      if (result.ok) {
        isEditingCharacter = false;
        editBuffer = '';
        await loadCharacter(characterSlug);
      } else if (result.conflict) {
        editError =
          'The DM wrote to this sheet while you were editing. Reloading…';
        await loadCharacter(characterSlug);
      } else {
        editError = result.message;
      }
    } catch (err) {
      editError = err instanceof Error ? err.message : String(err);
    } finally {
      editSaving = false;
    }
  }

  async function openRoleplayPrompt(): Promise<void> {
    if (!characterSlug) return;
    promptModalOpen = true;
    promptError = null;
    promptText = '';
    promptCopied = false;
    try {
      promptText = await fetchRoleplayPrompt(characterSlug, selectedCampaign, selectedRun);
    } catch (err) {
      promptError = err instanceof Error ? err.message : String(err);
    }
  }

  async function copyRoleplayPrompt(): Promise<void> {
    if (!promptText) return;
    try {
      await navigator.clipboard.writeText(promptText);
      promptCopied = true;
    } catch {
      promptCopied = false;
    }
  }

  // Wire the remote MediaStream to the audio element whenever either
  // side updates.
  $effect(() => {
    if (audioEl && remoteStream) {
      audioEl.srcObject = remoteStream;
    }
  });

  const isBusy = $derived(
    sessionState === 'requesting-mic' ||
      sessionState === 'connecting' ||
      sessionState === 'disconnecting'
  );
  const isConnected = $derived(sessionState === 'connected');
  const isEnding = $derived(sessionState === 'ending');

  // Coarse tone classification for the status pill — separate from the
  // displayed label so the pill component doesn't have to enumerate
  // every transport-specific state name.
  const statusTone = $derived(
    sessionState === 'connected'
      ? ('connected' as const)
      : sessionState === 'connecting' ||
          sessionState === 'requesting-mic' ||
          sessionState === 'reconnecting' ||
          sessionState === 'ending'
        ? ('busy' as const)
        : sessionState === 'error'
          ? ('error' as const)
          : ('idle' as const)
  );

  async function toggle() {
    if (isConnected) {
      await session.end();
      return;
    }
    if (isEnding) {
      await session.stop();
      return;
    }
    if (sessionState === 'connecting' || sessionState === 'requesting-mic') {
      await session.stop();
      return;
    }
    if (!selectedRun) {
      errorMessage = 'Pick a campaign and run before starting a session.';
      return;
    }
    await session.start({ locale, campaignId: selectedCampaign, runId: selectedRun });
  }

  function primaryButtonLabel(): string {
    if (isConnected) return 'End session';
    if (isEnding) return 'Wrapping up… (click to cut)';
    if (sessionState === 'reconnecting') return 'Reconnecting…';
    if (sessionState === 'connecting') return 'Connecting…';
    if (sessionState === 'requesting-mic') return 'Requesting mic…';
    if (sessionState === 'disconnecting') return 'Disconnecting…';
    return 'Start session';
  }

  onDestroy(() => {
    void session.stop();
  });
</script>

<svelte:head>
  <title>AIDM — voice DM</title>
</svelte:head>

<main class="mx-auto flex max-w-5xl flex-col gap-6 px-6 pt-8 pb-16">
  <header class="flex flex-wrap items-baseline gap-x-4 gap-y-2">
    <h1 class="text-3xl font-semibold tracking-tight">AIDM</h1>
    <p class="text-muted-foreground text-sm">Voice-first AI Dungeon Master</p>

    <nav class="ml-auto flex items-center gap-3 font-mono text-xs uppercase tracking-widest">
      <span class="text-primary">voice</span>
      <a href="/text" class="text-muted-foreground hover:text-primary transition-colors">
        text
      </a>
    </nav>

    <ModeBadge mode={serverMode} />
  </header>

  <!-- Session controls: campaign + run picker + locale + Start/End + status. -->
  <Card.Root>
    <Card.Content class="flex flex-wrap items-end gap-3">
      <RunPicker
        {campaigns}
        {runs}
        {selectedCampaign}
        {selectedRun}
        disabled={isBusy || isConnected}
        {creatingRun}
        onCampaignChange={onCampaignChange}
        onRunChange={onRunChange}
        onSubmitNewRun={submitNewRun}
      />

      <div class="flex flex-col gap-1">
        <Label class="text-muted-foreground text-[0.65rem] uppercase tracking-wider">
          Locale
        </Label>
        <Select.Root
          type="single"
          value={locale}
          onValueChange={(v) => (locale = v as 'hu' | 'en')}
          disabled={isBusy || isConnected}
        >
          <Select.Trigger class="h-8 min-w-[110px] text-sm">
            {locale === 'hu' ? 'Magyar' : 'English'}
          </Select.Trigger>
          <Select.Content>
            <Select.Item value="hu">Magyar</Select.Item>
            <Select.Item value="en">English</Select.Item>
          </Select.Content>
        </Select.Root>
      </div>

      <Button
        size="lg"
        variant={isConnected || isEnding ? 'destructive' : 'default'}
        onclick={toggle}
        disabled={sessionState === 'requesting-mic' || sessionState === 'disconnecting'}
        class="rounded-pill px-6"
      >
        {primaryButtonLabel()}
      </Button>

      <div class="ml-auto">
        <StatusPill label={sessionState} tone={statusTone} />
      </div>
    </Card.Content>
  </Card.Root>

  {#if errorMessage}
    <div
      class="border-destructive/40 bg-destructive/10 text-destructive rounded-md border px-4 py-3 font-mono text-sm"
      role="alert"
    >
      {errorMessage}
    </div>
  {/if}

  <!-- Transcript (2/3) + side panels (1/3) on desktop; stacked on mobile. -->
  <section class="grid grid-cols-1 gap-4 md:grid-cols-[2fr_1fr]">
    <Card.Root>
      <Card.Header>
        <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
          Transcript
        </Card.Title>
      </Card.Header>
      <Card.Content>
        {#if transcript.length === 0 && !pendingBotText}
          <p class="text-muted-foreground italic text-sm">
            {#if isConnected}
              Waiting for the first turn…
            {:else}
              Start a session to see the transcript.
            {/if}
          </p>
        {:else}
          <ScrollArea class="h-[60vh] pr-3">
            <ol class="flex flex-col gap-3">
              {#each transcript as entry, i (i)}
                <li class="grid grid-cols-[4rem_1fr] items-baseline gap-3 text-[0.95rem] leading-relaxed">
                  <span
                    class="text-right font-mono text-[0.65rem] uppercase tracking-wider pt-0.5
                           {entry.speaker.kind === 'npc' ? 'text-primary' : 'text-muted-foreground'}"
                  >
                    {speakerLabel(entry.speaker)}
                  </span>
                  <span
                    class={entry.role === 'user'
                      ? 'text-foreground'
                      : entry.speaker.kind === 'npc'
                        ? 'text-primary'
                        : 'text-ink italic'}
                  >
                    {entry.text}
                  </span>
                </li>
              {/each}
              {#if pendingBotText}
                <li class="grid grid-cols-[4rem_1fr] items-baseline gap-3 text-[0.95rem] leading-relaxed">
                  <span
                    class="text-right font-mono text-[0.65rem] uppercase tracking-wider pt-0.5
                           {pendingBotSpeaker.kind === 'npc' ? 'text-primary' : 'text-muted-foreground'}"
                  >
                    {speakerLabel(pendingBotSpeaker)}
                  </span>
                  <span class={pendingBotSpeaker.kind === 'npc' ? 'text-primary' : 'text-ink italic'}>
                    {pendingBotText}<span class="text-primary ml-0.5 animate-[blink_0.85s_steps(1)_infinite]">▍</span>
                  </span>
                </li>
              {/if}
            </ol>
          </ScrollArea>
        {/if}
      </Card.Content>
    </Card.Root>

    <div class="flex flex-col gap-4">
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

      <Card.Root>
        <Card.Header>
          <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
            Character
          </Card.Title>
        </Card.Header>
        <Card.Content>
          {#if characterSlug}
            <div class="flex flex-wrap items-center gap-2 mb-3">
              <p class="text-primary font-mono text-sm flex-1">{characterSlug}</p>
              {#if characterSheet && !isEditingCharacter}
                <Button size="xs" variant="outline" onclick={startEditCharacter}>Edit</Button>
                <Button size="xs" variant="outline" onclick={() => void openRoleplayPrompt()}>
                  Prompt
                </Button>
              {/if}
            </div>
            {#if characterError}
              <p class="text-destructive text-xs">{characterError}</p>
            {:else if isEditingCharacter}
              <div class="flex flex-col gap-2">
                <textarea
                  class="bg-background border-border text-foreground min-h-[18rem] w-full
                         resize-vertical rounded-md border p-2.5 font-mono text-sm"
                  bind:value={editBuffer}
                  disabled={editSaving}
                  aria-label="Character sheet markdown"
                ></textarea>
                {#if editError}
                  <p class="text-destructive text-xs">{editError}</p>
                {/if}
                <div class="flex gap-2">
                  <Button size="sm" onclick={() => void saveEditCharacter()} disabled={editSaving}>
                    {editSaving ? 'Saving…' : 'Save'}
                  </Button>
                  <Button size="sm" variant="outline" onclick={cancelEditCharacter} disabled={editSaving}>
                    Cancel
                  </Button>
                </div>
              </div>
            {:else if characterSheet}
              <div class="sheet">{@html characterSheetHtml}</div>
            {:else}
              <div class="space-y-2">
                <Skeleton class="h-3 w-2/3" />
                <Skeleton class="h-3 w-1/2" />
                <Skeleton class="h-20 w-full" />
              </div>
            {/if}
          {:else}
            <p class="text-muted-foreground italic text-sm">No character selected.</p>
          {/if}
        </Card.Content>
      </Card.Root>
    </div>
  </section>

  <!-- autoplay needs a user gesture, which the Start button provides;
       the bind happens via $effect once a remote track arrives. -->
  <audio bind:this={audioEl} autoplay playsinline></audio>
</main>

<!-- Roleplay prompt dialog (read-only, copy to clipboard) -->
<Dialog.Root
  open={promptModalOpen}
  onOpenChange={(v) => {
    promptModalOpen = v;
    if (!v) {
      promptText = '';
      promptError = null;
      promptCopied = false;
    }
  }}
>
  <Dialog.Content class="sm:max-w-3xl max-h-[85dvh] flex flex-col">
    <Dialog.Header>
      <Dialog.Title class="font-mono text-base">
        Roleplay prompt — {characterSlug}
      </Dialog.Title>
    </Dialog.Header>
    {#if promptError}
      <p class="text-destructive text-xs">{promptError}</p>
    {:else if promptText}
      <textarea
        class="bg-background border-border text-foreground flex-1 resize-none rounded-md border p-3 font-mono text-sm"
        readonly
        value={promptText}
        aria-label="Generated prompt text"
      ></textarea>
      <Dialog.Footer>
        <Button onclick={() => void copyRoleplayPrompt()}>
          {promptCopied ? 'Copied ✓' : 'Copy to clipboard'}
        </Button>
      </Dialog.Footer>
    {:else}
      <p class="text-muted-foreground italic text-sm">Generating…</p>
    {/if}
  </Dialog.Content>
</Dialog.Root>

<style>
  /* Local: blink keyframe used by the streaming caret. */
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  /* Marked-rendered character-sheet markdown. Vault content is authored
   * server-side, so :global() selectors here only style markup we
   * control — no XSS surface change. */
  .sheet {
    background: var(--background);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    max-height: 40vh;
    overflow-y: auto;
    color: var(--foreground);
    font-size: 0.85rem;
    line-height: 1.55;
  }
  .sheet :global(h1),
  .sheet :global(h2),
  .sheet :global(h3) {
    margin: 0.5rem 0 0.4rem;
    color: var(--primary);
    font-size: 0.95rem;
    letter-spacing: 0.02em;
  }
  .sheet :global(h1):first-child,
  .sheet :global(h2):first-child {
    margin-top: 0;
  }
  .sheet :global(p) {
    margin: 0.35rem 0;
  }
  .sheet :global(ul),
  .sheet :global(ol) {
    margin: 0.35rem 0;
    padding-left: 1.25rem;
  }
  .sheet :global(li) {
    margin: 0.1rem 0;
  }
  .sheet :global(code) {
    background: var(--card);
    border-radius: 3px;
    padding: 0.05rem 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }
  .sheet :global(hr) {
    border: 0;
    border-top: 1px solid var(--border);
    margin: 0.6rem 0;
  }
  .sheet :global(strong) {
    color: var(--foreground);
  }
  .sheet :global(em) {
    color: var(--muted-foreground);
  }
</style>
