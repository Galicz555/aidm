<svelte:options runes={true} />

<script lang="ts">
  // ResumeSheet — 3-pane resume-session dialog.
  // Campaigns → Sessions → Preview. The parent controls open/close via
  // the bindable `open` prop. When the user clicks "Resume this session"
  // the component fires `onresumepick(choice)` and the parent handles the
  // actual session.start() call (it owns session, locale, run, etc.).

  import * as Dialog from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from '$lib/components/ui/badge';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import type { Campaign, ResumeSession } from '$lib/types/aidm';

  // Slug sentinel for the "orphan sessions" bucket in Pane 1.
  const ORPHAN_SLUG = '__orphans__';

  interface Props {
    open: boolean;
    campaigns: Campaign[];
    orphanSessions: ResumeSession[];
    onresumepick: (choice: string) => void | Promise<void>;
  }

  let {
    open = $bindable(),
    campaigns,
    orphanSessions,
    onresumepick
  }: Props = $props();

  // ── Internal navigation state ────────────────────────────────────────
  let focusedCampaign = $state<string | null>(null);
  let previewSession = $state<ResumeSession | null>(null);

  // When the sheet opens, default-focus the first campaign that has sessions.
  // When it closes, reset preview so the next open starts clean.
  $effect(() => {
    if (!open) {
      previewSession = null;
      return;
    }
    const withSessions = campaigns.find((c) => c.sessions.length > 0);
    if (withSessions) {
      focusedCampaign = withSessions.slug;
    } else if (orphanSessions.length > 0) {
      focusedCampaign = ORPHAN_SLUG;
    } else if (campaigns.length > 0) {
      focusedCampaign = campaigns[0].slug;
    } else {
      focusedCampaign = null;
    }
    previewSession = null;
  });

  // ── Derived ──────────────────────────────────────────────────────────
  const focusedCampaignData = $derived(() => {
    if (focusedCampaign === ORPHAN_SLUG) return null;
    return campaigns.find((c) => c.slug === focusedCampaign) ?? null;
  });

  const focusedSessions = $derived((): ResumeSession[] => {
    if (focusedCampaign === ORPHAN_SLUG) return orphanSessions;
    const c = focusedCampaignData();
    return c ? c.sessions : [];
  });

  // ── Handlers ─────────────────────────────────────────────────────────

  function selectCampaign(slug: string): void {
    focusedCampaign = slug;
    previewSession = null;
  }

  function previewPick(s: ResumeSession): void {
    previewSession = s;
  }

  function commitResumePick(): void {
    if (!previewSession) return;
    const choice = `${previewSession.source}:${previewSession.id}`;
    open = false;
    previewSession = null;
    onresumepick(choice);
  }
</script>

<Dialog.Root bind:open>
  <Dialog.Content class="max-h-[85dvh] sm:max-w-4xl flex flex-col">
    <Dialog.Header>
      <Dialog.Title>Resume previous session</Dialog.Title>
      <Dialog.Description>
        Pick a campaign, then a past session log. Use Resume to seed the new
        conversation with that session's memory.
      </Dialog.Description>
    </Dialog.Header>

    <div class="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden md:grid-cols-3">
      <!-- Pane 1: Campaigns -->
      <section class="border-border flex flex-col overflow-hidden rounded-md border">
        <h3 class="text-muted-foreground border-border border-b px-3 py-2 font-mono text-xs uppercase tracking-wider">
          Campaigns
        </h3>
        <ScrollArea class="flex-1">
          {#if campaigns.length === 0 && orphanSessions.length === 0}
            <p class="text-muted-foreground p-3 italic text-sm">No campaigns yet.</p>
          {:else}
            <ol class="p-2 space-y-1">
              {#each campaigns as c (c.slug)}
                <li>
                  <button
                    type="button"
                    class={'hover:bg-muted/40 w-full rounded-md p-2 text-left text-sm transition-colors ' +
                      (focusedCampaign === c.slug ? 'bg-muted' : '')}
                    onclick={() => selectCampaign(c.slug)}
                  >
                    <div class="text-foreground font-medium">{c.title}</div>
                    <div class="text-muted-foreground mt-0.5 flex items-center gap-2 text-xs">
                      {#if c.status}<span>{c.status}</span>{/if}
                      <span>{c.sessions.length} session{c.sessions.length === 1 ? '' : 's'}</span>
                    </div>
                  </button>
                </li>
              {/each}
              {#if orphanSessions.length > 0}
                <li>
                  <button
                    type="button"
                    class={'hover:bg-muted/40 w-full rounded-md p-2 text-left text-sm transition-colors ' +
                      (focusedCampaign === ORPHAN_SLUG ? 'bg-muted' : '')}
                    onclick={() => selectCampaign(ORPHAN_SLUG)}
                  >
                    <div class="text-destructive font-medium">⚠ Orphan sessions</div>
                    <div class="text-muted-foreground mt-0.5 text-xs">{orphanSessions.length} unlinked</div>
                  </button>
                </li>
              {/if}
            </ol>
          {/if}
        </ScrollArea>
      </section>

      <!-- Pane 2: Sessions -->
      <section class="border-border flex flex-col overflow-hidden rounded-md border">
        <h3 class="text-muted-foreground border-border border-b px-3 py-2 font-mono text-xs uppercase tracking-wider">
          {#if focusedCampaign === ORPHAN_SLUG}
            Orphan sessions
          {:else if focusedCampaignData()}
            {focusedCampaignData()?.title}
          {:else}
            Sessions
          {/if}
        </h3>
        <ScrollArea class="flex-1">
          {#if focusedSessions().length === 0}
            <p class="text-muted-foreground p-3 italic text-sm">No sessions in this bucket.</p>
          {:else}
            <ol class="p-2 space-y-1">
              {#each focusedSessions() as s (s.source + ':' + s.id)}
                <li>
                  <button
                    type="button"
                    class={'hover:bg-muted/40 w-full rounded-md p-2 text-left text-sm transition-colors ' +
                      (previewSession?.id === s.id && previewSession?.source === s.source
                        ? 'bg-muted'
                        : '')}
                    onclick={() => previewPick(s)}
                  >
                    <div class="flex items-center gap-2">
                      <span aria-hidden="true">{s.source === 'log' ? '📕' : '📜'}</span>
                      {#if s.date}
                        <span class="text-muted-foreground font-mono text-xs">{s.date}</span>
                      {/if}
                      <span class="text-foreground flex-1 truncate font-medium">{s.title}</span>
                    </div>
                    {#if s.characters.length > 0 || s.locations.length > 0}
                      <div class="mt-1 flex flex-wrap gap-1">
                        {#each s.characters as ch (ch)}
                          <Badge variant="outline" class="text-[0.6rem]">{ch}</Badge>
                        {/each}
                        {#each s.locations as loc (loc)}
                          <Badge variant="secondary" class="text-[0.6rem]">{loc}</Badge>
                        {/each}
                      </div>
                    {/if}
                    {#if s.snippet}
                      <p class="text-muted-foreground mt-1 line-clamp-2 text-xs">{s.snippet}</p>
                    {/if}
                  </button>
                </li>
              {/each}
            </ol>
          {/if}
        </ScrollArea>
      </section>

      <!-- Pane 3: Preview -->
      <section class="border-border flex flex-col overflow-hidden rounded-md border">
        <h3 class="text-muted-foreground border-border border-b px-3 py-2 font-mono text-xs uppercase tracking-wider">
          Preview
        </h3>
        <div class="flex-1 overflow-y-auto p-3">
          {#if previewSession}
            <div class="mb-2">
              <strong class="text-foreground">{previewSession.title}</strong>
              {#if previewSession.date}
                <span class="text-muted-foreground text-xs"> · {previewSession.date}</span>
              {/if}
            </div>
            {#if previewSession.characters.length > 0 || previewSession.locations.length > 0}
              <div class="mb-2 flex flex-wrap gap-1">
                {#each previewSession.characters as ch (ch)}
                  <Badge variant="outline" class="text-[0.6rem]">{ch}</Badge>
                {/each}
                {#each previewSession.locations as loc (loc)}
                  <Badge variant="secondary" class="text-[0.6rem]">{loc}</Badge>
                {/each}
              </div>
            {/if}
            <p class="text-ink mb-3 text-sm italic leading-relaxed">{previewSession.snippet}</p>
            <Button onclick={commitResumePick} class="w-full">🚀 Resume this session</Button>
          {:else}
            <p class="text-muted-foreground italic text-sm">Pick a session to see its summary.</p>
          {/if}
        </div>
      </section>
    </div>
  </Dialog.Content>
</Dialog.Root>
