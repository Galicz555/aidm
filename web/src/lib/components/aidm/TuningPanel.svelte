<svelte:options runes={true} />

<script lang="ts">
  // TuningPanel — Campaign Tone + active-run Beat editor cards.
  // State (toneText/beatText and drafts) lives entirely here.
  // The parent page receives the loaded tone text via `ontoneloaded`
  // so it can pass it to ModeBadge.
  // When the SSE `beat.change` event fires, the page sets `externalBeat`
  // and this component syncs the internal state without clobbering an
  // in-progress edit.

  import { untrack } from 'svelte';
  import { toast } from 'svelte-sonner';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    selectedPickerCampaign: string;
    selectedRun: string;
    /** Set to a non-null string whenever the SSE `beat.change` event fires
     *  so TuningPanel can sync beatText (and optionally beatDraft) live. */
    externalBeat?: string | null;
    isConnected?: boolean;
    ontoneloaded?: (text: string) => void;
  }

  let {
    selectedPickerCampaign,
    selectedRun,
    externalBeat = null,
    isConnected = false,
    ontoneloaded
  }: Props = $props();

  // ── Tone state ──────────────────────────────────────────────────────
  let toneText = $state<string>('');
  let toneDraft = $state<string>('');
  let toneSaving = $state<boolean>(false);
  let toneError = $state<string | null>(null);
  const toneDirty = $derived(toneDraft.trim() !== toneText.trim());

  // ── Beat state ──────────────────────────────────────────────────────
  let beatText = $state<string>('');
  let beatDraft = $state<string>('');
  let beatSaving = $state<boolean>(false);
  let beatError = $state<string | null>(null);
  const beatDirty = $derived(beatDraft.trim() !== beatText.trim());

  // ── SSE beat.change signal ──────────────────────────────────────────
  // Only re-runs when externalBeat changes.  Reads beatDraft/beatDirty
  // under untrack() so editing the textarea doesn't retrigger the effect.
  $effect(() => {
    const beat = externalBeat;
    if (beat === null) return;
    untrack(() => {
      const wasDirty = beatDraft.trim() !== beatText.trim();
      beatText = beat;
      if (!wasDirty) beatDraft = beat;
    });
  });

  // ── Auto-reload when campaign / run changes ─────────────────────────
  $effect(() => {
    void selectedPickerCampaign;
    void selectedRun;
    void loadSceneTuning();
  });

  // ── API ─────────────────────────────────────────────────────────────

  async function loadSceneTuning(): Promise<void> {
    if (!selectedPickerCampaign) {
      toneText = '';
      toneDraft = '';
      beatText = '';
      beatDraft = '';
      ontoneloaded?.('');
      return;
    }
    try {
      const toneRes = await fetch(
        `/api/vault/tone?campaign=${encodeURIComponent(selectedPickerCampaign)}`
      );
      toneText = toneRes.ok ? await toneRes.text() : '';
      toneDraft = toneText;
      toneError = null;
      ontoneloaded?.(toneText);
    } catch {
      toneText = '';
      toneDraft = '';
      ontoneloaded?.('');
    }
    if (!selectedRun) {
      beatText = '';
      beatDraft = '';
      return;
    }
    try {
      const beatRes = await fetch(
        `/api/vault/beat?campaign=${encodeURIComponent(selectedPickerCampaign)}` +
          `&run=${encodeURIComponent(selectedRun)}`
      );
      beatText = beatRes.ok ? await beatRes.text() : '';
      beatDraft = beatText;
      beatError = null;
    } catch {
      beatText = '';
      beatDraft = '';
    }
  }

  async function saveTone(): Promise<void> {
    if (!selectedPickerCampaign) {
      toneError = 'No active campaign.';
      return;
    }
    toneSaving = true;
    toneError = null;
    try {
      const res = await fetch(
        `/api/vault/tone?campaign=${encodeURIComponent(selectedPickerCampaign)}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'text/plain' },
          body: toneDraft
        }
      );
      if (!res.ok) {
        toneError = `save failed (${res.status})`;
        return;
      }
      toneText = toneDraft;
      ontoneloaded?.(toneText);
      toast.success('Tone updated');
    } catch (err) {
      toneError = err instanceof Error ? err.message : String(err);
    } finally {
      toneSaving = false;
    }
  }

  async function saveBeat(): Promise<void> {
    if (!selectedPickerCampaign || !selectedRun) {
      beatError = 'No active campaign/run.';
      return;
    }
    beatSaving = true;
    beatError = null;
    try {
      const res = await fetch(
        `/api/vault/beat?campaign=${encodeURIComponent(selectedPickerCampaign)}` +
          `&run=${encodeURIComponent(selectedRun)}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'text/plain' },
          body: beatDraft
        }
      );
      if (!res.ok) {
        beatError = `save failed (${res.status})`;
        return;
      }
      beatText = beatDraft;
      toast.success('Beat updated');
    } catch (err) {
      beatError = err instanceof Error ? err.message : String(err);
    } finally {
      beatSaving = false;
    }
  }
</script>

<!-- Tone card -->
<Card.Root class="m-3 mt-0">
  <Card.Header>
    <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
      Tone
    </Card.Title>
  </Card.Header>
  <Card.Content>
    <p class="text-muted-foreground mb-2 text-xs italic">
      Campaign-level tone. Slow-moving register the DM honours across
      every session.
    </p>
    <textarea
      class="bg-background border-border text-foreground min-h-[6rem] w-full resize-vertical rounded-md border p-2 font-mono text-xs"
      bind:value={toneDraft}
      disabled={toneSaving || !selectedPickerCampaign}
      aria-label="Campaign tone"
      placeholder="Witcher-szerű szürke realizmus, kozmikus súly háttérben."
    ></textarea>
    {#if toneError}
      <p class="text-destructive mt-1 text-xs">{toneError}</p>
    {/if}
    <div class="mt-2 flex items-center gap-2">
      <Button
        size="xs"
        onclick={() => void saveTone()}
        disabled={toneSaving || !toneDirty || !selectedPickerCampaign}
      >
        {toneSaving ? 'Saving…' : 'Save tone'}
      </Button>
      {#if toneDirty}
        <Button
          size="xs"
          variant="outline"
          onclick={() => (toneDraft = toneText)}
          disabled={toneSaving}
        >
          Reset
        </Button>
      {/if}
    </div>
  </Card.Content>
</Card.Root>

<!-- Beat card -->
<Card.Root class="m-3 mt-0">
  <Card.Header>
    <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
      Beat
    </Card.Title>
  </Card.Header>
  <Card.Content>
    <p class="text-muted-foreground mb-2 text-xs italic">
      Short prose the DM reads every turn. Tempo, tension, closeness,
      cruelty, humour. The DM may rewrite it mid-scene.
    </p>
    <textarea
      class="bg-background border-border text-foreground min-h-[8rem] w-full resize-vertical rounded-md border p-2 font-mono text-xs"
      bind:value={beatDraft}
      disabled={beatSaving || !selectedPickerCampaign || !selectedRun}
      aria-label="Current narrative beat"
      placeholder="Lassú felfedezés. Feszültség halk. Közelség: bizalmas."
    ></textarea>
    {#if beatError}
      <p class="text-destructive mt-1 text-xs">{beatError}</p>
    {/if}
    <div class="mt-2 flex items-center gap-2">
      <Button
        size="xs"
        onclick={() => void saveBeat()}
        disabled={beatSaving || !beatDirty || !selectedPickerCampaign || !selectedRun}
      >
        {beatSaving ? 'Saving…' : 'Save beat'}
      </Button>
      {#if beatDirty}
        <Button
          size="xs"
          variant="outline"
          onclick={() => (beatDraft = beatText)}
          disabled={beatSaving}
        >
          Reset
        </Button>
      {/if}
    </div>
  </Card.Content>
</Card.Root>
