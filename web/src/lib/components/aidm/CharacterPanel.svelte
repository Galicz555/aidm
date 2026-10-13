<svelte:options runes={true} />

<script lang="ts">
  // CharacterPanel — Character sheet viewer/editor + "Generate prompt" modal.
  // Watches the `characterSlug` prop: when it changes (set by the page from
  // SSE character.change events), it fetches the sheet automatically.
  // When the slug becomes null (session ends), internal state is cleared.

  import { renderMarkdown } from '$lib/markdown';
  import * as Card from '$lib/components/ui/card';
  import * as Dialog from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import { Skeleton } from '$lib/components/ui/skeleton';
  import {
    fetchCharacter,
    updateCharacter,
    fetchRoleplayPrompt
  } from '$lib/character';

  interface Props {
    characterSlug: string | null;
    selectedPickerCampaign: string;
    selectedRun: string;
    isConnected?: boolean;
    oncharacterupdated?: (
      slug: string,
      sheet: string,
      lastModified: string | null
    ) => void;
  }

  let {
    characterSlug,
    selectedPickerCampaign,
    selectedRun,
    isConnected = false,
    oncharacterupdated
  }: Props = $props();

  // ── Internal state ───────────────────────────────────────────────────
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

  const characterSheetHtml = $derived(characterSheet ? renderMarkdown(characterSheet) : '');

  // ── React to slug changes ────────────────────────────────────────────
  // Fires on mount and whenever the page sets characterSlug (from SSE or
  // session end).
  $effect(() => {
    const slug = characterSlug;
    if (slug) {
      void loadCharacter(slug);
    } else {
      // Session ended or no character — clear everything.
      characterSheet = null;
      characterLastModified = null;
      characterError = null;
      isEditingCharacter = false;
      editBuffer = '';
      editError = null;
      promptModalOpen = false;
      promptText = '';
      promptError = null;
      promptCopied = false;
    }
  });

  // ── API ──────────────────────────────────────────────────────────────

  async function loadCharacter(slug: string): Promise<void> {
    characterError = null;
    characterSheet = null;
    characterLastModified = null;
    isEditingCharacter = false;
    editError = null;
    try {
      const { markdown, lastModified } = await fetchCharacter(
        slug,
        selectedPickerCampaign,
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
        selectedPickerCampaign,
        selectedRun,
        editBuffer,
        characterLastModified
      );
      if (result.ok) {
        isEditingCharacter = false;
        editBuffer = '';
        // Re-fetch so the rendered view shows the auto-filled
        // last_updated and the fresh mtime baseline for the next edit.
        await loadCharacter(characterSlug);
        oncharacterupdated?.(
          characterSlug,
          characterSheet ?? '',
          characterLastModified
        );
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
      promptText = await fetchRoleplayPrompt(
        characterSlug,
        selectedPickerCampaign,
        selectedRun
      );
    } catch (err) {
      promptError = err instanceof Error ? err.message : String(err);
    }
  }

  function closeRoleplayPrompt(): void {
    promptModalOpen = false;
    promptText = '';
    promptError = null;
    promptCopied = false;
  }

  async function copyRoleplayPrompt(): Promise<void> {
    if (!promptText) return;
    try {
      await navigator.clipboard.writeText(promptText);
      promptCopied = true;
    } catch {
      // Clipboard API may be unavailable on insecure origins. Fall
      // back to selecting the textarea so the user can ctrl-C.
      promptCopied = false;
    }
  }
</script>

<!-- Character card -->
<Card.Root class="m-3 mt-0">
  <Card.Header>
    <Card.Title class="text-muted-foreground text-xs uppercase tracking-wider font-mono">
      Character
    </Card.Title>
  </Card.Header>
  <Card.Content>
    {#if characterSlug}
      <div class="mb-3 flex flex-wrap items-center gap-2">
        <p class="text-primary flex-1 font-mono text-sm">{characterSlug}</p>
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
            class="bg-background border-border text-foreground min-h-[18rem] w-full resize-vertical rounded-md border p-2.5 font-mono text-sm"
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

<!-- Roleplay-prompt dialog (read-only, copy to clipboard) -->
<Dialog.Root
  open={promptModalOpen}
  onOpenChange={(v) => {
    promptModalOpen = v;
    if (!v) closeRoleplayPrompt();
  }}
>
  <Dialog.Content class="max-h-[85dvh] sm:max-w-3xl flex flex-col">
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
  /* Marked-rendered character-sheet markdown. Vault output is server-authored,
   * so :global() selectors here only style markup we control. */
  .sheet {
    color: var(--foreground);
    font-size: 0.9rem;
    line-height: 1.55;
    background: var(--background);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    max-height: 40vh;
    overflow-y: auto;
    font-size: 0.85rem;
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
    color: var(--ink);
  }
</style>
