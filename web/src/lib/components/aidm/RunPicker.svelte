<script lang="ts">
  // Campaign + run dropdowns + "+ New run" button + create-run dialog.
  // Shared between the voice (/) and text (/text) routes so the picker
  // behavior is one source of truth.
  //
  // Owns the create-run dialog UI; data (campaigns, runs, selection)
  // and side effects (localStorage, toasts) live in the parent.

  import * as Select from '$lib/components/ui/select';
  import * as Dialog from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import type { Campaign, Run } from '$lib/runs';

  interface Props {
    campaigns: Campaign[];
    runs: Run[];
    selectedCampaign: string;
    selectedRun: string;
    disabled: boolean;
    creatingRun: boolean;
    onCampaignChange: (value: string) => void | Promise<void>;
    onRunChange: (value: string) => void;
    onSubmitNewRun: (title: string) => void | Promise<void>;
  }

  let {
    campaigns,
    runs,
    selectedCampaign,
    selectedRun,
    disabled,
    creatingRun,
    onCampaignChange,
    onRunChange,
    onSubmitNewRun
  }: Props = $props();

  let newRunModalOpen = $state(false);
  let newRunTitle = $state('');

  // Display labels (the slug shown by default in Select.Trigger is
  // unreadable; pre-resolve the title via a lookup so the trigger
  // surface — and screen readers — read the human label).
  const campaignTitle = $derived(
    campaigns.find((c) => c.slug === selectedCampaign)?.title ?? 'Pick a campaign'
  );
  const runTitle = $derived(
    runs.find((r) => r.slug === selectedRun)?.title ?? 'Pick a run'
  );

  async function submit(): Promise<void> {
    const title = newRunTitle.trim();
    if (!title) return;
    await onSubmitNewRun(title);
    if (!creatingRun) {
      newRunTitle = '';
      newRunModalOpen = false;
    }
  }
</script>

<div class="flex flex-wrap items-end gap-2">
  <div class="flex flex-col gap-1">
    <Label class="text-muted-foreground text-[0.65rem] uppercase tracking-wider">
      Campaign
    </Label>
    <Select.Root
      type="single"
      value={selectedCampaign}
      onValueChange={onCampaignChange}
      {disabled}
    >
      <Select.Trigger class="h-8 min-w-[140px] text-sm">{campaignTitle}</Select.Trigger>
      <Select.Content>
        {#each campaigns as c (c.slug)}
          <Select.Item value={c.slug}>{c.title}</Select.Item>
        {/each}
        {#if campaigns.length === 0}
          <Select.Item value="" disabled>— loading —</Select.Item>
        {/if}
      </Select.Content>
    </Select.Root>
  </div>

  <div class="flex flex-col gap-1">
    <Label class="text-muted-foreground text-[0.65rem] uppercase tracking-wider">
      Run
    </Label>
    <Select.Root
      type="single"
      value={selectedRun}
      onValueChange={onRunChange}
      disabled={disabled || !selectedCampaign}
    >
      <Select.Trigger class="h-8 min-w-[180px] text-sm">{runTitle}</Select.Trigger>
      <Select.Content>
        {#each runs as r (r.slug)}
          <Select.Item value={r.slug}>
            {r.title} <span class="text-muted-foreground">({r.session_count})</span>
          </Select.Item>
        {/each}
        {#if runs.length === 0}
          <Select.Item value="" disabled>— no runs yet —</Select.Item>
        {/if}
      </Select.Content>
    </Select.Root>
  </div>

  <Button
    variant="outline"
    size="sm"
    disabled={disabled || !selectedCampaign}
    onclick={() => {
      newRunTitle = '';
      newRunModalOpen = true;
    }}
  >
    + Run
  </Button>
</div>

<Dialog.Root bind:open={newRunModalOpen}>
  <Dialog.Content class="sm:max-w-md">
    <Dialog.Header>
      <Dialog.Title>Create a new run</Dialog.Title>
      <Dialog.Description>
        A new playthrough of <strong>{selectedCampaign}</strong>. The world canon
        stays shared; characters, sessions, arcs and notes live under this run.
      </Dialog.Description>
    </Dialog.Header>
    <div class="space-y-2">
      <Label for="new-run-title">Title</Label>
      <Input
        id="new-run-title"
        placeholder="e.g. First party, Replay 2026"
        bind:value={newRunTitle}
        disabled={creatingRun}
        onkeydown={(e) => {
          if (e.key === 'Enter') void submit();
          if (e.key === 'Escape') newRunModalOpen = false;
        }}
      />
    </div>
    <Dialog.Footer>
      <Button variant="ghost" onclick={() => (newRunModalOpen = false)} disabled={creatingRun}>
        Cancel
      </Button>
      <Button onclick={() => void submit()} disabled={creatingRun || !newRunTitle.trim()}>
        {creatingRun ? 'Creating…' : 'Create'}
      </Button>
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.Root>
