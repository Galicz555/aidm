<script lang="ts">
  // shadcn-svelte primitives sandbox. Eyeball the palette + accessibility
  // before migrating /+page.svelte and /text/+page.svelte onto these
  // components. NOT linked from the app — visit /sandbox manually.
  //
  // Goal: every primitive AIDM needs across both pages renders here in
  // a Card so density / typography / palette read against the real
  // dark-fantasy mood, not against a default Tailwind page.

  import * as Card from '$lib/components/ui/card';
  import * as Dialog from '$lib/components/ui/dialog';
  import * as Select from '$lib/components/ui/select';
  import * as Sheet from '$lib/components/ui/sheet';
  import * as Tabs from '$lib/components/ui/tabs';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { Badge } from '$lib/components/ui/badge';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import { Skeleton } from '$lib/components/ui/skeleton';
  import { toast } from 'svelte-sonner';

  let dialogOpen = $state(false);
  let sheetOpen = $state(false);
  let inputValue = $state<string>('');

  // Mock run inventory. The selected value is the slug; the visible
  // label is the title — an accessibility fix (don't read the slug aloud
  // to screen readers / show it in the trigger).
  type Run = { slug: string; title: string; sessionCount: number };
  const runs: Run[] = [
    { slug: 'default', title: 'Default Run', sessionCount: 12 },
    { slug: 'replay-2026', title: 'Replay 2026', sessionCount: 3 },
    { slug: 'first-party', title: 'First Party', sessionCount: 0 }
  ];
  let selectedSlug = $state<string>('default');
  const selectedTitle = $derived(
    runs.find((r) => r.slug === selectedSlug)?.title ?? 'Pick a run'
  );

  let activeTab = $state<string>('characters');
</script>

<svelte:head>
  <title>AIDM — sandbox</title>
</svelte:head>

<div class="mx-auto max-w-4xl space-y-6 p-8">
  <header class="space-y-1">
    <h1 class="text-2xl font-semibold tracking-tight">shadcn-svelte sandbox</h1>
    <p class="text-muted-foreground text-sm">
      Foundation eyeball — gold accent, dried-blood destructive, Inter UI
      font, JetBrains Mono for system labels. Every primitive renders
      against a <code class="font-mono text-xs">Card</code> so density reads honestly.
    </p>
  </header>

  <Card.Root>
    <Card.Header>
      <Card.Title>Button</Card.Title>
      <Card.Description>Variants + sizes. <code class="font-mono">size="xs"</code> is the density mode for transcript chrome.</Card.Description>
    </Card.Header>
    <Card.Content class="space-y-3">
      <div class="flex flex-wrap items-center gap-2">
        <Button>Primary</Button>
        <Button variant="secondary">Secondary</Button>
        <Button variant="outline">Outline</Button>
        <Button variant="ghost">Ghost</Button>
        <Button variant="destructive">Destructive</Button>
        <Button variant="link">Link</Button>
        <Button disabled>Disabled</Button>
      </div>
      <div class="flex flex-wrap items-center gap-2">
        <Button size="xs">xs</Button>
        <Button size="sm">sm</Button>
        <Button size="default">default</Button>
        <Button size="lg">lg</Button>
      </div>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Input + Label</Card.Title>
      <Card.Description>Form chrome — the run-create modal will use this exact pair.</Card.Description>
    </Card.Header>
    <Card.Content class="space-y-2">
      <Label for="title">Run title</Label>
      <Input id="title" placeholder="e.g. First party" bind:value={inputValue} />
      <p class="text-muted-foreground text-xs">
        Value: <span class="font-mono">{inputValue || '(empty)'}</span>
      </p>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Select — run picker pattern</Card.Title>
      <Card.Description>Trigger shows the human title, not the slug; screen readers hear "Replay 2026", not "replay-2026".</Card.Description>
    </Card.Header>
    <Card.Content>
      <Select.Root type="single" bind:value={selectedSlug}>
        <Select.Trigger class="w-[240px]">
          {selectedTitle}
        </Select.Trigger>
        <Select.Content>
          {#each runs as run (run.slug)}
            <Select.Item value={run.slug}>
              {run.title} <span class="text-muted-foreground">({run.sessionCount})</span>
            </Select.Item>
          {/each}
        </Select.Content>
      </Select.Root>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Badge — status pills</Card.Title>
      <Card.Description>State chips on the session bar (idle / generating / connected / error).</Card.Description>
    </Card.Header>
    <Card.Content class="flex flex-wrap gap-2">
      <Badge>active</Badge>
      <Badge variant="secondary">private mode</Badge>
      <Badge variant="outline">idle</Badge>
      <Badge variant="destructive">error</Badge>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Dialog — create new run</Card.Title>
      <Card.Description>The flow we'll wire to <code class="font-mono">POST /api/campaigns/&#123;slug&#125;/runs</code>.</Card.Description>
    </Card.Header>
    <Card.Content>
      <Dialog.Root bind:open={dialogOpen}>
        <Dialog.Trigger>
          {#snippet child({ props })}
            <Button {...props}>Open dialog</Button>
          {/snippet}
        </Dialog.Trigger>
        <Dialog.Content>
          <Dialog.Header>
            <Dialog.Title>Create a new run</Dialog.Title>
            <Dialog.Description>
              A new playthrough of <strong>this campaign</strong>. World canon stays
              shared; characters, sessions and notes live under this run.
            </Dialog.Description>
          </Dialog.Header>
          <div class="space-y-2">
            <Label for="run-title">Title</Label>
            <Input id="run-title" placeholder="e.g. Replay 2026" />
          </div>
          <Dialog.Footer>
            <Button variant="ghost" onclick={() => (dialogOpen = false)}>Cancel</Button>
            <Button onclick={() => (dialogOpen = false)}>Create</Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog.Root>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Toast (Sonner)</Card.Title>
      <Card.Description>Visual error channel — voice mode speaks errors, text mode toasts them.</Card.Description>
    </Card.Header>
    <Card.Content class="flex flex-wrap gap-2">
      <Button size="sm" onclick={() => toast('Reconnecting…')}>Info toast</Button>
      <Button size="sm" variant="outline" onclick={() => toast.success('Run created: replay-2026')}>
        Success toast
      </Button>
      <Button size="sm" variant="destructive" onclick={() => toast.error('Daily budget cap reached')}>
        Error toast
      </Button>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Tooltip</Card.Title>
      <Card.Description>Status-dot hover, mode-badge hint, persona-variant explainer.</Card.Description>
    </Card.Header>
    <Card.Content class="flex items-center gap-3">
      <Tooltip.Provider>
        <Tooltip.Root>
          <Tooltip.Trigger>
            {#snippet child({ props })}
              <Badge {...props} variant="outline">private mode</Badge>
            {/snippet}
          </Tooltip.Trigger>
          <Tooltip.Content>
            Local stack: Whisper + Gemma + MOSS-TTS-Nano
          </Tooltip.Content>
        </Tooltip.Root>
      </Tooltip.Provider>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Skeleton</Card.Title>
      <Card.Description>Transcript / character-sheet placeholders while the vault fetch is in flight.</Card.Description>
    </Card.Header>
    <Card.Content class="space-y-2">
      <Skeleton class="h-4 w-2/3" />
      <Skeleton class="h-4 w-1/2" />
      <Skeleton class="h-20 w-full" />
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>ScrollArea — transcript</Card.Title>
      <Card.Description>Styled scrollbars; native ones look shabby on Linux.</Card.Description>
    </Card.Header>
    <Card.Content>
      <ScrollArea class="h-40 rounded-md border p-3">
        <div class="space-y-2 text-sm">
          {#each Array(20) as _, i (i)}
            <p>
              <span class="text-muted-foreground font-mono text-xs">DM:</span>
              <span class="text-ink italic">A folyosó mélyén egy halvány fény dereng — gyertyák a kápolnából.</span>
            </p>
          {/each}
        </div>
      </ScrollArea>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Tabs — sidebar pattern</Card.Title>
      <Card.Description>Text page side panel: characters / scene / notes.</Card.Description>
    </Card.Header>
    <Card.Content>
      <Tabs.Root bind:value={activeTab}>
        <Tabs.List>
          <Tabs.Trigger value="characters">Characters</Tabs.Trigger>
          <Tabs.Trigger value="scene">Scene</Tabs.Trigger>
          <Tabs.Trigger value="notes">Notes</Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="characters" class="text-muted-foreground pt-3 text-sm">
          Active party + the NPC sheet currently in scene.
        </Tabs.Content>
        <Tabs.Content value="scene" class="text-muted-foreground pt-3 text-sm">
          Current location, lighting, weather, who's listening from the shadows.
        </Tabs.Content>
        <Tabs.Content value="notes" class="text-muted-foreground pt-3 text-sm">
          Running session summary the auto-summarizer appends every N turns.
        </Tabs.Content>
      </Tabs.Root>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Sheet — mobile drawer</Card.Title>
      <Card.Description>The text page's side panel becomes a Sheet under <code class="font-mono">md:</code>.</Card.Description>
    </Card.Header>
    <Card.Content>
      <Sheet.Root bind:open={sheetOpen}>
        <Sheet.Trigger>
          {#snippet child({ props })}
            <Button {...props} variant="outline">Open sheet</Button>
          {/snippet}
        </Sheet.Trigger>
        <Sheet.Content side="right">
          <Sheet.Header>
            <Sheet.Title>Scene + Character</Sheet.Title>
            <Sheet.Description>
              Side-channel events the persona LLM emits — scene.change,
              character.change, npc_update_state.
            </Sheet.Description>
          </Sheet.Header>
          <div class="p-4">
            <p class="text-muted-foreground text-sm">
              The Sheet wraps the same panels the desktop layout shows
              side-by-side, so the markup is shared across breakpoints.
            </p>
          </div>
        </Sheet.Content>
      </Sheet.Root>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Typography preview</Card.Title>
      <Card.Description>Inter UI + JetBrains Mono system. Hungarian diacritics are the real test.</Card.Description>
    </Card.Header>
    <Card.Content class="space-y-2">
      <p class="text-lg font-semibold">Vasfátyol Birodalom — A Fellegvár ostroma</p>
      <p>
        Halkan koppan a datapad az asztalon. Egy pillanat, hadd nézzem át —
        <span class="text-ink italic">a folyosó mélyén egy halvány fény dereng.</span>
      </p>
      <p class="font-mono text-xs text-muted-foreground">
        session=29330591 turn=12 latency=1.2s
      </p>
    </Card.Content>
  </Card.Root>
</div>
