<script lang="ts">
  // Server-mode chip — public / private / unknown — with a tooltip
  // explaining the stack. Optionally renders a tone chip next to it
  // when the active campaign carries a `tone.md`.

  import * as Tooltip from '$lib/components/ui/tooltip';
  import { Badge } from '$lib/components/ui/badge';

  interface Props {
    mode: 'public' | 'private' | 'unknown';
    /** Campaign tone (raw markdown from tone.md). The chip shows the
     *  first line as a one-glance register; the full body is the
     *  tooltip body. Empty string == campaign has no tone.md. */
    tone?: string;
  }

  let { mode, tone = '' }: Props = $props();

  const tooltip = $derived(
    mode === 'private'
      ? 'Local stack: Whisper + Gemma + MOSS-TTS-Nano'
      : mode === 'public'
        ? 'Cloud stack: Groq / Claude / ElevenLabs'
        : 'Server mode unknown — orchestrator may not be running'
  );

  const badgeVariant = $derived(
    mode === 'unknown' ? 'destructive' : ('outline' as const)
  );

  // First non-empty line of tone.md as the chip text. Markdown headers
  // (`# Gritty`) are stripped for the chip; the tooltip carries the
  // full body for readers who want it.
  const toneTrimmed = $derived(tone.trim());
  const toneLabel = $derived(
    toneTrimmed
      .split('\n')
      .map((l) => l.trim())
      .find((l) => l.length > 0)
      ?.replace(/^#+\s*/, '')
      .slice(0, 32) ?? ''
  );
</script>

<div class="flex items-center gap-2">
  <Tooltip.Provider>
    <Tooltip.Root>
      <Tooltip.Trigger>
        {#snippet child({ props })}
          <Badge
            {...props}
            variant={badgeVariant}
            class="cursor-help font-mono text-[0.65rem] uppercase tracking-widest"
          >
            {mode === 'unknown' ? '—' : mode}
          </Badge>
        {/snippet}
      </Tooltip.Trigger>
      <Tooltip.Content>{tooltip}</Tooltip.Content>
    </Tooltip.Root>
  </Tooltip.Provider>

  {#if toneLabel}
    <Tooltip.Provider>
      <Tooltip.Root>
        <Tooltip.Trigger>
          {#snippet child({ props })}
            <Badge
              {...props}
              variant="secondary"
              class="cursor-help font-mono text-[0.65rem] uppercase tracking-widest"
            >
              {toneLabel}
            </Badge>
          {/snippet}
        </Tooltip.Trigger>
        <Tooltip.Content>
          <div class="max-w-xs whitespace-pre-wrap text-[0.7rem] leading-snug">
            {toneTrimmed}
          </div>
        </Tooltip.Content>
      </Tooltip.Root>
    </Tooltip.Provider>
  {/if}
</div>
