<script lang="ts">
  // Live session-state chip — dot + uppercase label. The dot colour
  // tracks a coarse status (connected / busy / error / idle), separate
  // from the displayed label so the page can pass e.g. "generating" as
  // the label without inventing a matching token.

  interface Props {
    label: string;
    tone: 'connected' | 'busy' | 'error' | 'idle';
  }

  let { label, tone }: Props = $props();

  const dotClass = $derived(
    tone === 'connected'
      ? 'bg-emerald-400'
      : tone === 'busy'
        ? 'bg-amber-400'
        : tone === 'error'
          ? 'bg-destructive'
          : 'bg-muted-foreground'
  );
</script>

<span
  class="text-muted-foreground inline-flex items-center gap-2 font-mono text-xs uppercase tracking-wider"
  data-state={label}
>
  <span class="inline-block size-2.5 rounded-full {dotClass}" aria-hidden="true"></span>
  {label}
</span>
