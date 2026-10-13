<script lang="ts">
  // Root layout — imports the central Tailwind + shadcn theme stylesheet
  // once so every page picks up the design tokens, and binds
  // ``mode-watcher`` so the dark/light/system mode is a class on <html>
  // (no per-component awareness needed).
  //
  // We ship dark-only today; ``defaultMode="dark"`` keeps mode-watcher
  // from honoring the system pref in the wrong direction while the
  // light palette is still placeholder. Flip to ``"system"`` when the
  // light theme is real.
  import '../app.css';
  import { ModeWatcher } from 'mode-watcher';
  import { Toaster } from '$lib/components/ui/sonner';

  let { children } = $props();
</script>

<ModeWatcher defaultMode="dark" />
<Toaster />

{@render children()}
