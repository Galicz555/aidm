// SPA mode: no SSR, no prerendering. Vite dev serves the app; later a
// production build under `web/build/` will be served by the orchestrator
// (or any static host). SvelteKit's adapter-static produces that bundle.
//
// `fallback: 'index.html'` makes every unknown route resolve to the
// shell, which is what a single-page app needs — there's only ever one
// route ('/') today, but this future-proofs the bundle.

import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  compilerOptions: {
    // Svelte 5 runes mode for the whole project — opting in once here
    // beats sprinkling <svelte:options runes /> in every component, and
    // it makes svelte-check honor the $state/$derived/$effect types.
    runes: true
  },
  kit: {
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html',
      precompress: false,
      strict: true
    })
  }
};

export default config;
