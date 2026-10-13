// During `npm run dev` (port 5173) the orchestrator FastAPI server runs
// at http://127.0.0.1:8000. We proxy /api/* and /health through Vite so
// the browser sees a single origin — that keeps WebRTC, CORS, and
// future cookie-based session ids consistent between dev and prod.
//
// In production the orchestrator serves the built SvelteKit bundle from
// the same origin, so no proxy is needed and the same client code works
// unchanged.

import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

const ORCHESTRATOR_URL =
  process.env.AIDM_ORCHESTRATOR_URL || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  test: { include: ['src/**/*.test.ts'] },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: ORCHESTRATOR_URL,
        changeOrigin: true
      },
      '/health': {
        target: ORCHESTRATOR_URL,
        changeOrigin: true
      }
    }
  }
});
