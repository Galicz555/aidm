// SPA mode: WebRTC, mic access, and the audio element all need a real
// browser. SSR would just render an empty page and then the client
// would have to rehydrate — pointless for this app. Disabling SSR also
// removes Node.js polyfill bloat from the bundle.
export const ssr = false;
export const prerender = false;
