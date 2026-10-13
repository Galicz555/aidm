// Browser-side client for the character vault endpoints.
//
// fetchCharacter returns the markdown plus the Last-Modified header so
// callers can pass it back to updateCharacter for optimistic concurrency
// — the DM can write the same file mid-edit, and the PUT then 409s
// instead of silently clobbering.
//
// Campaign + run scoped: every call takes both slugs since character
// sheets live under vault/<campaign>/runs/<run>/characters/. Different
// (campaign, run) pairs have different character state; passing both
// disambiguates which scope the caller wants to read or write.

export interface CharacterSheet {
  markdown: string;
  lastModified: string | null;
}

export type UpdateResult =
  | { ok: true }
  | { ok: false; conflict: boolean; status: number; message: string };

function scopedUrl(slug: string, campaignId: string, runId: string, suffix = ''): string {
  const base = `/api/vault/characters/${encodeURIComponent(slug)}${suffix}`;
  const qs = `?campaign=${encodeURIComponent(campaignId)}&run=${encodeURIComponent(runId)}`;
  return base + qs;
}

export async function fetchCharacter(
  slug: string,
  campaignId: string,
  runId: string
): Promise<CharacterSheet> {
  const response = await fetch(scopedUrl(slug, campaignId, runId));
  if (!response.ok) {
    throw new Error(`Failed to load sheet (${response.status})`);
  }
  return {
    markdown: await response.text(),
    lastModified: response.headers.get('Last-Modified'),
  };
}

export async function updateCharacter(
  slug: string,
  campaignId: string,
  runId: string,
  markdown: string,
  lastModified: string | null
): Promise<UpdateResult> {
  const headers: Record<string, string> = { 'Content-Type': 'text/plain' };
  if (lastModified) headers['If-Unmodified-Since'] = lastModified;
  const response = await fetch(scopedUrl(slug, campaignId, runId), {
    method: 'PUT',
    headers,
    body: markdown,
  });
  if (response.status === 204) return { ok: true };
  let message = `Save failed (${response.status})`;
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) message = body.detail;
  } catch {
    // body wasn't JSON; keep the generic message
  }
  return {
    ok: false,
    conflict: response.status === 409,
    status: response.status,
    message,
  };
}

export async function fetchRoleplayPrompt(
  slug: string,
  campaignId: string,
  runId: string
): Promise<string> {
  const response = await fetch(scopedUrl(slug, campaignId, runId, '/roleplay-prompt'));
  if (!response.ok) {
    throw new Error(`Failed to load prompt (${response.status})`);
  }
  return await response.text();
}
