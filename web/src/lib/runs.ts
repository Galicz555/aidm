// Campaign + run inventory helpers.
//
// One orchestrator process serves exactly one campaign (its VAULT_PATH),
// but the API surface is multi-campaign-shaped so the same UI works once
// we move to multitenancy. The picker fetches campaigns first, then runs
// per campaign; an empty run list is a fresh campaign that needs a first
// run created.

export interface Campaign {
  slug: string;
  title: string;
}

export interface Run {
  slug: string;
  title: string;
  status: 'active' | 'completed' | 'archived' | string;
  created: string;
  session_count: number;
  last_session_date: string | null;
}

export interface CreateRunInput {
  title: string;
  status?: 'active' | 'completed' | 'archived';
}

export async function listCampaigns(): Promise<Campaign[]> {
  const response = await fetch('/api/campaigns');
  if (!response.ok) throw new Error(`failed to list campaigns: ${response.status}`);
  return (await response.json()) as Campaign[];
}

export async function listRuns(campaignSlug: string): Promise<Run[]> {
  const response = await fetch(`/api/campaigns/${encodeURIComponent(campaignSlug)}/runs`);
  if (!response.ok) throw new Error(`failed to list runs: ${response.status}`);
  return (await response.json()) as Run[];
}

export async function createRun(campaignSlug: string, input: CreateRunInput): Promise<Run> {
  const response = await fetch(`/api/campaigns/${encodeURIComponent(campaignSlug)}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input)
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`failed to create run (${response.status}): ${detail}`);
  }
  return (await response.json()) as Run;
}
