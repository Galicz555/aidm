// Shared domain types for the AIDM text page and its extracted components.

export interface ResumeSession {
  id: string;
  source: 'log' | 'running';
  title: string;
  mtime: number;
  snippet: string;
  campaign: string | null;
  date: string | null;
  characters: string[];
  locations: string[];
}

export interface Campaign {
  slug: string;
  title: string;
  status: string;
  tone: string;
  lang: string;
  sessions: ResumeSession[];
}
