import DOMPurify from 'dompurify';
import { marked } from 'marked';

// Markdown to HTML that is safe to inject with {@html}. marked passes raw
// HTML through untouched and the text comes from the LLM or from a PUT on
// the vault, so everything that reaches the DOM goes through DOMPurify.
export function renderMarkdown(source: string): string {
  const html = marked.parse(source, { gfm: true, breaks: false, async: false }) as string;
  return DOMPurify.sanitize(html);
}
