// @vitest-environment jsdom
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { renderMarkdown } from './markdown';

describe('renderMarkdown', () => {
  it('keeps ordinary markdown', () => {
    const html = renderMarkdown(
      '# Lyra\n\n**brave** and [kind](https://example.org)\n\n| a | b |\n|---|---|\n| 1 | 2 |'
    );
    expect(html).toContain('<h1>Lyra</h1>');
    expect(html).toContain('<strong>brave</strong>');
    expect(html).toContain('href="https://example.org"');
    expect(html).toContain('<table>');
  });

  it('drops script elements', () => {
    expect(renderMarkdown('hi <script>window.pwned = 1</script>')).not.toMatch(/<script/i);
  });

  it('drops inline event handlers from raw html', () => {
    const html = renderMarkdown('<img src=x onerror="window.pwned = 1">');
    expect(html).not.toMatch(/onerror/i);
  });

  it('drops javascript: urls from markdown links', () => {
    const html = renderMarkdown('[click](javascript:window.pwned=1)');
    expect(html).not.toMatch(/javascript:/i);
  });

  it('drops iframes and svg handlers', () => {
    expect(renderMarkdown('<iframe src="https://evil.example"></iframe>')).not.toMatch(/<iframe/i);
    expect(renderMarkdown('<svg onload="window.pwned = 1"></svg>')).not.toMatch(/onload/i);
  });
});

describe('markdown rendering call sites', () => {
  // Every {@html} sink must get its markup from renderMarkdown; a page that
  // imports marked directly would bring the unsanitised path back.
  function sourceFiles(dir: string): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) return sourceFiles(path);
      return /\.(svelte|ts)$/.test(entry.name) ? [path] : [];
    });
  }

  it('only markdown.ts imports marked', () => {
    const offenders = sourceFiles('src')
      .filter((path) => !path.endsWith('markdown.ts') && !path.endsWith('markdown.test.ts'))
      .filter((path) => /from ['"]marked['"]/.test(readFileSync(path, 'utf8')));
    expect(offenders).toEqual([]);
  });
});
