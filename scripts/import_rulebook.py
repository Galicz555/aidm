"""Convert rulebook PDFs into vault-ready markdown chunks.

Reads PDFs from ``rulebooks/<campaign>/*.pdf``, converts each via
``pymupdf4llm`` (preserves headings, tables, lists), splits on top-level
``# `` chapters, and writes one frontmatter-prefixed file per chapter
into ``vault/<campaign>/rules/``. Builds an ``index.md`` listing every
chunk so the DM-LLM has a single eager-read entry point at session
start.

Run::

    uv run --with pymupdf4llm python scripts/import_rulebook.py \\
        --campaign ember-hollow --lang hu

By default refuses to overwrite existing rules files — pass ``--force``
to replace them. ``--dry-run`` prints what would be written without
touching disk.

The conversion is 80% quality; expect to hand-edit the resulting
markdown. The point is to skip the OCR/copy-paste step, not to produce
a final knowledge base.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Chunk:
    source_pdf: str
    title: str
    slug: str
    body: str


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return cleaned or "untitled"


def pdf_to_markdown(pdf_path: Path, pages: list[int] | None = None) -> str:
    import pymupdf4llm  # type: ignore[import-not-found]

    # use_ocr=False: skip image-page OCR (needs system tesseract + langs we
    # don't ship). Text PDFs convert fine; scanned pages will be empty —
    # hand-fill them afterward from the original PDF.
    return pymupdf4llm.to_markdown(str(pdf_path), use_ocr=False, pages=pages)


def split_by_toc(pdf_path: Path, level: int) -> list[Chunk]:
    """Chunk a PDF using its outline (bookmarks) at the given level.

    Each TOC entry at ``level`` becomes one chunk spanning the pages from
    that entry up to (but not including) the next entry at the same level.
    Falls back to a single chunk if the PDF has no usable outline.
    """
    import pymupdf  # type: ignore[import-not-found]

    pdf_slug = slugify(pdf_path.stem)
    doc = pymupdf.open(str(pdf_path))
    toc = doc.get_toc()
    page_count = doc.page_count

    entries = [(title, page) for lvl, title, page in toc if lvl == level]
    if not entries:
        markdown = pdf_to_markdown(pdf_path)
        return [
            Chunk(
                source_pdf=pdf_path.name,
                title=pdf_path.stem,
                slug=pdf_slug,
                body=markdown.strip(),
            )
        ]

    chunks: list[Chunk] = []
    for idx, (title, start_page) in enumerate(entries):
        end_page = entries[idx + 1][1] - 1 if idx + 1 < len(entries) else page_count
        # pymupdf TOC pages are 1-indexed; pymupdf4llm wants 0-indexed.
        page_range = list(range(start_page - 1, end_page))
        if not page_range:
            continue
        body = pdf_to_markdown(pdf_path, pages=page_range).strip()
        chunks.append(
            Chunk(
                source_pdf=pdf_path.name,
                title=title,
                slug=f"{pdf_slug}-{slugify(title)}",
                body=body,
            )
        )
    return chunks


def split_chapters(markdown: str, source_pdf: str) -> list[Chunk]:
    """Split on top-level ``# `` headings. If none exist, return one chunk."""
    pdf_slug = slugify(Path(source_pdf).stem)
    lines = markdown.splitlines()

    headings: list[tuple[int, str]] = [
        (i, line[2:].strip())
        for i, line in enumerate(lines)
        if line.startswith("# ") and not line.startswith("## ")
    ]

    if not headings:
        return [
            Chunk(
                source_pdf=source_pdf,
                title=Path(source_pdf).stem,
                slug=pdf_slug,
                body=markdown.strip(),
            )
        ]

    chunks: list[Chunk] = []
    for idx, (start, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        chunks.append(
            Chunk(
                source_pdf=source_pdf,
                title=title,
                slug=f"{pdf_slug}-{slugify(title)}",
                body=body,
            )
        )
    return chunks


def render_frontmatter(chunk: Chunk, lang: str, system: str | None) -> str:
    lines = [
        "---",
        "type: rules",
        f"lang: {lang}",
        f"title: {yaml_string(chunk.title)}",
        f"source: {yaml_string(chunk.source_pdf)}",
    ]
    if system:
        lines.append(f"system: {yaml_string(system)}")
    lines.append("---")
    return "\n".join(lines)


def yaml_string(value: str) -> str:
    if re.search(r'[:#&*!|>%@`"\']', value) or value != value.strip():
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


def render_index(chunks: list[Chunk], lang: str) -> str:
    header_hu = "# Szabálykönyv index\n\n*PDF-ből konvertálva. Töréseket és hiányzó táblázatokat kézzel javíts.*\n\n"
    header_en = "# Rulebook index\n\n*Converted from PDF. Hand-fix any broken layout or missing tables.*\n\n"
    header = header_hu if lang == "hu" else header_en

    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_pdf, []).append(chunk)

    parts = [header]
    for source, source_chunks in by_source.items():
        parts.append(f"## {source}\n")
        for chunk in source_chunks:
            parts.append(f"- [[{chunk.slug}|{chunk.title}]]")
        parts.append("")
    body = "\n".join(parts)

    frontmatter = "\n".join(
        [
            "---",
            "type: rules",
            f"lang: {lang}",
            f"title: {yaml_string('Szabálykönyv index' if lang == 'hu' else 'Rulebook index')}",
            "---",
            "",
        ]
    )
    return frontmatter + body


def write_chunk(chunk: Chunk, target_dir: Path, lang: str, system: str | None) -> Path:
    path = target_dir / f"{chunk.slug}.md"
    content = render_frontmatter(chunk, lang, system) + "\n\n" + chunk.body + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="campaign slug (e.g. ember-hollow)")
    parser.add_argument(
        "--src",
        default=None,
        help="directory of PDFs (default: rulebooks/<campaign>/)",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="output directory (default: vault/<campaign>/rules/)",
    )
    parser.add_argument("--lang", default="hu", help="lang frontmatter value (default: hu)")
    parser.add_argument(
        "--system",
        default=None,
        help="optional system tag (e.g. 'OSR') written to frontmatter",
    )
    parser.add_argument(
        "--toc-level",
        type=int,
        default=0,
        help="if > 0, chunk by PDF outline at this level instead of by markdown headings "
        "(D&D 5e SRD: level 2 = main chapters)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be written without touching disk",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    src = (Path(args.src) if args.src else REPO_ROOT / "rulebooks" / args.campaign).resolve()
    dest = (
        Path(args.dest) if args.dest else REPO_ROOT / "vault" / args.campaign / "rules"
    ).resolve()

    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found in {src}", file=sys.stderr)
        return 1

    if not dest.exists():
        print(f"destination missing: {dest}", file=sys.stderr)
        return 1

    existing = [p for p in dest.glob("*.md") if p.name != ".gitkeep"]
    if existing and not args.force and not args.dry_run:
        print(
            f"refusing to overwrite {len(existing)} existing files in {dest}; "
            "pass --force to replace.",
            file=sys.stderr,
        )
        return 1

    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        print(f"converting {pdf.name} ...", file=sys.stderr)
        if args.toc_level > 0:
            chunks = split_by_toc(pdf, args.toc_level)
        else:
            markdown = pdf_to_markdown(pdf)
            chunks = split_chapters(markdown, pdf.name)
        print(f"  → {len(chunks)} chunk(s)", file=sys.stderr)
        all_chunks.extend(chunks)

    if args.dry_run:
        print("\n--- dry run ---")
        for chunk in all_chunks:
            print(f"  {chunk.slug}.md  ({len(chunk.body)} chars)  ← {chunk.source_pdf}")
        print(f"  index.md  ({len(all_chunks)} entries)")
        return 0

    if args.force:
        for path in existing:
            path.unlink()

    for chunk in all_chunks:
        path = write_chunk(chunk, dest, args.lang, args.system)
        print(f"wrote {path.relative_to(REPO_ROOT)}", file=sys.stderr)

    index_path = dest / "index.md"
    index_path.write_text(render_index(all_chunks, args.lang), encoding="utf-8")
    print(f"wrote {index_path.relative_to(REPO_ROOT)}", file=sys.stderr)

    print(f"\n✓ {len(all_chunks)} chunks + index written to {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
