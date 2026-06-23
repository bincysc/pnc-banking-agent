"""
Structure-aware chunking for markdown policy documents.

Production RAG chunking respects document structure rather than splitting
at arbitrary character counts. Markdown documents have explicit boundaries
(headings) that almost always correspond to semantic units — a policy
section, a fee category, an eligibility rule. Splitting at these boundaries
preserves the unit so retrieval surfaces complete answers.

When a section is too large (above MAX_CHUNK_CHARS), we fall back to
paragraph-level splitting within that section. The hard upper bound prevents
any single chunk from exceeding the embedding model's input limit.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Embedding models have input token limits. Amazon Titan Text Embeddings v2
# accepts up to 8192 tokens (~32k chars). We use a more conservative limit
# to keep retrieval focused and to leave headroom for instructions.
MAX_CHUNK_CHARS = 1500

# Below this size, splitting further would fragment the meaning.
MIN_CHUNK_CHARS = 50


@dataclass
class Chunk:
    """One retrievable unit of a document."""

    chunk_id: str          # Unique stable ID: "doc_name#section_index"
    document_name: str     # Source filename, used in citations
    section_title: str     # The heading this chunk belongs under
    content: str           # The actual text to embed and retrieve

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "section_title": self.section_title,
            "content": self.content,
        }


def chunk_markdown_file(file_path: Path) -> list[Chunk]:
    """
    Chunk a markdown file by heading structure.

    Algorithm:
    1. Read the file.
    2. Split at H2 headings ("## ..."). Each section becomes one chunk.
    3. If a section exceeds MAX_CHUNK_CHARS, split it further at paragraph
       boundaries until each piece is under the limit.
    4. Discard chunks below MIN_CHUNK_CHARS (likely heading-only fragments).
    """
    content = file_path.read_text(encoding="utf-8")
    document_name = file_path.stem  # filename without extension

    # Split at H2 boundaries. The regex captures the heading line itself
    # so we can use it as section_title.
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    for section_index, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract the section title from the first line if it's a heading
        lines = section.split("\n", 1)
        if lines[0].startswith("## "):
            section_title = lines[0].lstrip("#").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
        elif lines[0].startswith("# "):
            # The H1 document title — use it as the section title for the
            # opening preamble (text before any H2)
            section_title = lines[0].lstrip("#").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
        else:
            section_title = "Introduction"
            body = section

        if len(body) < MIN_CHUNK_CHARS:
            continue

        # If the section fits in one chunk, emit it as-is
        if len(body) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(
                chunk_id=f"{document_name}#{section_index}",
                document_name=document_name,
                section_title=section_title,
                content=body,
            ))
        else:
            # Section is too large — split by paragraph
            sub_chunks = _split_long_section(body)
            for sub_index, sub_text in enumerate(sub_chunks):
                chunks.append(Chunk(
                    chunk_id=f"{document_name}#{section_index}.{sub_index}",
                    document_name=document_name,
                    section_title=section_title,
                    content=sub_text,
                ))

    logger.info("chunked file=%s chunks=%d", document_name, len(chunks))
    return chunks


def _split_long_section(text: str) -> list[str]:
    """Split a too-large section at paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def chunk_knowledge_base(kb_dir: Path) -> list[Chunk]:
    """Chunk every .md file under the knowledge base directory."""
    all_chunks: list[Chunk] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        all_chunks.extend(chunk_markdown_file(md_file))
    logger.info("chunked_all total_chunks=%d source_files=%d",
                len(all_chunks), len(list(kb_dir.glob("*.md"))))
    return all_chunks