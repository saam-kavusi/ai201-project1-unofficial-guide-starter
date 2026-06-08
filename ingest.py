"""
Milestone 3 — Document ingestion and chunking.
Loads .txt and .pdf files from documents/raw, cleans text, splits into
token-aware chunks with overlap, and saves to data/chunks.json.
"""

import json
import os
import re

import pdfplumber

DOCS_DIR = "documents/raw"
OUTPUT_FILE = "data/chunks.json"

TARGET_TOKENS = 250   # midpoint of 200–300
OVERLAP_TOKENS = 50   # midpoint of 40–60

# Rough approximation: 1 token ≈ 4 characters (good enough without loading a tokenizer)
CHARS_PER_TOKEN = 4
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN     # ~1000
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN   # ~200


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _parse_txt_metadata(lines: list[str]) -> tuple[dict, list[str]]:
    """Extract the Source/URL/Course/Type header and return (meta, body_lines)."""
    meta = {"source": "", "url": "", "course": "", "type": ""}
    keys = {"source": "source", "url": "url", "course": "course", "type": "type"}
    body_start = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        matched = False
        for prefix, key in keys.items():
            if lower.startswith(prefix + ":"):
                meta[key] = line[len(prefix) + 1:].strip()
                matched = True
                break
        if not matched and i > 0:
            body_start = i
            break
    return meta, lines[body_start:]


def load_txt(filepath: str) -> dict:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    meta, body_lines = _parse_txt_metadata(lines)
    meta["filename"] = os.path.basename(filepath)
    meta["text"] = "".join(body_lines)
    return meta


def load_pdf(filepath: str) -> dict:
    filename = os.path.basename(filepath)
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return {
        "source": filename,
        "url": "",
        "course": "",
        "type": "Official PDF",
        "filename": filename,
        "text": "\n\n".join(pages),
    }


def load_documents(directory: str) -> list[dict]:
    docs = []
    for fname in sorted(os.listdir(directory)):
        path = os.path.join(directory, fname)
        if fname.endswith(".txt"):
            docs.append(load_txt(path))
        elif fname.endswith(".pdf"):
            docs.append(load_pdf(path))
    return docs


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    # Collapse runs of whitespace within lines, drop truly empty strings
    lines = text.splitlines()
    cleaned = []
    blank_run = 0
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            blank_run += 1
            if blank_run <= 1:          # allow a single blank line as paragraph break
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(line)
    # Remove leading/trailing blank lines
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines, yielding non-empty paragraph strings."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(text: str) -> list[str]:
    """
    Build chunks of ~TARGET_CHARS with ~OVERLAP_CHARS overlap.
    Paragraphs are kept together when they fit; oversized paragraphs are
    split at sentence boundaries.
    """
    paragraphs = _split_into_paragraphs(text)

    # Expand oversized paragraphs into sentence-sized pieces
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= TARGET_CHARS:
            pieces.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            pieces.extend(s for s in sentences if s.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece)
        if current_len + piece_len > TARGET_CHARS and current:
            chunks.append("\n\n".join(current))
            # Retain the overlap tail from the previous chunk
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                if overlap_len + len(part) <= OVERLAP_CHARS:
                    overlap_parts.insert(0, part)
                    overlap_len += len(part)
                else:
                    break
            current = overlap_parts
            current_len = overlap_len
        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_chunks(docs: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in docs:
        text = clean_text(doc["text"])
        if not text:
            continue
        base_meta = {k: doc[k] for k in ("source", "url", "course", "type", "filename")}
        for idx, chunk_text_val in enumerate(chunk_text(text)):
            all_chunks.append({
                **base_meta,
                "chunk_id": f"{doc['filename']}__chunk{idx}",
                "text": chunk_text_val,
            })
    return all_chunks


def main():
    docs = load_documents(DOCS_DIR)
    print(f"Documents loaded: {len(docs)}")

    chunks = build_chunks(docs)
    print(f"Chunks created:   {len(chunks)}")

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}\n")

    print("--- 5 sample chunks ---")
    step = max(1, len(chunks) // 5)
    samples = chunks[::step][:5]
    for i, chunk in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(f"  source   : {chunk['source']}")
        print(f"  url      : {chunk['url']}")
        print(f"  course   : {chunk['course']}")
        print(f"  type     : {chunk['type']}")
        print(f"  filename : {chunk['filename']}")
        print(f"  chunk_id : {chunk['chunk_id']}")
        tokens_est = len(chunk["text"]) // CHARS_PER_TOKEN
        print(f"  ~tokens  : {tokens_est}")
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"  text     : {preview}{'...' if len(chunk['text']) > 300 else ''}")


if __name__ == "__main__":
    main()
