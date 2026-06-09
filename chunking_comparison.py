"""
Stretch feature — Chunking Strategy Comparison.

Standalone evaluation script (does NOT touch the production app, data/chunks.json,
or the persistent data/chroma_db collection). For each chunking strategy it:

  1. loads + cleans the same documents from documents/raw via the ingest.py
     pipeline (load_documents / clean_text / chunk_text),
  2. chunks them at that strategy's token size + overlap,
  3. embeds the chunks with all-MiniLM-L6-v2 (reusing vector_store.embed_texts,
     so the model loads once and matches production),
  4. builds a temporary IN-MEMORY ChromaDB collection (cosine distance),
  5. runs the five planning.md evaluation questions against it.

It prints, for every query/strategy, the top-3 chunk_ids/sources, their cosine
distances, and a relevance flag; then prints and writes a summary table to
chunking_comparison.md.

Run:  python chunking_comparison.py
"""

import re

import chromadb

import ingest
from vector_store import embed_texts

OUTPUT_FILE = "chunking_comparison.md"
TOP_K = 3

# (label, target_tokens, overlap_tokens). Char budgets are derived with the same
# 1 token ≈ 4 chars approximation ingest.py uses, so these line up with the
# "~tokens" figures elsewhere in the project.
STRATEGIES = [
    ("Small  (~150 tok / 30 overlap)", 150, 30),
    ("Default(~250 tok / 50 overlap)", 250, 50),
    ("Large  (~400 tok / 80 overlap)", 400, 80),
]

# The five evaluation questions from planning.md, each paired with the course
# codes we'd expect a genuinely relevant top result to touch. Used only for the
# lightweight relevance heuristic below — not for retrieval.
EVAL_QUESTIONS = [
    ("Is it a good idea to take CSE 340 in the summer with Bazzi?",
     ["CSE 340"]),
    ("How bad is it to take CSE 330, CSE 340, and CSE 355 together in the summer?",
     ["CSE 330", "CSE 340", "CSE 355"]),
    ("Which one is harder: CSE 340 or CSE 355?",
     ["CSE 340", "CSE 355"]),
    ("Who should I take for CSE 330 or CSE 355?",
     ["CSE 330", "CSE 355"]),
    ("What should I expect from CSE 340 based on both the syllabus and student reviews?",
     ["CSE 340"]),
]


# ---------------------------------------------------------------------------
# Chunking (mirrors ingest.build_chunks, but with configurable sizing)
# ---------------------------------------------------------------------------

def build_chunks_for_strategy(docs, target_chars, overlap_chars):
    """Clean + chunk every doc at the given char budget, returning chunk dicts.

    Same shape and metadata as ingest.build_chunks so the rest of the pipeline
    is identical to production — only the chunk sizing differs.
    """
    all_chunks = []
    for doc in docs:
        text = ingest.clean_text(doc["text"])
        if not text:
            continue
        base_meta = {k: doc[k] for k in ("source", "url", "course", "type", "filename")}
        pieces = ingest.chunk_text(text, target_chars=target_chars, overlap_chars=overlap_chars)
        for idx, chunk_text_val in enumerate(pieces):
            all_chunks.append({
                **base_meta,
                "chunk_id": f"{doc['filename']}__chunk{idx}",
                "text": chunk_text_val,
            })
    return all_chunks


# ---------------------------------------------------------------------------
# Temporary in-memory vector store
# ---------------------------------------------------------------------------

def build_temp_collection(chunks, label):
    """Embed chunks and load them into a throwaway in-memory Chroma collection.

    Uses an EphemeralClient so nothing is written to disk — the production
    data/chroma_db store is never opened or modified.
    """
    client = chromadb.EphemeralClient()
    # Collection names must be alphanumeric/-/_; derive a safe one from the label.
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
    collection = client.create_collection(
        name=f"cmp_{safe_name}",
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = embed_texts([c["text"] for c in chunks])
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"course": c["course"], "source": c["source"]} for c in chunks],
        embeddings=embeddings,
    )
    return collection


def query_top_k(collection, question, top_k=TOP_K):
    """Return the top_k results as a list of {chunk_id, source, course, distance, text}."""
    q_emb = embed_texts([question])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    out = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({
            "chunk_id": cid,
            "source": meta.get("source", ""),
            "course": meta.get("course", ""),
            "distance": dist,
            "text": doc,
        })
    return out


# ---------------------------------------------------------------------------
# Relevance heuristic
# ---------------------------------------------------------------------------

def is_relevant(result, expected_courses):
    """Cheap relevance check: does the top result touch an expected course?

    Looks at both the chunk's `course` metadata and its text (course codes are
    matched space-insensitively, e.g. "CSE 340" also matches "CSE340"). This is
    a coarse signal for the comparison, not a precision metric.
    """
    haystack = f"{result['course']} {result['text']}".lower()
    haystack_nospace = haystack.replace(" ", "")
    for course in expected_courses:
        c = course.lower()
        if c in haystack or c.replace(" ", "") in haystack_nospace:
            return True
    return False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def chunk_stats(chunks):
    """Return (count, avg_chars, approx_avg_tokens) for a chunk list."""
    if not chunks:
        return 0, 0.0, 0.0
    lengths = [len(c["text"]) for c in chunks]
    avg_chars = sum(lengths) / len(lengths)
    return len(chunks), avg_chars, avg_chars / ingest.CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run():
    docs = ingest.load_documents(ingest.DOCS_DIR)
    print(f"Documents loaded: {len(docs)}\n")

    md = []  # markdown lines accumulated for chunking_comparison.md
    md.append("# Chunking Strategy Comparison\n")
    md.append(
        "Stretch feature. Compares three paragraph-aware token chunking sizes on "
        "the same five evaluation questions from `planning.md`, embedded with "
        "`all-MiniLM-L6-v2` and retrieved from a temporary in-memory ChromaDB "
        "collection (cosine distance — lower is better). The production app, "
        "`data/chunks.json`, and `data/chroma_db` are untouched.\n")

    summary = []  # one row of stats per strategy for the final table

    for label, target_tokens, overlap_tokens in STRATEGIES:
        target_chars = target_tokens * ingest.CHARS_PER_TOKEN
        overlap_chars = overlap_tokens * ingest.CHARS_PER_TOKEN

        chunks = build_chunks_for_strategy(docs, target_chars, overlap_chars)
        count, avg_chars, avg_tokens = chunk_stats(chunks)
        collection = build_temp_collection(chunks, label)

        header = f"Strategy: {label.strip()}  —  {count} chunks, avg ~{avg_tokens:.0f} tok ({avg_chars:.0f} chars)"
        print("\n" + "#" * 90)
        print(f"# {header}")
        print("#" * 90)

        md.append(f"\n## {label.strip()}\n")
        md.append(f"- Chunks: **{count}**")
        md.append(f"- Average chunk length: **~{avg_tokens:.0f} tokens** ({avg_chars:.0f} chars)\n")

        top1_distances = []
        relevant_count = 0

        for q_num, (question, expected) in enumerate(EVAL_QUESTIONS, 1):
            results = query_top_k(collection, question, TOP_K)
            top1 = results[0]
            top1_distances.append(top1["distance"])
            relevant = is_relevant(top1, expected)
            relevant_count += int(relevant)

            print(f"\n[Q{q_num}] {question}")
            print(f"      expected courses: {', '.join(expected)}")
            for rank, r in enumerate(results, 1):
                print(f"   {rank}. d={r['distance']:.4f}  {r['chunk_id']:38s} "
                      f"[{r['course'] or '—'}]  ({r['source']})")
            print(f"      top-1 relevant by course/title? {'YES' if relevant else 'no'}")

            md.append(f"\n**Q{q_num}. {question}**  ")
            md.append(f"_Expected courses: {', '.join(expected)}_\n")
            md.append("| Rank | Distance | chunk_id | Course | Source |")
            md.append("|------|----------|----------|--------|--------|")
            for rank, r in enumerate(results, 1):
                md.append(f"| {rank} | {r['distance']:.4f} | `{r['chunk_id']}` | "
                          f"{r['course'] or '—'} | {r['source']} |")
            md.append(f"\nTop-1 relevant by course/title? **{'YES' if relevant else 'no'}**\n")

        avg_top1 = sum(top1_distances) / len(top1_distances)
        summary.append({
            "label": label.strip(),
            "count": count,
            "avg_tokens": avg_tokens,
            "avg_chars": avg_chars,
            "avg_top1": avg_top1,
            "relevant": relevant_count,
            "total_q": len(EVAL_QUESTIONS),
        })

    # --- Summary table ---------------------------------------------------
    print("\n\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    col = f"{'Strategy':32s} {'#chunks':>8s} {'avg tok':>8s} {'avg top-1 d':>12s} {'top-1 relevant':>16s}"
    print(col)
    print("-" * 90)
    for row in summary:
        rel = f"{row['relevant']}/{row['total_q']}"
        print(f"{row['label']:32s} {row['count']:>8d} {row['avg_tokens']:>8.0f} "
              f"{row['avg_top1']:>12.4f} {rel:>16s}")

    md.append("\n## Summary\n")
    md.append("| Strategy | # chunks | Avg chunk length | Avg top-1 distance | Top-1 relevant | Notes |")
    md.append("|----------|----------|------------------|--------------------|----------------|-------|")
    best_dist = min(r["avg_top1"] for r in summary)
    best_rel = max(r["relevant"] for r in summary)
    for row in summary:
        notes = []
        if row["avg_top1"] == best_dist:
            notes.append("lowest avg top-1 distance")
        if row["relevant"] == best_rel:
            notes.append("most top-1 relevant hits")
        md.append(
            f"| {row['label']} | {row['count']} | ~{row['avg_tokens']:.0f} tok "
            f"({row['avg_chars']:.0f} chars) | {row['avg_top1']:.4f} | "
            f"{row['relevant']}/{row['total_q']} | {'; '.join(notes) or '—'} |")

    md.append(
        "\n_Lower average top-1 distance means the nearest retrieved chunk is, on "
        "average, semantically closer to the query. \"Top-1 relevant\" counts how "
        "many of the five questions returned a top result whose course metadata or "
        "text mentions an expected course code — a coarse on-topic signal, not a "
        "precision score._\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWrote results to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
