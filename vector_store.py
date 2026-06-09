"""
Milestone 4 — Embedding and retrieval.
Loads chunks from data/chunks.json, embeds each chunk's text with the
all-MiniLM-L6-v2 sentence-transformers model, stores them in a persistent
ChromaDB collection with metadata, and exposes retrieve_chunks() for
semantic similarity search.

Generation (Milestone 5) lives elsewhere — this file only builds the vector
store and retrieves chunks.
"""

import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/chunks.json"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "asu_cs_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

# Metadata fields copied from each chunk into ChromaDB (everything except "text",
# which is stored as the document body).
METADATA_FIELDS = ("source", "url", "course", "type", "filename", "chunk_id")

# Reciprocal Rank Fusion constant. The standard value (60) dampens the influence
# of any single high rank so neither the semantic nor the keyword list dominates.
RRF_K = 60

# Course filtering is post-filtered in Python (course tags can be compound and
# Chroma `where` has no substring match). To avoid the filter starving the
# result set, we over-fetch a candidate pool from Chroma and slice to top_k
# AFTER filtering. This multiplier (and floor) size that pool.
COURSE_FILTER_POOL_MULTIPLIER = 10
COURSE_FILTER_POOL_MIN = 50

# Loaded once and reused — loading the model is the slow part.
_model: SentenceTransformer | None = None

# BM25 index + its backing chunks, built once on first hybrid query.
_bm25: BM25Okapi | None = None
_bm25_chunks: list[dict] | None = None


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def get_model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts and return plain Python lists for ChromaDB.

    Embeddings are L2-normalized so that cosine distance (configured on the
    collection) behaves consistently between indexing and querying.
    """
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(rebuild: bool = False) -> "chromadb.Collection":
    """Create (or open) the persistent ChromaDB collection and populate it.

    Re-running is safe: if the collection already holds exactly the chunks in
    data/chunks.json it is returned untouched. Otherwise it is rebuilt from
    scratch so we never end up with stale or duplicated entries.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    chunks = load_chunks()

    # If a fully-populated collection already exists, reuse it as-is.
    if not rebuild:
        existing = _get_collection_if_exists(client)
        if existing is not None and existing.count() == len(chunks):
            print(f"Vector store already built: {existing.count()} chunks.")
            return existing

    # (Re)build from scratch. delete_collection is a no-op-safe reset here.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    # cosine distance pairs well with normalized sentence-transformer vectors.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{field: chunk[field] for field in METADATA_FIELDS} for chunk in chunks]

    print(f"Embedding {len(documents)} chunks with {MODEL_NAME}...")
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Stored {collection.count()} chunks in '{COLLECTION_NAME}' at {CHROMA_DIR}.")
    return collection


def _get_collection_if_exists(client) -> "chromadb.Collection | None":
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _course_matches(chunk_course: str | None, course_filter: str | None) -> bool:
    """True if `course_filter` appears in a chunk's (possibly compound) course tag.

    Course metadata is not always a single course. Some chunks are tagged with
    compound values like "CSE 330, CSE 355" or
    "General ASU CS / CSE 330, CSE 340, CSE 355". An exact-equality filter would
    exclude those when the user filters by a single course (e.g. "CSE 355"),
    even though they are exactly the cross-listed threads we want. A substring
    ("contains") test keeps them. The course filter values come from a fixed
    dropdown (e.g. "CSE 355"), so substring matching is unambiguous here.

    An empty/None filter matches everything, preserving unconstrained behavior.
    """
    if not course_filter:
        return True
    return course_filter in (chunk_course or "")


def _build_where(
    type_filter: str | None = None,
    filename_filter: str | None = None,
) -> dict | None:
    """Build a ChromaDB `where` filter from the exact-match metadata constraints.

    Only `type` and `filename` are expressed here, as exact matches ($eq).
    Course filtering is intentionally NOT included: course tags can be compound
    (e.g. "CSE 330, CSE 355") and Chroma `where` has no substring operator, so
    course is post-filtered in Python via _course_matches() instead.

    Returns None when no filters are given (so retrieval is unconstrained),
    a single-field clause when exactly one is given, or an $and of both.
    """
    conditions = []
    if type_filter:
        conditions.append({"type": type_filter})
    if filename_filter:
        conditions.append({"filename": filename_filter})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    course_filter: str | None = None,
    type_filter: str | None = None,
    filename_filter: str | None = None,
) -> list[dict]:
    """Return the top_k most relevant chunks for `query`.

    Optional metadata filters restrict the search to matching chunks. `type` and
    `filename` are exact matches applied as a ChromaDB `where` filter. `course`
    is matched inclusively (substring/"contains") and post-filtered in Python,
    so a single-course filter like "CSE 355" also matches compound tags such as
    "CSE 330, CSE 355" (see _course_matches). When all filters are None this
    behaves exactly like the original interface — an unconstrained search.

    Each result is a dict with the chunk text, its cosine distance score, and
    all stored metadata (source, url, course, type, filename, chunk_id).
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = _get_collection_if_exists(client)
    if collection is None or collection.count() == 0:
        # Vector store missing/empty — build it before querying.
        collection = build_vector_store()

    query_embedding = embed_texts([query])[0]

    # When post-filtering by course, over-fetch so the top_k survivors after
    # filtering are still the most relevant matching chunks. Without a course
    # filter we fetch exactly top_k, preserving the original behavior.
    n_results = top_k
    if course_filter:
        pool = max(top_k * COURSE_FILTER_POOL_MULTIPLIER, COURSE_FILTER_POOL_MIN)
        n_results = min(pool, collection.count())

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    where = _build_where(type_filter, filename_filter)
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # query() returns parallel lists nested one level per query; we sent one.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for text, meta, distance in zip(documents, metadatas, distances):
        # Inclusive course post-filter (compound tags like "CSE 330, CSE 355").
        if not _course_matches(meta.get("course", ""), course_filter):
            continue
        chunks.append({
            "text": text,
            "distance": distance,
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "course": meta.get("course", ""),
            "type": meta.get("type", ""),
            "filename": meta.get("filename", ""),
            "chunk_id": meta.get("chunk_id", ""),
        })
        if len(chunks) >= top_k:
            break
    return chunks


# ---------------------------------------------------------------------------
# Keyword (BM25) retrieval
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer.

    Splitting on non-alphanumerics keeps course codes and names intact as
    separate tokens — "CSE 340" -> ["cse", "340"], "Bazzi" -> ["bazzi"] — which
    is exactly what we want BM25 to match on for exact-name/course queries.
    """
    return _TOKEN_RE.findall(text.lower())


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    """Return the shared BM25 index over data/chunks.json, building it once.

    The index and its backing chunk list stay parallel (same order), so a score
    at position i belongs to chunk i.
    """
    global _bm25, _bm25_chunks
    if _bm25 is None:
        _bm25_chunks = load_chunks()
        tokenized = [_tokenize(chunk["text"]) for chunk in _bm25_chunks]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _bm25_chunks


def _chunk_matches(
    chunk: dict,
    course_filter: str | None,
    type_filter: str | None,
    filename_filter: str | None,
) -> bool:
    """Apply the same metadata filters used on the semantic path.

    `course` is matched inclusively (substring) so compound tags survive;
    `type` and `filename` stay exact. Mirrors retrieve_chunks exactly.
    """
    if not _course_matches(chunk.get("course"), course_filter):
        return False
    if type_filter and chunk.get("type") != type_filter:
        return False
    if filename_filter and chunk.get("filename") != filename_filter:
        return False
    return True


def keyword_retrieve(
    query: str,
    top_k: int = 5,
    course_filter: str | None = None,
    type_filter: str | None = None,
    filename_filter: str | None = None,
) -> list[dict]:
    """Return the top_k chunks for `query` ranked by BM25 keyword relevance.

    Metadata filters are applied in Python (mirroring the ChromaDB `where`
    filter on the semantic path). Results carry a `bm25_score` instead of a
    cosine `distance`, plus the same metadata fields as retrieve_chunks.
    """
    bm25, chunks = _get_bm25()
    scores = bm25.get_scores(_tokenize(query))

    scored = [
        (score, chunk)
        for chunk, score in zip(chunks, scores)
        if _chunk_matches(chunk, course_filter, type_filter, filename_filter)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, chunk in scored[:top_k]:
        results.append({
            "text": chunk["text"],
            "bm25_score": float(score),
            "source": chunk.get("source", ""),
            "url": chunk.get("url", ""),
            "course": chunk.get("course", ""),
            "type": chunk.get("type", ""),
            "filename": chunk.get("filename", ""),
            "chunk_id": chunk.get("chunk_id", ""),
        })
    return results


# ---------------------------------------------------------------------------
# Hybrid retrieval (semantic + keyword via Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def retrieve_chunks_hybrid(
    query: str,
    top_k: int = 5,
    course_filter: str | None = None,
    type_filter: str | None = None,
    filename_filter: str | None = None,
    semantic_k: int = 10,
    keyword_k: int = 10,
) -> list[dict]:
    """Combine semantic and keyword retrieval into one ranked list.

    Pulls `semantic_k` chunks from Chroma and `keyword_k` from BM25 (both with
    the same metadata filters applied), merges and deduplicates them by
    chunk_id, and re-ranks with Reciprocal Rank Fusion (RRF). RRF is rank-based,
    so it sidesteps the problem that cosine distances and BM25 scores live on
    incomparable scales.

    Each result has the same shape as retrieve_chunks (text + source/url/course/
    type/filename/chunk_id) plus:
        score        : the fused RRF score (higher = better; results are sorted
                       by this, descending)
        distance     : the cosine distance if the chunk surfaced semantically,
                       else None
        bm25_score   : the BM25 score if the chunk surfaced via keywords, else
                       None
    """
    semantic = retrieve_chunks(
        query,
        top_k=semantic_k,
        course_filter=course_filter,
        type_filter=type_filter,
        filename_filter=filename_filter,
    )
    keyword = keyword_retrieve(
        query,
        top_k=keyword_k,
        course_filter=course_filter,
        type_filter=type_filter,
        filename_filter=filename_filter,
    )

    merged: dict[str, dict] = {}
    fused: dict[str, float] = {}

    def _fuse(chunk_id: str, rank: int) -> None:
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    for rank, chunk in enumerate(semantic, 1):
        cid = chunk["chunk_id"]
        entry = merged.setdefault(cid, {
            "text": chunk["text"],
            "distance": chunk.get("distance"),
            "bm25_score": None,
            "source": chunk.get("source", ""),
            "url": chunk.get("url", ""),
            "course": chunk.get("course", ""),
            "type": chunk.get("type", ""),
            "filename": chunk.get("filename", ""),
            "chunk_id": cid,
        })
        entry["distance"] = chunk.get("distance")
        _fuse(cid, rank)

    for rank, chunk in enumerate(keyword, 1):
        cid = chunk["chunk_id"]
        entry = merged.setdefault(cid, {
            "text": chunk["text"],
            "distance": None,
            "bm25_score": None,
            "source": chunk.get("source", ""),
            "url": chunk.get("url", ""),
            "course": chunk.get("course", ""),
            "type": chunk.get("type", ""),
            "filename": chunk.get("filename", ""),
            "chunk_id": cid,
        })
        entry["bm25_score"] = chunk.get("bm25_score")
        _fuse(cid, rank)

    for cid, entry in merged.items():
        entry["score"] = fused[cid]

    ranked = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    "Is it a good idea to take CSE 340 in the summer with Bazzi?",
    "How bad is it to take CSE 330, CSE 340, and CSE 355 together in the summer?",
    "Which one is harder: CSE 340 or CSE 355?",
]


def main():
    # Build (or reuse) the store once, up front.
    build_vector_store()

    for q_num, question in enumerate(EVAL_QUESTIONS, 1):
        print("\n" + "=" * 80)
        print(f"[Query {q_num}] {question}")
        print("=" * 80)
        results = retrieve_chunks(question, top_k=5)
        for rank, chunk in enumerate(results, 1):
            print(f"\n  Result {rank}  (distance: {chunk['distance']:.4f})")
            print(f"    source   : {chunk['source']}")
            print(f"    url      : {chunk['url']}")
            print(f"    course   : {chunk['course']}")
            print(f"    type     : {chunk['type']}")
            print(f"    filename : {chunk['filename']}")
            print(f"    chunk_id : {chunk['chunk_id']}")
            preview = chunk["text"][:300].replace("\n", " ")
            print(f"    text     : {preview}{'...' if len(chunk['text']) > 300 else ''}")


def _metric(chunk: dict) -> str:
    """One-line relevance summary spanning both retrieval modes."""
    bits = []
    if chunk.get("score") is not None:
        bits.append(f"score={chunk['score']:.4f}")
    if chunk.get("distance") is not None:
        bits.append(f"d={chunk['distance']:.4f}")
    if chunk.get("bm25_score") is not None:
        bits.append(f"bm25={chunk['bm25_score']:.2f}")
    return " ".join(bits) or "n/a"


def compare(question: str, top_k: int = 5, course_filter: str | None = None) -> None:
    """Print Semantic vs Hybrid top_k side by side for one question."""
    note = f"  (course_filter={course_filter})" if course_filter else ""
    print("\n" + "=" * 80)
    print(f"[Compare] {question}{note}")
    print("=" * 80)

    print("\n  --- Semantic ---")
    for rank, c in enumerate(retrieve_chunks(question, top_k, course_filter=course_filter), 1):
        print(f"    {rank}. {c['chunk_id']:42s} [{c['course']}]  {_metric(c)}")

    print("\n  --- Hybrid ---")
    for rank, c in enumerate(retrieve_chunks_hybrid(question, top_k, course_filter=course_filter), 1):
        print(f"    {rank}. {c['chunk_id']:42s} [{c['course']}]  {_metric(c)}")


def main():
    # Build (or reuse) the store once, up front.
    build_vector_store()

    for q_num, question in enumerate(EVAL_QUESTIONS, 1):
        print("\n" + "=" * 80)
        print(f"[Query {q_num}] {question}")
        print("=" * 80)
        results = retrieve_chunks(question, top_k=5)
        for rank, chunk in enumerate(results, 1):
            print(f"\n  Result {rank}  (distance: {chunk['distance']:.4f})")
            print(f"    source   : {chunk['source']}")
            print(f"    url      : {chunk['url']}")
            print(f"    course   : {chunk['course']}")
            print(f"    type     : {chunk['type']}")
            print(f"    filename : {chunk['filename']}")
            print(f"    chunk_id : {chunk['chunk_id']}")
            preview = chunk["text"][:300].replace("\n", " ")
            print(f"    text     : {preview}{'...' if len(chunk['text']) > 300 else ''}")

    # Semantic vs Hybrid comparison (Hybrid stretch feature).
    print("\n\n" + "#" * 80)
    print("# SEMANTIC vs HYBRID retrieval comparison")
    print("#" * 80)
    compare("Is CSE 340 good in summer with Bazzi?")
    compare("Who should I take for CSE 330 or CSE 355?")
    compare("Which one is harder: CSE 340 or CSE 355?")
    compare("Is CSE 340 good in summer with Bazzi?", course_filter="CSE 340")


if __name__ == "__main__":
    main()
