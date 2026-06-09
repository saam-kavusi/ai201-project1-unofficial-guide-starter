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

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/chunks.json"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "asu_cs_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

# Metadata fields copied from each chunk into ChromaDB (everything except "text",
# which is stored as the document body).
METADATA_FIELDS = ("source", "url", "course", "type", "filename", "chunk_id")

# Loaded once and reused — loading the model is the slow part.
_model: SentenceTransformer | None = None


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

def _build_where(
    course_filter: str | None = None,
    type_filter: str | None = None,
    filename_filter: str | None = None,
) -> dict | None:
    """Build a ChromaDB `where` filter from optional metadata constraints.

    Returns None when no filters are given (so retrieval is unconstrained),
    a single-field clause when exactly one is given, or an $and of all the
    given clauses. Each clause is an exact metadata match ($eq), so filtering
    by e.g. course="CSE 340" matches only chunks tagged exactly "CSE 340".
    """
    conditions = []
    if course_filter:
        conditions.append({"course": course_filter})
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

    Optional metadata filters (course, type, filename) restrict the search to
    matching chunks via a ChromaDB `where` filter. When all filters are None
    this behaves exactly like the original interface — an unconstrained search.

    Each result is a dict with the chunk text, its cosine distance score, and
    all stored metadata (source, url, course, type, filename, chunk_id).
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = _get_collection_if_exists(client)
    if collection is None or collection.count() == 0:
        # Vector store missing/empty — build it before querying.
        collection = build_vector_store()

    query_embedding = embed_texts([query])[0]
    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    where = _build_where(course_filter, type_filter, filename_filter)
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # query() returns parallel lists nested one level per query; we sent one.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for text, meta, distance in zip(documents, metadatas, distances):
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
    return chunks


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


if __name__ == "__main__":
    main()
