"""
Milestone 5 — Generation.

Connects Milestone 4 retrieval (vector_store.retrieve_chunks) to the Groq
llama-3.3-70b-versatile LLM. The public entry point is ask(question), which:

  1. retrieves the top-k most relevant chunks,
  2. formats them as numbered context blocks,
  3. sends the question + context to Groq under a strict grounding prompt, and
  4. returns the answer together with a deduplicated source list and the raw
     retrieved chunks (with distance scores) for debugging/evaluation.

Source attribution is built programmatically from chunk metadata rather than
trusting the LLM to cite correctly — see build_sources().
"""

import os

from dotenv import load_dotenv
from groq import Groq

from vector_store import retrieve_chunks, retrieve_chunks_hybrid

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

# Retrieval strategies selectable from ask()/the UI. "Hybrid" fuses semantic and
# BM25 keyword search; "Semantic" is the original vector-only path.
RETRIEVAL_MODES = ("Hybrid", "Semantic")
DEFAULT_RETRIEVAL_MODE = "Hybrid"

# The exact sentence the model must use when the context is insufficient.
INSUFFICIENT_INFO = (
    "I don't have enough information in the provided sources to answer that."
)

# ---------------------------------------------------------------------------
# Grounding prompt
# ---------------------------------------------------------------------------
# This system prompt is the grounding contract. It forces the model to answer
# only from the retrieved context and to refuse when the context is too thin.
SYSTEM_PROMPT = f"""You are a careful assistant for an "Unofficial Guide" to \
ASU computer science courses (focused on CSE 330, 340, and 355). You answer \
student questions using ONLY the numbered context passages provided with each \
question.

Strict rules — follow all of them:
1. Answer ONLY using information found in the provided context passages. Do not \
use any outside knowledge, prior training, or assumptions.
2. If the context does not contain enough information to answer the question, \
respond with EXACTLY this sentence and nothing else: \
"{INSUFFICIENT_INFO}"
3. Never invent or guess professor names, course policies, grades, \
assignments, deadlines, workloads, or recommendations. If a detail is not in \
the context, do not state it.
4. The context includes informal student opinions (e.g. Reddit) as well as \
official ASU documents. When sources disagree, or when a claim rests only on \
one or two student opinions, say so explicitly and describe it as opinion \
rather than fact.
5. Be concise and directly address the question. You may summarize and \
synthesize across the passages, but every claim must be traceable to the \
context.

A separate, trusted source list is attached to your answer automatically, so \
focus on giving an accurate, grounded answer rather than formatting citations \
yourself."""


# ---------------------------------------------------------------------------
# Conversational memory (stretch feature)
# ---------------------------------------------------------------------------
# Memory is used ONLY to interpret follow-up questions. It shapes the *retrieval
# query* so that a terse follow-up like "What about CSE 355?" pulls back the
# right chunks. It is never fed to the LLM as fact and never becomes a source —
# grounding still comes entirely from the retrieved chunks (see ask()).

def build_contextual_query(question: str, history: list[dict] | None) -> str:
    """Combine the previous question with the current one into a retrieval query.

    `history` is a list of prior turns, each a dict with a "question" key
    (and typically an "answer"). We use only the most recent question so a
    follow-up carries enough context to retrieve relevant chunks.

    With no usable history this returns the question unchanged, so the
    single-turn behavior is preserved exactly.

        >>> build_contextual_query(
        ...     "What about CSE 355?",
        ...     [{"question": "Who should I take for CSE 330 or CSE 355?"}],
        ... )
        'Previous question: Who should I take for CSE 330 or CSE 355? Follow-up question: What about CSE 355?'
    """
    if not history:
        return question
    prev_question = (history[-1].get("question") or "").strip()
    if not prev_question:
        return question
    return f"Previous question: {prev_question} Follow-up question: {question}"


# ---------------------------------------------------------------------------
# Context + source formatting
# ---------------------------------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered context blocks for the prompt."""
    if not chunks:
        return "(no context passages were retrieved)"

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] Source: {chunk.get('source', 'unknown')}"
        course = chunk.get("course")
        if course:
            header += f" | Course: {course}"
        source_type = chunk.get("type")
        if source_type:
            header += f" | Type: {source_type}"
        blocks.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(blocks)


def build_sources(chunks: list[dict]) -> list[dict]:
    """Build a deduplicated source list from chunk metadata.

    Source attribution is guaranteed here, in code — independent of whatever
    the LLM writes. Sources are deduplicated by (source, url) so multiple
    chunks from the same thread/document collapse into one entry, while the
    contributing chunk_ids are preserved for traceability.
    """
    sources: dict[tuple, dict] = {}
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("url", ""))
        if key not in sources:
            sources[key] = {
                "source": chunk.get("source", ""),
                "url": chunk.get("url", ""),
                "course": chunk.get("course", ""),
                "type": chunk.get("type", ""),
                "filename": chunk.get("filename", ""),
                "chunk_ids": [],
            }
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id and chunk_id not in sources[key]["chunk_ids"]:
            sources[key]["chunk_ids"].append(chunk_id)
    return list(sources.values())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_client: Groq | None = None


def get_client() -> Groq:
    """Return the shared Groq client, creating it on first use."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(question: str, context: str) -> str:
    """Send the question + numbered context to Groq and return the answer."""
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, following all the rules."
    )
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # low — we want faithful, not creative, answers
    )
    return response.choices[0].message.content.strip()


def ask(
    question: str,
    top_k: int = TOP_K,
    course_filter: str | None = None,
    type_filter: str | None = None,
    filename_filter: str | None = None,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    retrieval_query: str | None = None,
) -> dict:
    """Answer a question, grounded only in retrieved chunks.

    Optional metadata filters (course, type, filename) are passed straight
    through to retrieval; leaving them as None retrieves over all chunks, so
    ask(question) behaves exactly as before.

    `retrieval_mode` selects the retriever: "Hybrid" (default) fuses semantic
    and BM25 keyword search; "Semantic" uses vector search only. Anything other
    than "Hybrid" falls back to the semantic path, so the original behavior is
    always one argument away.

    `retrieval_query` lets a caller retrieve with text that differs from the
    user's literal question — used by conversational memory to inject the
    previous question into a terse follow-up (see build_contextual_query). It
    affects ONLY what chunks come back; the LLM still answers the original
    `question` against those chunks, and sources are still built from chunk
    metadata. Defaults to `question`, preserving single-turn behavior.

    Returns a dict:
        {
          "answer": str,             # grounded natural-language answer
          "sources": list[dict],     # deduplicated source metadata
          "retrieved_chunks": list[dict],  # raw chunks w/ score/distance
        }
    """
    retrieval_query = retrieval_query or question

    if retrieval_mode == "Hybrid":
        retrieved_chunks = retrieve_chunks_hybrid(
            retrieval_query,
            top_k=top_k,
            course_filter=course_filter,
            type_filter=type_filter,
            filename_filter=filename_filter,
        )
    else:
        retrieved_chunks = retrieve_chunks(
            retrieval_query,
            top_k=top_k,
            course_filter=course_filter,
            type_filter=type_filter,
            filename_filter=filename_filter,
        )

    # No retrieval => nothing to ground on; refuse without calling the LLM.
    if not retrieved_chunks:
        return {
            "answer": INSUFFICIENT_INFO,
            "sources": [],
            "retrieved_chunks": [],
        }

    context = format_context(retrieved_chunks)
    answer = generate_answer(question, context)
    sources = build_sources(retrieved_chunks)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }


# ---------------------------------------------------------------------------
# Test block — end-to-end generation on the evaluation questions
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "Is it a good idea to take CSE 340 in the summer with Bazzi?",
    "Which one is harder: CSE 340 or CSE 355?",
    "Who should I take for CSE 330 or CSE 355?",
    "What clubs are best for ASU students?",  # expected: refuse / not enough info
]


def main():
    for q_num, question in enumerate(TEST_QUESTIONS, 1):
        print("\n" + "=" * 80)
        print(f"[Q{q_num}] {question}")
        print("=" * 80)
        result = ask(question)

        print("\nANSWER:\n" + result["answer"])

        print("\nSOURCES:")
        if result["sources"]:
            for s in result["sources"]:
                print(f"  - {s['source']} ({s['course']}, {s['type']})")
                print(f"      {s['url']}")
                print(f"      chunks: {', '.join(s['chunk_ids'])}")
        else:
            print("  (none)")

        print("\nRETRIEVED CHUNKS (relevance scores):")
        for rank, chunk in enumerate(result["retrieved_chunks"], 1):
            preview = chunk["text"][:120].replace("\n", " ")
            # Hybrid chunks carry a fused `score` and may have distance/bm25 None
            # (a chunk surfaced only via BM25 has no cosine distance, and vice
            # versa), so format defensively rather than assuming a float.
            bits = []
            if chunk.get("score") is not None:
                bits.append(f"score={chunk['score']:.4f}")
            if chunk.get("distance") is not None:
                bits.append(f"d={chunk['distance']:.4f}")
            if chunk.get("bm25_score") is not None:
                bits.append(f"bm25={chunk['bm25_score']:.2f}")
            metric = " ".join(bits) or "n/a"
            print(f"  {rank}. {metric}  {chunk['source']}  | {preview}...")


if __name__ == "__main__":
    main()
