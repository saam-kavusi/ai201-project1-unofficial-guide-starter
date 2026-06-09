"""
Milestone 5 — Gradio interface (+ conversational memory stretch feature).

A Blocks UI over query.ask(): type a question, get a grounded answer, the
source list it was built from, and (optional) retrieved-chunk debug info with
distance scores. Both the Ask button and the Enter key submit.

Conversational memory: the app keeps the last few turns in Gradio State and
uses them to interpret terse follow-ups ("What about CSE 355?"). Memory shapes
ONLY the retrieval query — the answer is still generated from retrieved chunks
and sources are still built programmatically from chunk metadata, so memory
never becomes a source of factual truth.

Run with:  python app.py
"""

import gradio as gr

from query import (
    DEFAULT_RETRIEVAL_MODE,
    RETRIEVAL_MODES,
    ask,
    build_contextual_query,
)

# Conversational memory only ever holds the last MAX_MEMORY_TURNS turns, so the
# retrieval query stays small no matter how long the chat runs.
MAX_MEMORY_TURNS = 3

MEMORY_NOTE = (
    "ℹ️ Conversation memory is used only to interpret follow-up questions. "
    "Answers are still grounded only in retrieved sources."
)

# Optional metadata filters. "All" means "do not apply this filter" and maps to
# None before reaching retrieval. `type` is matched exactly against chunk
# metadata; `course` is matched inclusively, so picking "CSE 355" also surfaces
# chunks with compound tags like "CSE 330, CSE 355" (see vector_store).
ALL = "All"
COURSE_CHOICES = [ALL, "CSE 330", "CSE 340", "CSE 355", "General ASU CS"]
TYPE_CHOICES = [
    ALL,
    "Official PDF",
    "Official syllabus",
    "Student discussion",
    "Student professor/course discussion",
    "Student professor recommendation thread",
]


def format_sources(sources: list[dict]) -> str:
    """Render the programmatic source list as readable Markdown."""
    if not sources:
        return "_No sources retrieved._"

    lines = []
    for i, s in enumerate(sources, 1):
        title = s.get("source") or "Unknown source"
        bits = [b for b in (s.get("course"), s.get("type")) if b]
        suffix = f" ({', '.join(bits)})" if bits else ""
        url = s.get("url")
        if url:
            lines.append(f"{i}. [{title}]({url}){suffix}")
        else:
            lines.append(f"{i}. {title}{suffix}")
    return "\n".join(lines)


def _format_metric(c: dict) -> str:
    """Describe a chunk's relevance score(s) for the debug view.

    Semantic-only chunks carry a cosine `distance`; hybrid chunks also carry a
    fused `score` (and possibly a `bm25_score`). We show whatever is present so
    the same renderer works for both retrieval modes.
    """
    parts = []
    if c.get("score") is not None:
        parts.append(f"score={c['score']:.4f}")
    if c.get("distance") is not None:
        parts.append(f"distance={c['distance']:.4f}")
    if c.get("bm25_score") is not None:
        parts.append(f"bm25={c['bm25_score']:.2f}")
    return " ".join(parts) or "n/a"


def format_debug(chunks: list[dict]) -> str:
    """Render retrieved chunks with relevance scores for debugging."""
    if not chunks:
        return "_No chunks retrieved._"

    lines = []
    for rank, c in enumerate(chunks, 1):
        preview = c.get("text", "")[:300].replace("\n", " ")
        lines.append(
            f"**{rank}. {_format_metric(c)}** — "
            f"{c.get('source', '?')} "
            f"(`{c.get('chunk_id', '?')}`)\n\n{preview}…"
        )
    return "\n\n---\n\n".join(lines)


def history_to_messages(history: list[dict]) -> list[dict]:
    """Render the stored turn history as Gradio Chatbot `messages`."""
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    return messages


def respond(
    question: str,
    history: list[dict],
    course: str = ALL,
    source_type: str = ALL,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
):
    """Run the RAG pipeline for one chat turn.

    `history` is the conversational-memory State: a list of recent
    {"question", "answer"} turns. It is used only to build the retrieval query
    for terse follow-ups — the answer and sources still come from retrieved
    chunks alone.

    `course` / `source_type` come from the optional dropdowns; the "All"
    sentinel means no filter and is converted to None before retrieval.
    `retrieval_mode` chooses between "Semantic" and "Hybrid" retrieval.

    Returns (chatbot_messages, new_history, question_box, sources_md, debug_md).
    """
    history = history or []
    question = (question or "").strip()
    if not question:
        # Nothing to do: leave the conversation and panels untouched.
        return history_to_messages(history), history, "", gr.update(), gr.update()

    course_filter = None if course == ALL else course
    type_filter = None if source_type == ALL else source_type

    # Memory shapes ONLY the retrieval query; the answer prompt still gets the
    # literal current question (handled inside ask()).
    retrieval_query = build_contextual_query(question, history)

    result = ask(
        question,
        course_filter=course_filter,
        type_filter=type_filter,
        retrieval_mode=retrieval_mode,
        retrieval_query=retrieval_query,
    )
    answer = result["answer"]
    sources_md = format_sources(result["sources"])

    debug_md = ""
    if retrieval_query != question:
        debug_md += f"**Contextual retrieval query:** {retrieval_query}\n\n---\n\n"
    debug_md += format_debug(result["retrieved_chunks"])

    # Append this turn and keep only the last few for memory.
    new_history = (history + [{"question": question, "answer": answer}])[
        -MAX_MEMORY_TURNS:
    ]

    # Clear the question box after submitting.
    return history_to_messages(new_history), new_history, "", sources_md, debug_md


def clear_conversation():
    """Reset the chat, memory State, and result panels."""
    return [], [], "", "", ""


with gr.Blocks(title="The Unofficial Guide — ASU CS") as demo:
    gr.Markdown(
        "# The Unofficial Guide — ASU CS\n"
        "Ask about CSE 330 / 340 / 355 (the Trifecta). Answers are grounded "
        "only in retrieved sources — Reddit discussions and official ASU "
        "syllabi. Ask follow-up questions naturally; the assistant remembers "
        "the last few turns to interpret them."
    )
    gr.Markdown(MEMORY_NOTE)

    # Conversational memory: the last few {"question", "answer"} turns. Used
    # only to interpret follow-ups, never as a source of truth.
    history_state = gr.State([])

    chatbot = gr.Chatbot(label="Conversation", height=380)

    question_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. Who should I take for CSE 330 or CSE 355?",
        lines=2,
    )

    with gr.Row():
        course_dd = gr.Dropdown(
            choices=COURSE_CHOICES,
            value=ALL,
            label="Course (optional filter)",
        )
        type_dd = gr.Dropdown(
            choices=TYPE_CHOICES,
            value=ALL,
            label="Source type (optional filter)",
        )
        mode_dd = gr.Dropdown(
            choices=list(RETRIEVAL_MODES),
            value=DEFAULT_RETRIEVAL_MODE,
            label="Retrieval mode",
        )

    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear conversation")

    sources_box = gr.Markdown(label="Sources / retrieved from")

    with gr.Accordion("Retrieved chunks (debug — distance scores)", open=False):
        debug_box = gr.Markdown()

    # Button click and Enter key both submit. The answer appears in the chat
    # transcript; the sources/debug panels reflect the latest turn.
    inputs = [question_box, history_state, course_dd, type_dd, mode_dd]
    outputs = [chatbot, history_state, question_box, sources_box, debug_box]
    ask_btn.click(fn=respond, inputs=inputs, outputs=outputs)
    question_box.submit(fn=respond, inputs=inputs, outputs=outputs)

    clear_btn.click(
        fn=clear_conversation,
        inputs=None,
        outputs=[chatbot, history_state, question_box, sources_box, debug_box],
    )


if __name__ == "__main__":
    demo.launch()
