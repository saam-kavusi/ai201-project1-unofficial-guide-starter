"""
Milestone 5 — Gradio interface.

A simple Blocks UI over query.ask(): type a question, get a grounded answer,
the source list it was built from, and (optional) retrieved-chunk debug info
with distance scores. Both the Ask button and the Enter key submit.

Run with:  python app.py
"""

import gradio as gr

from query import ask

# Optional metadata filters. "All" means "do not apply this filter" and maps to
# None before reaching retrieval. The non-"All" values are exact matches against
# the `course` / `type` metadata stored on each chunk.
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


def format_debug(chunks: list[dict]) -> str:
    """Render retrieved chunks with distance scores for debugging."""
    if not chunks:
        return "_No chunks retrieved._"

    lines = []
    for rank, c in enumerate(chunks, 1):
        preview = c.get("text", "")[:300].replace("\n", " ")
        lines.append(
            f"**{rank}. distance={c.get('distance', 0):.4f}** — "
            f"{c.get('source', '?')} "
            f"(`{c.get('chunk_id', '?')}`)\n\n{preview}…"
        )
    return "\n\n---\n\n".join(lines)


def respond(question: str, course: str = ALL, source_type: str = ALL):
    """Run the RAG pipeline and return (answer, sources_md, debug_md).

    `course` / `source_type` come from the optional dropdowns; the "All"
    sentinel means no filter and is converted to None before retrieval.
    """
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", ""

    course_filter = None if course == ALL else course
    type_filter = None if source_type == ALL else source_type

    result = ask(question, course_filter=course_filter, type_filter=type_filter)
    answer = result["answer"]
    sources_md = format_sources(result["sources"])
    debug_md = format_debug(result["retrieved_chunks"])
    return answer, sources_md, debug_md


with gr.Blocks(title="The Unofficial Guide — ASU CS") as demo:
    gr.Markdown(
        "# The Unofficial Guide — ASU CS\n"
        "Ask about CSE 330 / 340 / 355 (the Trifecta). Answers are grounded "
        "only in retrieved sources — Reddit discussions and official ASU "
        "syllabi."
    )

    question_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. Is it a good idea to take CSE 340 in the summer with Bazzi?",
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

    ask_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown(label="Sources / retrieved from")

    with gr.Accordion("Retrieved chunks (debug — distance scores)", open=False):
        debug_box = gr.Markdown()

    # Button click and Enter key both submit.
    inputs = [question_box, course_dd, type_dd]
    outputs = [answer_box, sources_box, debug_box]
    ask_btn.click(fn=respond, inputs=inputs, outputs=outputs)
    question_box.submit(fn=respond, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
