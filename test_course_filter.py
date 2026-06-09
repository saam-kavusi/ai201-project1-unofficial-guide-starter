"""
Tests for inclusive (contains-based) course filtering.

Background: course metadata can be compound, e.g. "CSE 330, CSE 355" or
"General ASU CS / CSE 330, CSE 340, CSE 355". An exact-equality course filter
excluded those chunks when the user filtered by a single course (e.g.
"CSE 355"), so cross-listed professor/recommendation threads disappeared. The
fix makes course filtering a substring ("contains") match while keeping
`type` and `filename` exact.

These are plain assert-based tests (the project has no pytest dependency); run
directly:

    python test_course_filter.py

Two layers:
  * Unit tests — pure logic (_course_matches, _build_where). Fast, no model/DB.
  * Retrieval tests — exercise the real semantic, keyword, and hybrid paths
    against the persisted Chroma store. They load the embedding model, so they
    are slower, but they make NO LLM/Groq calls (filtering is a retrieval-layer
    concern, so we assert on retrieved chunks rather than generated answers).
"""

import vector_store as vs
from query import build_contextual_query

CSE330_355_TEACHERS = "cse330_cse355_teachers.txt"
REC_THREAD_TYPE = "Student professor recommendation thread"


# ---------------------------------------------------------------------------
# Unit tests — pure logic, no embedding model or vector store needed
# ---------------------------------------------------------------------------

def test_course_matches_inclusive():
    """A single-course filter matches exact AND compound course tags."""
    assert vs._course_matches("CSE 355", "CSE 355")
    assert vs._course_matches("CSE 330, CSE 355", "CSE 355")
    assert vs._course_matches(
        "General ASU CS / CSE 330, CSE 340, CSE 355", "CSE 355"
    )
    # Non-matching course is still excluded.
    assert not vs._course_matches("CSE 340", "CSE 355")
    assert not vs._course_matches("CSE 330", "CSE 355")


def test_course_matches_no_filter_matches_all():
    """An empty/None filter matches everything (unconstrained behavior)."""
    assert vs._course_matches("CSE 340", None)
    assert vs._course_matches("anything", "")
    assert vs._course_matches(None, None)
    assert vs._course_matches("", None)


def test_build_where_excludes_course():
    """Course is post-filtered in Python and must NOT enter the Chroma where."""
    # Course alone => no where clause at all.
    assert vs._build_where(type_filter=None, filename_filter=None) is None

    # type/filename stay as exact matches.
    assert vs._build_where(type_filter="Official PDF", filename_filter=None) == {
        "type": "Official PDF"
    }
    where = vs._build_where(type_filter="Official PDF", filename_filter="a.pdf")
    assert where == {"$and": [{"type": "Official PDF"}, {"filename": "a.pdf"}]}


def test_chunk_matches_course_inclusive_type_exact():
    """Keyword-path filter: course inclusive, type/filename exact."""
    chunk = {
        "course": "CSE 330, CSE 355",
        "type": REC_THREAD_TYPE,
        "filename": CSE330_355_TEACHERS,
    }
    # Inclusive course match survives the compound tag.
    assert vs._chunk_matches(chunk, "CSE 355", None, None)
    # Exact type match passes; a near-miss type is rejected.
    assert vs._chunk_matches(chunk, "CSE 355", REC_THREAD_TYPE, None)
    assert not vs._chunk_matches(chunk, "CSE 355", "Student discussion", None)
    # Exact filename behavior preserved.
    assert vs._chunk_matches(chunk, "CSE 355", None, CSE330_355_TEACHERS)
    assert not vs._chunk_matches(chunk, "CSE 355", None, "other.txt")


# ---------------------------------------------------------------------------
# Retrieval tests — real semantic/keyword/hybrid paths (no LLM calls)
# ---------------------------------------------------------------------------

def _filenames(chunks):
    return {c["filename"] for c in chunks}


def _courses(chunks):
    return {c["course"] for c in chunks}


def test_compound_course_chunk_surfaces_with_single_filter():
    """Course=CSE 355 + "Who should I take..." retrieves the cross-listed thread.

    The cse330_cse355_teachers thread is tagged "CSE 330, CSE 355"; under the
    old exact filter it was excluded. It must now appear in hybrid retrieval.
    """
    chunks = vs.retrieve_chunks_hybrid(
        "Who should I take for CSE 330 or CSE 355?",
        top_k=5,
        course_filter="CSE 355",
    )
    assert chunks, "expected results for CSE 355 filter"
    assert CSE330_355_TEACHERS in _filenames(chunks), (
        f"compound-tagged thread missing; got {_filenames(chunks)}"
    )
    # Every survivor genuinely contains the filtered course.
    assert all("CSE 355" in c["course"] for c in chunks), _courses(chunks)


def test_followup_about_lee_retrieves_355_professor_chunk():
    """Course=CSE 355 follow-up "What about Lee?" surfaces a 355 professor chunk.

    Mirrors the conversational-memory path: the retrieval query is built from
    the previous question + the terse follow-up, then filtered to CSE 355.
    """
    history = [{"question": "Who should I take for CSE 330 or CSE 355?"}]
    retrieval_query = build_contextual_query("What about Lee?", history)

    chunks = vs.retrieve_chunks_hybrid(
        retrieval_query, top_k=5, course_filter="CSE 355"
    )
    assert chunks, "expected results for CSE 355 follow-up"
    assert CSE330_355_TEACHERS in _filenames(chunks), (
        f"expected CSE 355 professor thread; got {_filenames(chunks)}"
    )
    assert all("CSE 355" in c["course"] for c in chunks), _courses(chunks)


def test_cse340_bazzi_still_returns_cse340():
    """Regression: Course=CSE 340 + Bazzi still returns only CSE 340 sources."""
    for retrieve in (vs.retrieve_chunks, vs.retrieve_chunks_hybrid):
        chunks = retrieve(
            "Is CSE 340 good in summer with Bazzi?",
            top_k=5,
            course_filter="CSE 340",
        )
        assert chunks, f"{retrieve.__name__}: expected CSE 340 results"
        assert all("CSE 340" in c["course"] for c in chunks), (
            f"{retrieve.__name__}: {_courses(chunks)}"
        )


def test_type_filter_remains_exact():
    """Type filtering stays an exact match across all three retrieval paths."""
    q = "Who should I take for CSE 330 or CSE 355?"
    for chunks in (
        vs.retrieve_chunks(q, top_k=8, type_filter=REC_THREAD_TYPE),
        vs.keyword_retrieve(q, top_k=8, type_filter=REC_THREAD_TYPE),
        vs.retrieve_chunks_hybrid(q, top_k=8, type_filter=REC_THREAD_TYPE),
    ):
        assert chunks, "expected results for the type filter"
        assert all(c["type"] == REC_THREAD_TYPE for c in chunks), (
            {c["type"] for c in chunks}
        )


def test_no_filters_unconstrained():
    """Backwards compatibility: no filters returns exactly top_k chunks."""
    chunks = vs.retrieve_chunks("CSE 340 summer", top_k=5)
    assert len(chunks) == 5


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    # Ensure the vector store exists before the retrieval tests run.
    vs.build_vector_store()

    tests = [
        test_course_matches_inclusive,
        test_course_matches_no_filter_matches_all,
        test_build_where_excludes_course,
        test_chunk_matches_course_inclusive_type_exact,
        test_compound_course_chunk_surfaces_with_single_filter,
        test_followup_about_lee_retrieves_355_professor_chunk,
        test_cse340_bazzi_still_returns_cse340,
        test_type_filter_remains_exact,
        test_no_filters_unconstrained,
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 — surface any error per-test
            failures += 1
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"PASS  {test.__name__}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
