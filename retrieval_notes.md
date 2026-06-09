# Milestone 4 — Retrieval Notes

Validation observations from running `vector_store.py` retrieval against the
evaluation questions. These are notes only; retrieval code is unchanged.

## Professor-review source check

Triggered by the weak performance of the query
**"Who should I take for CSE 330 or CSE 355?"** (off-topic `asu_cs_major_map.pdf`
chunks appearing at ranks 3 and 5).

- **The professor-review source is present** in `data/chunks.json`:
  `cs_teachers_review.txt` → "Reddit r/ASU — Review of different CS professors"
  (`https://www.reddit.com/r/ASU/comments/1ixpih7/review_of_different_cs_professors/`).
  This was *not* a missing-source / coverage gap.
- It contributes **9 chunks** (`cs_teachers_review.txt__chunk0` … `__chunk8`).
- **Only `chunk5` is relevant to CSE 330** (it reviews Ming Zhao). The other 8
  chunks review unrelated courses (CSE 205, 240, 259, 230, 310, 360), and the
  thread contains **no CSE 355 professor content at all**.

## Why it ranks low for the 330/355 professor query

The one relevant chunk (`chunk5`) ranks worst among the nine
(~rank 62, distance ~0.81), well outside the top 5, for two reasons:

1. **It crosses a course boundary** — the chunk opens with the tail of the
   CSE 310 review before reaching the CSE 330 content, diluting the "CSE 330"
   signal.
2. **The wording is complaint-style, not recommendation-style** ("lazy
   instructor," "no slides," "Google Groups"), so it embeds far from the
   intent of a "who should I take" question.

## What still works

The dedicated **`cse330_cse355_teachers.txt`** thread ("CSE 330 and 355 profs")
retrieves correctly at **ranks 1–2** (distances ~0.32 and ~0.47), so the query's
top results are still on-target. The corpus simply thins out after those two
chunks, which is why the major-map tables backfill ranks 3 and 5.

## Follow-up for stretch features

This is a useful weakness to revisit later — no code change now:

- **Chunking strategy comparison:** the boundary-crossing `chunk5` shows that
  comment-aware or per-professor chunking could keep a single course review
  intact and improve its ranking.
- **Metadata filtering:** filtering professor-recommendation queries to
  discussion-type sources (and away from the `asu_cs_major_map.pdf` course
  tables) would remove the off-topic backfill at ranks 3 and 5.
