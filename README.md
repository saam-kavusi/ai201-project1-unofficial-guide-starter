# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

This system covers unofficial ASU Computer Science course guidance for CSE 330, CSE 340, and CSE 355, with a focus on course difficulty, workload, professor experiences, summer classes, and the “trifecta” of taking these classes close together.

This knowledge is valuable because students often need practical advice before choosing classes, such as which professor to take, how difficult a course feels, how time-consuming projects are, and whether taking multiple hard CS classes together is realistic. Official sources like syllabi and major maps explain requirements, policies, and schedules, but they usually do not capture student experience, teaching style, exam difficulty, workload, or survival strategies.

This information is hard to find through official channels because it is spread across Reddit threads, course syllabi, major maps, and student comments. The goal of this project is to bring those sources together into one grounded RAG assistant that can answer student questions while still showing where the information came from.

---

## Document Sources


| # | Source | Type | Description | URL or location |
|---|--------|------|-------------|-----------------|
| 1 | ASU CS Major Map — Official ASU PDF | Official ASU degree map | Overall CS degree path and where courses like CSE 330, CSE 340, and CSE 355 fit in the program. | https://degrees.apps.asu.edu/major-map/ASU00/ESCSEBS/null/ONLINE/2024/fetchpdf |
| 2 | Reddit r/ASU — “Opinions on hardest CS major classes and future” | Student discussion | Student opinions on the hardest ASU CS classes and future course planning. | https://www.reddit.com/r/ASU/comments/14b4dsh/opinions_on_hardest_cs_major_classes_and_future/ |
| 3 | Reddit r/ASU — “What is the hardest class for CS majors? Why?” | Student discussion | Student opinions on the most difficult CS major courses and why they are challenging. | https://www.reddit.com/r/ASU/comments/1i8h9ss/what_is_the_hardest_class_for_cs_majors_why_do/ |
| 4 | Reddit r/ASU — “Review of different CS professors” | Student professor review thread | Student opinions about different ASU CS professors, including teaching style, difficulty, and recommendations. | https://www.reddit.com/r/ASU/comments/1ixpih7/review_of_different_cs_professors/ |
| 5 | Reddit r/ASU — “CSE 330/340/355 over summer” | Student discussion | Discussion about whether taking CSE 330, CSE 340, and CSE 355 together during summer is manageable. | https://www.reddit.com/r/ASU/comments/1nv0vsg/cse_330340355_over_summer/ |
| 6 | Reddit r/ASU — “CS majors taking the trifecta this summer” | Student discussion | Student discussion about the “trifecta” experience of taking CSE 330, CSE 340, and CSE 355 together. | https://www.reddit.com/r/ASU/comments/1kldo1z/cs_majors_taking_the_trifecta_this_summer/ |
| 7 | Reddit r/ASU — “How difficult is CSE 340?” | Student discussion | Student discussion about CSE 340 difficulty, workload, and advice. | https://www.reddit.com/r/ASU/comments/1knq3au/how_difficult_is_cse_340/ |
| 8 | Reddit r/ASU — “CSE 340” | Student discussion | General student discussion about CSE 340. | https://www.reddit.com/r/ASU/comments/1pnnsmk/cse_340/ |
| 9 | Reddit r/ASU — “Which is most challenging: CSE 355, CSE 360, or CSE ___?” | Student discussion | Student discussion comparing the difficulty of CSE 355 with other ASU CS courses. | https://www.reddit.com/r/ASU/comments/1crjzby/which_is_most_challenging_cse_355_cse_360_or_cse/ |
| 10 | ASU SCAI — CSE 330 Syllabus SP25 | Official syllabus PDF | Official CSE 330 course structure, topics, assignments, grading, and expectations. | https://scai.engineering.asu.edu/wp-content/uploads/sites/31/2025/03/CSE-330-Sylllabus-SP25.pdf |
| 11 | ASU SCAI — CSE 340 Syllabus SP25 | Official syllabus PDF | Official CSE 340 course structure, topics, assignments, grading, and expectations. | https://scai.engineering.asu.edu/wp-content/uploads/sites/31/2025/03/CSE-340-Syllabus-SP25.pdf |
| 12 | ASU SCAI — CSE 355 Syllabus SP25 | Official syllabus PDF | Official CSE 355 course structure, topics, assignments, grading, and expectations. | https://scai.engineering.asu.edu/wp-content/uploads/sites/31/2025/03/CSE-355-Syllabus-SP25.pdf |
| 13 | Reddit r/ASU — “CSE 340 with Bazzi during the summer” | Student professor/course discussion | Student opinions about taking CSE 340 with Bazzi during summer, including professor-specific expectations and workload. | https://www.reddit.com/r/ASU/comments/1rr8aya/cse340_with_bazzi_during_the_summer/ |
| 14 | Reddit r/ASU — “CSE 330 and 355 profs” | Student professor recommendation thread | Student recommendations and opinions about professors for CSE 330 and CSE 355. | https://www.reddit.com/r/ASU/comments/178cksm/cse_330_and_355_profs/ |
---

## Chunking Strategy

**Chunk size:**  
I used paragraph-aware chunks targeting about 250 tokens per chunk. In the code, this is approximated with character-based limits, but the goal was to keep each chunk around 200–300 tokens. This size is large enough to preserve meaning from Reddit comments, syllabus sections, and course descriptions, but small enough that retrieval can return focused results.

**Overlap:**  
I used about 50 tokens of overlap between chunks. The overlap helps prevent important context from being lost when a topic spans across chunk boundaries, such as a Reddit post followed by related comments or a syllabus policy that continues into the next paragraph.

**Why these choices fit your documents:**  
The document set includes a mix of official PDFs, syllabi, and informal Reddit discussions. Reddit threads are naturally organized by posts and comments, while syllabi are organized by sections and policies. Because of that, the chunking strategy tries to preserve paragraph and comment boundaries instead of cutting text randomly. Before chunking, the pipeline cleans the extracted text by removing extra whitespace, control characters, broken PDF artifacts, and other noisy formatting. This makes the chunks easier to embed and retrieve reliably.

**Final chunk count:**  
The final production dataset contains 90 chunks across 14 documents.

---

## Embedding Model

## Embedding Model

**Model used:**  
I used `sentence-transformers/all-MiniLM-L6-v2` to embed the document chunks. I chose this model because it is lightweight, fast, free to run locally, and commonly used for semantic search projects. It produces good-quality embeddings for short to medium text chunks, which fits this project because my chunks are around 200–300 tokens and mostly contain Reddit comments, syllabus sections, and course descriptions.

**Production tradeoff reflection:**  
If I were deploying this system for real users and cost was not a constraint, I would compare this model against larger or API-hosted embedding models. A larger model might improve retrieval accuracy, especially for vague student questions, professor names, and mixed official/unofficial language. I would also consider context length limits, since longer-context embedding models could better handle full syllabus sections or long Reddit comments without splitting them as much.

Multilingual support would matter if students asked questions in other languages or if sources included multilingual content. Latency would also be important: a local model like `all-MiniLM-L6-v2` is fast and cheap, but a stronger hosted model might improve accuracy at the cost of API latency, dependency on an external service, and higher cost. For this class project, the local model was the right balance of speed, simplicity, and retrieval quality.

---

## Grounded Generation

**System prompt grounding instruction:**  
The generation step uses a system prompt that explicitly limits the model to the retrieved context. The prompt tells the model to answer only from the numbered context passages, not from outside knowledge. It also tells the model that if the retrieved context does not contain enough information, it must respond with the exact sentence:

> I don't have enough information in the provided sources to answer that.

The prompt also tells the model not to invent professor names, course policies, grades, assignments, or recommendations. If the sources disagree or are only student opinions, the model is instructed to mention that uncertainty instead of presenting the answer as official fact.

Structurally, the retrieved chunks are formatted as numbered context blocks before being sent to the LLM. Each block includes the source metadata and chunk text, so the model receives only the evidence that retrieval found. Conversational memory, when used, only helps form a better retrieval query for follow-up questions; it is not treated as a factual source in the LLM prompt. The final answer is still generated only from the retrieved chunks.

**How source attribution is surfaced in the response:**  
Source attribution is programmatically generated from the retrieved chunk metadata rather than left for the LLM to invent. After the model generates an answer, the app builds a source list directly from the chunks returned by retrieval. Each source entry can include the source title, URL, course, type, filename, and chunk IDs. This means the source panel reflects what was actually retrieved, even if the LLM does not mention citations in the answer itself.

The Gradio interface shows the answer separately from a “Sources / retrieved from” section, and it also includes a debug section showing the retrieved chunks and distance or hybrid scores. This makes it possible to inspect whether the answer was grounded in relevant documents.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy | |---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is it a good idea to take CSE 340 in the summer with Bazzi? | The answer should explain that CSE 340 with Bazzi can be doable but demanding. It should mention that students describe Bazzi positively, but also say the projects are difficult/time-consuming and students should start early, use notes/practice materials, and avoid falling behind. | The system said CSE 340 with Bazzi in the summer is challenging but doable with preparation and time management. It summarized student comments saying Bazzi is a strong professor, his notes are useful, and the projects can take significant time. It also recommended starting projects early and using office hours. | Relevant | Accurate |
| 2 | How bad is it to take CSE 330, CSE 340, and CSE 355 together in the summer? | The answer should warn that taking all three together, especially in summer, is very difficult because the workload is compressed and the classes are commonly viewed as demanding. It should rely on student discussion rather than presenting this as an official rule. | The system summarized student discussion saying the “trifecta” of CSE 330, CSE 340, and CSE 355 over the summer is a heavy workload and likely risky. It emphasized that the concern comes from student experience and that the compressed summer pace makes the combination harder. | Relevant | Accurate |
| 3 | Which one is harder: CSE 340 or CSE 355? | The answer should explain that students compare the courses differently: CSE 355 is often described as more conceptually difficult, while CSE 340 is described as requiring more coding effort and project time. It should avoid claiming a universal answer.| The system said students disagree somewhat, but one retrieved student comment directly compared them: CSE 355 is more conceptually difficult, while CSE 340 takes a lot of coding effort. It correctly framed this as student opinion rather than an official ranking. | Relevant | Accurate|
| 4 | Who should I take for CSE 330 or CSE 355?| The answer should summarize professor recommendations from student threads. It should mention that some students recommend Lee or Feng for CSE 355, while comments about CSE 330 include mixed views such as Gordon being easier but less useful for learning.| The system summarized student opinions about CSE 330 and CSE 355 professors. It mentioned Gordon as potentially easier but not as strong for learning, and it described positive comments about Lee and Feng for CSE 355. It also noted that these are student opinions and may depend on learning style.| Relevant | Accurate |
| 5 | What should I expect from CSE 340 based on both the syllabus and student reviews? | The answer should combine official syllabus-style expectations with student experience. It should mention course structure/workload at a high level, coding/project effort, exams or practice materials if supported, and distinguish official information from student reviews.| The system combined CSE 340 syllabus information with Reddit student comments. It described CSE 340 as a course with significant programming/project workload and student-reported difficulty, while also using official syllabus chunks for policies/course expectations. It was useful, but some details depended more heavily on student reviews than official course documents. | Relevant| Partially accurate |


**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**  
With the Course filter set to `CSE 355`, I asked:  
“Who should I take for CSE 330 or CSE 355?”  
Then I followed up with:  
“What about Lee?”

**What the system returned:**  
Before the fix, the system sometimes refused or returned a weak answer because it did not retrieve the best professor recommendation thread. The relevant source, `cse330_cse355_teachers.txt`, did not appear even though it contained useful CSE 355 professor discussion about Lee and Feng.

**Root cause (tied to a specific pipeline stage):**  
The failure came from the metadata filtering stage of retrieval. My course filter originally used exact metadata equality. That worked for chunks labeled exactly `CSE 355`, but some of my best chunks had compound course metadata such as `CSE 330, CSE 355`. When the user selected Course = `CSE 355`, the retrieval filter excluded those compound-tagged chunks even though they were relevant. Because the retrieval stage removed the best evidence before generation, the LLM did not have enough context to answer the follow-up correctly.

**What you would change to fix it:**  
I changed the course filter from exact matching to inclusive matching. Now Course = `CSE 355` matches chunks labeled `CSE 355`, `CSE 330, CSE 355`, or other metadata strings that contain `CSE 355`. I kept source type and filename filters exact because those fields do not need substring matching. This fixed the issue while preserving the ability to filter by course. In a larger production system, I would improve this further by storing course metadata as a list of normalized course tags instead of a single string.

---

## Spec Reflection

**One way the spec helped you during implementation:**  
The spec in `planning.md` helped keep the project focused on a specific domain instead of becoming a general ASU chatbot. By defining the domain as CSE 330, CSE 340, CSE 355, professor experiences, summer workload, and the “trifecta,” I knew which sources to collect and which questions the system should be able to answer. The evaluation questions in the spec also gave me a clear way to test whether retrieval was actually working before adding generation.

**One way your implementation diverged from the spec, and why:**  
My original spec planned a basic semantic retrieval system with `top_k=5`, but the final implementation went beyond that by adding metadata filtering, hybrid search, chunking comparison, and conversational memory. I added these because testing showed real weaknesses in the basic version, such as broad major-map chunks appearing for professor or difficulty questions and exact course filters missing compound course tags like `CSE 330, CSE 355`. These changes kept the original goal the same, but made the system more useful and easier to evaluate honestly.
---

## AI Usage

**Instance 1**

- *What I gave the AI:*  
  I gave Claude my project domain, source list, and chunking strategy from `planning.md`. I asked it to implement the ingestion and chunking pipeline for the documents in `documents/raw`, including support for both `.txt` and `.pdf` files, metadata extraction, text cleaning, and outputting chunks to `data/chunks.json`.

- *What it produced:*  
  Claude produced `ingest.py`, which loaded the raw documents, extracted metadata such as source, URL, course, type, filename, and chunk ID, cleaned the text, and generated paragraph-aware chunks. It also created `data/chunks.json` with 90 chunks across 14 documents.

- *What I changed or overrode:*  
  I did not accept the first output blindly. I tested the generated chunks and found PDF extraction issues, including corrupted text from the CSE 330 syllabus and noisy glyph artifacts from another syllabus. I directed Claude to add targeted cleanup rules and replaced the broken CSE 330 PDF extraction with a cleaned text version. I also verified that the final chunks had valid metadata, no empty chunks, no duplicate chunks, and reasonable chunk lengths before committing.

**Instance 2**

- *What I gave the AI:*  
  I gave Claude the Milestone 5 requirement to connect retrieval to an LLM and build a Gradio interface. I specifically told it that answers must be grounded in retrieved context only, that unsupported questions should be refused, and that source attribution needed to be generated programmatically instead of trusting the LLM to cite sources.

- *What it produced:*  
  Claude produced `query.py` and `app.py`. `query.py` connected retrieval to Groq’s `llama-3.3-70b-versatile` model, created the grounded system prompt, formatted retrieved chunks as context, and returned answers with sources. `app.py` created a Gradio interface with a question box, answer display, source list, and retrieved-chunk debug section.

- *What I changed or overrode:*  
  I checked that the grounding prompt was strict enough and that the source list came from retrieved chunk metadata instead of from the model’s generated text. Later, when testing showed that exact course filtering excluded compound metadata like `CSE 330, CSE 355`, I directed Claude to change course filtering to inclusive matching while keeping source type and filename filters exact. I also added tests to confirm that CSE 355 filters could still retrieve professor chunks from combined CSE 330/CSE 355 sources.
