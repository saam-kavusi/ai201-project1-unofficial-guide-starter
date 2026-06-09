# Chunking Strategy Comparison

Stretch feature. Compares three paragraph-aware token chunking sizes on the same five evaluation questions from `planning.md`, embedded with `all-MiniLM-L6-v2` and retrieved from a temporary in-memory ChromaDB collection (cosine distance — lower is better). The production app, `data/chunks.json`, and `data/chroma_db` are untouched.


## Small  (~150 tok / 30 overlap)

- Chunks: **147**
- Average chunk length: **~130 tokens** (520 chars)


**Q1. Is it a good idea to take CSE 340 in the summer with Bazzi?**  
_Expected courses: CSE 340_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3366 | `cse340_teacher_bazzi.txt__chunk0` | CSE 340 | Reddit r/ASU — CSE 340 with Bazzi during the summer |
| 2 | 0.4260 | `cse340_advice.txt__chunk0` | CSE 340 | Reddit r/ASU — CSE 340 |
| 3 | 0.4614 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk1` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |

Top-1 relevant by course/title? **YES**


**Q2. How bad is it to take CSE 330, CSE 340, and CSE 355 together in the summer?**  
_Expected courses: CSE 330, CSE 340, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.4059 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk1` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |
| 2 | 0.4344 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 3 | 0.4439 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk2` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |

Top-1 relevant by course/title? **YES**


**Q3. Which one is harder: CSE 340 or CSE 355?**  
_Expected courses: CSE 340, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.2621 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 2 | 0.3426 | `cse340_difficulty.txt__chunk0` | CSE 340 | Reddit r/ASU — How difficult is CSE 340? |
| 3 | 0.3874 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk2` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |

Top-1 relevant by course/title? **YES**


**Q4. Who should I take for CSE 330 or CSE 355?**  
_Expected courses: CSE 330, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.2486 | `cse330_cse355_teachers.txt__chunk0` | CSE 330, CSE 355 | Reddit r/ASU — CSE 330 and 355 profs |
| 2 | 0.3937 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk1` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |
| 3 | 0.4199 | `cse340_difficulty.txt__chunk0` | CSE 340 | Reddit r/ASU — How difficult is CSE 340? |

Top-1 relevant by course/title? **YES**


**Q5. What should I expect from CSE 340 based on both the syllabus and student reviews?**  
_Expected courses: CSE 340_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3505 | `cse340_difficulty.txt__chunk0` | CSE 340 | Reddit r/ASU — How difficult is CSE 340? |
| 2 | 0.3888 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 3 | 0.3991 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk2` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |

Top-1 relevant by course/title? **YES**


## Default(~250 tok / 50 overlap)

- Chunks: **90**
- Average chunk length: **~220 tokens** (881 chars)


**Q1. Is it a good idea to take CSE 340 in the summer with Bazzi?**  
_Expected courses: CSE 340_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3366 | `cse340_teacher_bazzi.txt__chunk0` | CSE 340 | Reddit r/ASU — CSE 340 with Bazzi during the summer |
| 2 | 0.4273 | `cse340_difficulty.txt__chunk2` | CSE 340 | Reddit r/ASU — How difficult is CSE 340? |
| 3 | 0.4488 | `cse340_advice.txt__chunk0` | CSE 340 | Reddit r/ASU — CSE 340 |

Top-1 relevant by course/title? **YES**


**Q2. How bad is it to take CSE 330, CSE 340, and CSE 355 together in the summer?**  
_Expected courses: CSE 330, CSE 340, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.4573 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk1` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |
| 2 | 0.4722 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk0` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |
| 3 | 0.4824 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |

Top-1 relevant by course/title? **YES**


**Q3. Which one is harder: CSE 340 or CSE 355?**  
_Expected courses: CSE 340, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3851 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 2 | 0.4710 | `asu_cs_major_map.pdf__chunk4` | General ASU CS | asu_cs_major_map.pdf |
| 3 | 0.4745 | `cse330_cse355_teachers.txt__chunk0` | CSE 330, CSE 355 | Reddit r/ASU — CSE 330 and 355 profs |

Top-1 relevant by course/title? **YES**


**Q4. Who should I take for CSE 330 or CSE 355?**  
_Expected courses: CSE 330, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3150 | `cse330_cse355_teachers.txt__chunk0` | CSE 330, CSE 355 | Reddit r/ASU — CSE 330 and 355 profs |
| 2 | 0.4661 | `cse330_cse355_teachers.txt__chunk1` | CSE 330, CSE 355 | Reddit r/ASU — CSE 330 and 355 profs |
| 3 | 0.4900 | `asu_cs_major_map.pdf__chunk4` | General ASU CS | asu_cs_major_map.pdf |

Top-1 relevant by course/title? **YES**


**Q5. What should I expect from CSE 340 based on both the syllabus and student reviews?**  
_Expected courses: CSE 340_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.4037 | `cse_hardest_classes_class_reviews.txt__chunk2` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 2 | 0.4145 | `CSE340_Syllabus_SP25.pdf__chunk0` | CSE 340 | CSE340_Syllabus_SP25.pdf |
| 3 | 0.4158 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |

Top-1 relevant by course/title? **YES**


## Large  (~400 tok / 80 overlap)

- Chunks: **62**
- Average chunk length: **~315 tokens** (1260 chars)


**Q1. Is it a good idea to take CSE 340 in the summer with Bazzi?**  
_Expected courses: CSE 340_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3776 | `cse340_teacher_bazzi.txt__chunk0` | CSE 340 | Reddit r/ASU — CSE 340 with Bazzi during the summer |
| 2 | 0.4488 | `cse340_advice.txt__chunk0` | CSE 340 | Reddit r/ASU — CSE 340 |
| 3 | 0.4938 | `cse340_difficulty.txt__chunk1` | CSE 340 | Reddit r/ASU — How difficult is CSE 340? |

Top-1 relevant by course/title? **YES**


**Q2. How bad is it to take CSE 330, CSE 340, and CSE 355 together in the summer?**  
_Expected courses: CSE 330, CSE 340, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.4699 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 2 | 0.4790 | `cse330_cse340_cse355_all_three_in_the_summer.txt__chunk0` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CSE 330/340/355 over summer |
| 3 | 0.4799 | `cse330_cse340_cse355_summer.txt__chunk3` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CS majors taking the trifecta this summer |

Top-1 relevant by course/title? **YES**


**Q3. Which one is harder: CSE 340 or CSE 355?**  
_Expected courses: CSE 340, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3834 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 2 | 0.4341 | `cse330_cse340_cse355_summer.txt__chunk3` | CSE 330, CSE 340, CSE 355 | Reddit r/ASU — CS majors taking the trifecta this summer |
| 3 | 0.4552 | `cse330_cse355_teachers.txt__chunk0` | CSE 330, CSE 355 | Reddit r/ASU — CSE 330 and 355 profs |

Top-1 relevant by course/title? **YES**


**Q4. Who should I take for CSE 330 or CSE 355?**  
_Expected courses: CSE 330, CSE 355_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3285 | `cse330_cse355_teachers.txt__chunk0` | CSE 330, CSE 355 | Reddit r/ASU — CSE 330 and 355 profs |
| 2 | 0.5101 | `cse340_difficulty.txt__chunk0` | CSE 340 | Reddit r/ASU — How difficult is CSE 340? |
| 3 | 0.5108 | `cse_hardest_classes_class_reviews.txt__chunk0` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |

Top-1 relevant by course/title? **YES**


**Q5. What should I expect from CSE 340 based on both the syllabus and student reviews?**  
_Expected courses: CSE 340_

| Rank | Distance | chunk_id | Course | Source |
|------|----------|----------|--------|--------|
| 1 | 0.3964 | `cse_hardest_classes_class_reviews.txt__chunk1` | General ASU CS / CSE 330, CSE 340, CSE 355 | Reddit r/ASU — What is the hardest class for CS majors? Why? |
| 2 | 0.4118 | `cse340_difficulty.txt__chunk0` | CSE 340 | Reddit r/ASU — How difficult is CSE 340? |
| 3 | 0.4145 | `CSE340_Syllabus_SP25.pdf__chunk0` | CSE 340 | CSE340_Syllabus_SP25.pdf |

Top-1 relevant by course/title? **YES**


## Summary

| Strategy | # chunks | Avg chunk length | Avg top-1 distance | Top-1 relevant | Notes |
|----------|----------|------------------|--------------------|----------------|-------|
| Small  (~150 tok / 30 overlap) | 147 | ~130 tok (520 chars) | 0.3207 | 5/5 | lowest avg top-1 distance; most top-1 relevant hits |
| Default(~250 tok / 50 overlap) | 90 | ~220 tok (881 chars) | 0.3795 | 5/5 | most top-1 relevant hits |
| Large  (~400 tok / 80 overlap) | 62 | ~315 tok (1260 chars) | 0.3912 | 5/5 | most top-1 relevant hits |

_Lower average top-1 distance means the nearest retrieved chunk is, on average, semantically closer to the query. "Top-1 relevant" counts how many of the five questions returned a top result whose course metadata or text mentions an expected course code — a coarse on-topic signal, not a precision score._

