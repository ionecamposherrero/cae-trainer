# Reference corpus

Documents the coach chat searches when a learner asks about the exam or the English
language. Learner data never goes here; the coach reads it from the database.

`c1_advanced_structure.md` is included. Everything else in this directory is gitignored:
Cambridge material is not redistributable, so supply your own copy.

## Adding documents

- Use Markdown (`.md`) or plain text (`.txt`); convert PDFs first.
- Structure documents with Markdown headings. Each chunk is cited by its heading path, for
  example `handbook.md — Reading and Use of English Part 4 > THE TASK`.
- Prefer material a model cannot reliably recall: exam formats, task conventions and
  marking criteria.
- Do not add past papers. They are not redistributable, and the generator must not
  reproduce published material.

## Building the index

```bash
python scripts/build_index.py --dry-run   # preview chunks and estimate cost
python scripts/build_index.py             # embed and persist to data/chroma/
python scripts/build_index.py --check     # what is indexed right now
```

The index is rebuilt from scratch every time. Without it, the coach answers exam questions
from the model's memory, which can be wrong.
