# Datasets

One folder per dataset: its README (what is specific to this repository), `manifest.json` (revision, checksums,
licence) and `raw/` (fetched, git-ignored). Today there is one: [`multihop-rag/`](multihop-rag/README.md).

## Candidates for the next ones

Researched on 20261029 (the last clock reading of that research was 17:38 UTC) from the sources linked. **Nothing here
has been run through Pinakes**; this ranks datasets by their properties. What a dataset needs to be useful here: relevance
labels that name documents (so a result maps to a file), real multi-chunk documents, room to improve (the demo and RFC
corpora in Pinakes saturate), no overlap with what the embedding models were trained on, and a fetch that can be pinned
and checksummed, since third-party data is never committed.

| Dataset | Why it would help | Main caveat | Source, licence |
|---|---|---|---|
| **FreshStack** | The most headroom on record: 672 questions over code and Markdown documentation, corpora of 25,482 (godot) to 117,288 (angular) chunks. In the paper's Table 4 the first-stage recall@50 is 0.20 to 0.44 for BM25 and 0.48 to 0.65 for a four-model fusion (read through a fetch tool: re-check before quoting). | Relevance is judged by GPT-4o per chunk against nuggets, not by people. Questions are long Stack Overflow posts. The corpus is pre-chunked (chunk ids look like `<repo path>_<start>_<end>`, inferred from examples), and has no null, filter or multi-hop questions. | [arXiv:2504.13128](https://arxiv.org/abs/2504.13128), [questions](https://huggingface.co/datasets/freshstack/queries-oct-2024), [corpus](https://huggingface.co/datasets/freshstack/corpus-oct-2024). Cards say CC-BY-SA-4.0; the source repositories carry their own licences, AGPL-3.0 among them. |
| **MuSiQue-Ans, dev** | Its `question_decomposition` names the supporting paragraph of each step (`paragraph_support_idx`), which maps onto Pinakes' scripted `hops`. 2,417 questions of 2, 3 and 4 hops (1,252 / 760 / 405), measured on the `fladhak/musique` copy. | The sub-questions are relation chains (`Green >> performer`, `#1 >> spouse`): each hop's query has to be built by substituting the previous answer. All questions are answerable and the paragraphs are short. | [arXiv:2108.00573](https://arxiv.org/abs/2108.00573); CC-BY-4.0 on the card of `fladhak/musique` (not re-read here). |
| **MTRAG-UN / MTRAG** | The only candidate where people judged a question unanswerable *against the corpus*: 666 / 842 tasks, about 28% / 15% unanswerable or partly answerable. Retrieval is far from solved (BM25 recall@5 0.25, BGE-base 0.37, ELSER 0.52, query-rewrite setting). | Multi-turn: use the rewritten standalone question. The four MTRAG corpora hold 49,607 to 183,408 passages (MTRAG-UN adds two more, not sized here). The paper's table and the GitHub page give the same four sets of counts under different corpus names, so check before building. The corpora's own licences were not stated on the page read. | [arXiv:2501.03468](https://arxiv.org/abs/2501.03468), [MTRAG-UN](https://arxiv.org/abs/2602.23184), [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark) (Apache-2.0 for the repository). |
| **SQuAD 2.0** | Adds unanswerable questions written to look answerable (11,873 validation questions in all). | They are unanswerable relative to *one* paragraph, not to a pooled corpus, so another paragraph may answer them (my inference, not a sourced claim). | [arXiv:1806.03822](https://arxiv.org/abs/1806.03822); CC BY-SA 4.0 on the [card](https://huggingface.co/datasets/rajpurkar/squad_v2). |
| **BrowseComp-Plus** | A testbed for the paid `--deep` loop: 830 questions over 100,195 documents, each with about 6.1 evidence documents, 2.9 gold and 76 hard negatives. | The corpus is 1.76 GB at about 5,179 words a document, so it would be subsampled; questions and answers are obfuscated and built for agents, so single-shot retrieval will sit near the floor. | [arXiv:2508.06600](https://arxiv.org/abs/2508.06600), [corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus); MIT. |
| **Open RAGBench** | Exercises PDF extraction as well as retrieval: 1,000 arXiv PDFs (400 with questions, 600 hard negatives), 3,045 questions, section-level relevance. | Non-commercial licence, so private use only; no unanswerable questions. | [card](https://huggingface.co/datasets/vectara/open_ragbench); CC BY-NC 4.0. |

Looked at and **not** recommended as the main instrument, measured on local copies of their Hub validation splits:

- **SQuAD 1.1** has only 48 articles in 2,067 paragraphs (median 700 characters, so one chunk each and the chunker is never
  exercised), 5 questions per paragraph, and the gold answer string also appears in another paragraph of the same article for
  30.7% of questions (an upper bound on ambiguous labels).
- **rag-dataset-1200** has exactly one question per document, and for 55% of them at least 80% of the question's content words
  appear in its document; its card says the questions were generated by GPT-4 over web text.

Not pursued, for size or licence, and **unverified here**: Natural Questions and MS MARCO (very large), FinanceBench and CRAG
(non-commercial, per a catalogue I read), BRIGHT (reasoning-heavy, off target for a personal knowledge base), and two very
recent benchmarks, HERB and DeclineCraft, whose counts and licences I could not read from their abstracts.

No standard benchmark for metadata filters turned up; the filter variants here are derived from the dataset's own metadata.

## Adding one

The pipeline is not generic yet: most modules in `src/pnkbench/` import
[`multihop_rag.py`](../src/pnkbench/multihop_rag.py) directly (ten did when this was written), and its `Dataset`, `Article` and `Query` are the seam a
second dataset would implement. A dataset also needs the same four things MultiHop-RAG has: a pinned revision with
checksums that `fetch` enforces, loader invariants that fail loudly, a rule for turning a document into a file (path,
sidecar title and tags, modification time), and a README in its own folder.
