# MAD RAG: Repository-Aware Multi-Agent Debate

This module evaluates architectural decisions with a multi-agent debate that
retrieves documentation from the repository associated with each Architecture
Decision Record (ADR). The repository is reconstructed at a historical commit,
indexed in a vector store, and queried during the debate so the agents can use
project-specific context.

The primary entry point is `extractor_ensambler.py`. It records the extraction
and debate state for each ADR in `main_dataset.db`, allowing runs to continue
from the work already stored in the database.

## Retrieval Configurations

The packaged RAG workflow indexes documentation-oriented repository files,
including ADRs, READMEs, `CONTRIBUTING.md`, guides, changelogs, and related
text files. `extractor.py` controls this selection in `_is_relevant_file()` and
adds the resulting text chunks to the Chroma store.

The debate supports two retrieval schedules. Set `retriever_mode` in
`main_debate()` in `debate_manager.py` to one of the following values:

- `continuous` is the default. Both opening statements retrieve repository
  context, and each rebuttal retrieves fresh context.
- `opening_only` retrieves context only for the affirmative and negative
  opening statements.

To test alternative RAG prompt roles, edit the debate and retrieval-header
templates in `Utils/Config.py`. `Utils/Nodes.py` controls how the current
prompt is used as a similarity-search query and when retrieved chunks are
added to a debater prompt.

For a guided-retrieval condition, create textual documents containing these
signals, then add them to the vector store in `extractor.py`:

- organizational and technical standards or constraints, such as material from
  `CONTRIBUTING.md`;
- cultural practices, including documentation activity and work distribution;
- individual contributor experience inferred from available profile metadata;
- project age and an approximate project type inferred from its structure or
  documentation.

The packaged extractor does not generate these metadata snippets automatically.

## What The Pipeline Does

The RAG workflow has two phases:

1. Repository extraction: identify the relevant historical commit, clone the
   repository, select documentation-oriented files, and create a Chroma vector
   store using OpenAI embeddings.
2. RAG debate: retrieve relevant chunks from that vector store while agents
   debate the ADR decision, then store the final answer and transcript.

The included source dataset contains 291 ADR records. In the current included
database, 239 records completed extraction and debate; the remaining records
could not be processed because of unavailable repositories, unresolved commits,
insufficient history, ADR mapping issues, or extraction errors.

## Debate And Retrieval

The debate uses four `gpt-4o` roles:

- `AffirmativeSide`: argues for a proposed decision.
- `NegativeSide`: challenges that position and offers an alternative.
- `Moderator`: decides whether a preference is clear after a round.
- `Judge`: produces the final answer when the debate reaches its round limit.

For each repository, the extractor creates a Chroma vector store using
`text-embedding-3-large`. During the debate, the affirmative and negative
agents query that store with their current prompt. The retrieved chunks are
appended to the prompt as repository context.

The default `continuous` retrieval mode retrieves context for both opening
statements and before every rebuttal. The moderator and judge evaluate the
debate output without an additional retrieval step.

The progress database stores the selected side, final decision, reason, and
full transcript, allowing retrieval artifacts and debate results to be reviewed
together for each ADR.

## Main Files

| File | Purpose |
| --- | --- |
| `extractor_ensambler.py` | Primary RAG workflow coordinator. Seeds the progress database, builds vector stores, and runs RAG debates. |
| `extractor.py` | Clones repositories at historical commits, selects files, chunks content, and creates Chroma stores. |
| `debate_manager.py` | Creates the debate agents and invokes the RAG-enabled LangGraph workflow. |
| `Utils/Nodes.py` | Performs Chroma similarity search and adds retrieved content to debater prompts. |
| `database.py` | SQLite database helper used for progress and extraction metadata. |
| `build_adr_tracking_db.py` | Creates the compact RAG tracking database from the original research database. |
| `test_repos.py` | Optional diagnostic for checking access to pending repository URLs. |
| `Comparison.py` | Aggregate comparison script for a legacy CSV result file. |

## Inputs

The primary input is `data_hash_only.csv`. Each row supplies:

- the GitHub repository URL
- ADR path and commit hash
- ADR context, decision drivers, and considered options
- the recorded human decision

`adr_data_rag_minimal.db` is also required and is included with the RAG module.
It contains the `adr_tracking_info` table used to resolve the first commit
associated with each ADR. Startup reads this database even when
`main_dataset.db` already contains previous extraction or debate results.

`build_adr_tracking_db.py` documents how the compact database is derived. It
reads a local `adr_data.db` source database and retains only the ADR tracking
fields and timestamp used by this workflow.

The extractor works three commits before that resolved commit. It selects
documentation-focused files, including ADRs, Markdown documentation, guides,
changelogs, and related text files. It does not currently index arbitrary
application source files.

## Setup

Install the dependencies in this directory:

```powershell
pip install -r requirements.txt
```

The workflow requires an OpenAI API key for `text-embedding-3-large` and
`gpt-4o`, as well as network access and Git access to the repositories in the
input data.

```powershell
$env:OPENAI_API_KEY="your-key"
```

## Run The RAG Workflow

Use `extractor_ensambler.py` for the actual RAG pipeline. Do not use
`Run_all.py` as the primary RAG command: it is a separate legacy batch path and
does not initialize the retriever state required by the RAG graph.

### 1. Optional Repository Check

```powershell
python test_repos.py
```

This checks a sample of pending repositories in `main_dataset.db` with
`git ls-remote`.

### 2. Build Vector Stores

`extractor_ensambler.py` first imports `data_hash_only.csv` into the `progress`
table and resolves the historical commits. To build vector stores from a fresh
database, run:

```powershell
python extractor_ensambler.py --mode extract
```

This clones the eligible repositories, creates the vector stores, and marks
successful records with `extraction_status = "success"`.

### 3. Run RAG Debates

After vector-store extraction succeeds, run:

```powershell
python extractor_ensambler.py --mode run_mad
```

This selects only successful records that do not yet have a saved message
history, then writes the RAG debate output back to the progress database. To
perform both phases in order, run:

```powershell
python extractor_ensambler.py --mode all
```

Use `--workers <count>` to choose a worker count. The defaults are 10 workers
for extraction and 1 worker for debates.

## Outputs

The primary outputs are database and local retrieval artifacts rather than a
single CSV file:

| Output | Contents |
| --- | --- |
| `main_dataset.db` | The `progress` table: source ADR metadata, extraction status, resolved commit, final answer, reason, message history, embedding model, and file count. |
| `dataset/<repository-and-commit>/info.db` | Per-repository file traversal metadata and vector-store test records. |
| `dataset/<repository-and-commit>/<repository>_chroma_db/` | Persistent Chroma vector store used for retrieval during debate. |

`Run_all.py` can write `MAD_rag_results.csv`, and `Comparison.py` can write
`MAD_rag_results_with_comparisons.csv`.
