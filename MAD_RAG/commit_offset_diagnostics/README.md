# Commit-offset diagnostics (MAD_RAG)

Exploratory retrieval diagnostics used to choose the **three-commit offset** for the
repository-aware (RAG) debate experiment. They support the statement in the paper
(Section 3.4) that retrieving repository context at, or immediately before, the commit
that introduces an ADR exposes post-decision information, and that an offset of three
commits was adopted as a conservative safeguard.

## What was run

For a given ADR, the pipeline identifies the earliest commit associated with the ADR file
(followed through the file's history, not by file name, since ADR files are sometimes
renamed), checks out the repository **`commit_offset` commits before that commit**,
builds the retrieval index from that snapshot, and runs the debate with retrieved
repository content injected into the agents' prompts. The ADR's *Decision Outcome*
section is never part of the prompt; agents only receive Context, Decision Drivers and
Considered Options.

Both retrieval strategies described in the paper were exercised. The table names use the
pipeline's internal labels:

| label in this folder | strategy in the paper | description |
|---|---|---|
| `undirected` | **Free RAG** | the model autonomously formulates queries derived from the ADR text and retrieves what it deems relevant |
| `directed` | **Guided RAG** | retrieval is constrained toward organizational, cultural and individual factors identified in the literature as influential in architectural decision-making |

Runs included here:

| table / folder | commit offset | retrieval | ADRs with a completed debate | run date |
|---|---|---|---|---|
| `offset0_directed`   | 0 | Guided RAG | 132 | 2026-04-21 |
| `offset0_undirected` | 0 | Free RAG   | 166 | 2026-04-28 |
| `offset1_directed`   | 1 | Guided RAG | 26  | 2026-05-05 |
| `offset1_undirected` | 1 | Free RAG   | 21  | 2026-05-05 |
| `offset3_directed`   | 3 | Guided RAG | 26  | 2026-05-05 |
| `offset3_undirected` | 3 | Free RAG   | 19  | 2026-05-05 |

The offset-0 runs cover a large share of the dataset. The offset-1 and offset-3 runs
were executed on a random sample of ADRs, the same sample for both offsets (two Free RAG
debates did not complete at offset 3), so that the effect of moving the retrieval point
back can be compared ADR by ADR. All runs used
GPT-4o (temperature 0) and `text-embedding-3-large`. ADRs whose repository could not be
cloned or whose debate did not complete are not included.

## Files

```
commit_offset_diagnostics/
├── README.md
├── commit_offset_diagnostics.db   # SQLite; one table per run (schema below), completed debates only,
│                                  # including the full debate trace of every ADR (message_history)
├── leak_check.py                  # reproduces the leakage counts, writes the per-run CSVs,
│                                  # and can export each trace as a plain-text file (--dump-traces)
└── summary/
    └── <run>.csv                  # one row per ADR: offset, files retrieved, debate answer, leakage flags
```

To read a trace without SQLite tooling, export a run to text files:

```
python leak_check.py commit_offset_diagnostics.db offset0_directed --dump-traces traces/offset0_directed
```

Each exported file has a short header (ADR, commit, offset, files retrieved, debate
answer) followed by the complete trace: prompts, retrieved repository content, agent
turns and judge verdict.

Table schema (identical for the six tables): `repo_name`, `github_address`, `adr_name`,
`commit_hash` and `first_commit_hash` (commits resolved by the pipeline for the ADR
file; the retrieval snapshot is the earliest ADR commit minus `commit_offset`),
`commit_offset`, `extraction_status`,
`context_considered_drivers` (the ADR sections given to the agents), `other_sections`
(the remaining ADR sections, including *Decision Outcome*; **withheld** from the agents),
`supported_side`, `debate_answer`, `reason`, `message_history` (full trace),
`timestamp`, `used_head_fallback`, `embedding_model`, `file_count`, `status`.

In a trace, retrieved repository content appears after the line
`Relevant information extracted from the repository:` inside each agent prompt.

## Leakage check

`leak_check.py` tests, for every completed debate, whether a verbatim 60-character
slice of the ADR's own *Decision Outcome* section occurs anywhere in the debate trace.
Since that section is withheld from the prompt, the only way it can appear in the trace
is through repository retrieval. Slices that also occur in the prompt (Context / Drivers /
Options) and MADR template placeholders are ignored, and ADRs whose outcome section was
never filled in (template text only) are excluded from the denominator.

```
python leak_check.py commit_offset_diagnostics.db offset0_directed --examples 5
```

| run | ADRs | assessable | outcome text found in trace | agent explicitly cites "the decision outcome from the repository" |
|---|---|---|---|---|
| offset 0, Guided RAG (`directed`)   | 132 | 106 | **67 (63%)** | 17 |
| offset 0, Free RAG (`undirected`)   | 166 | 141 | **122 (87%)** | 28 |
| offset 1, Guided RAG (`directed`)   | 26  | 18  | 2 (11%) | 0 |
| offset 1, Free RAG (`undirected`)   | 21  | 16  | 2 (12%) | 0 |
| offset 3, Guided RAG (`directed`)   | 26  | 18  | 1 (6%)  | 0 |
| offset 3, Free RAG (`undirected`)   | 19  | 15  | 1 (7%)  | 0 |

At offset 0 the snapshot contains the ADR file itself (Free RAG, which retrieves more broadly, is affected more often than Guided RAG), so retrieval usually returns the
decision being debated. One commit earlier removes the ADR file in most cases, but the
decision still surfaces for roughly one ADR in nine (e.g., when the ADR was committed
in several steps or related changes were already in the repository).
Three commits earlier the effect is rare. The counts are heuristic (verbatim matching) and
are meant to be read together with the traces, which were inspected manually.

## Examples

**Offset 0 (`offset0_undirected`, `55raidennetwork_raidenservices / 001-update-pfs-message.md`).**
The retrieved context handed to the agents is the ADR itself, including its outcome:

> Relevant information extracted from the repository:
> ## Considered Options
> 1. A solution where each participant broadcasts the signed, unblinded balance proofs […]
> 2. A solution where the message contains the capacities of both channel sides […]
> ## Decision Outcome
> Chosen option: "Option 2", because it allows for more privacy while offering only smaller safe capacities in the worst case. […]

Debate answer: *"Option 2: Broadcasting Aggregate Capacity Statements is the recommended approach […]"*.

**Offset 0 (`offset0_directed`, `284bbc_digitalpapereditfirebase / 2019-05-09-modular-architecture.md`).**
An agent argues directly from the leaked outcome:

> ### Decision Outcome
> The decision outcome from the repository indicates that Option 4 was chosen after further consideration. Initially, Option 3 was preferred for its uniform approach […]

Debate answer: *"Option 4 is the best proposed solution […]"*.

**Offset 0 (`offset0_directed`, `193hamletio_architecturaldecisionlog / 0002-layer-deployments.md`).**
The judge's verdict is justified by the leaked outcome:

> {"Whether there is a preference": "Yes", "Supported Side": "Negative", "Reason": "The negative side provides a clear alignment with the decision outcome extracted from the repository, which indicates that Option 1 was chosen as the best solution. […]"}

**Offset 1 (`offset1_directed`, `129digiraticouk_bfidiscovery / 0004-universal-viewer-customisations.md`).**
One commit before the ADR commit, the retrieved context still contains the full decision:

> ## Considered Options
> * Submodule the examples repository
> * Build components of UV in our solution […]
> ## Decision Outcome
> Chosen option: "Submodule the example repository", because it will keep the application largely agnostic of the viewers […]

Debate answer: *"Submodule the Examples Repository is the best option […]"*.

**Offset 3, residual case (`offset3_directed`, `127digiraticouk_bfidiscovery / 0003-choice-of-technology-and-framework.md`).**
Even three commits back, an earlier draft of the decision is present in the snapshot:

> Relevant information extracted from the repository:
> * Kotlin, Spring  * TypeScript, Express  * Python, Django
> ## Decision Outcome
> Chosen option: Kotlin/Spring, because our vast experience with the JVM and Spring will make this a straight-forward project. […]

The wording differs from the final ADR ("Kotlin is selected as the solution owing to
Digirati's vast experience with the JVM…"), i.e. the decision was drafted in the
repository before the commit identified as introducing the ADR. This is the kind of case
that motivates treating the three-commit margin as a conservative heuristic rather than
a guarantee (Section 6, internal validity).

## Provenance

The tables were exported from the original run databases (`main_dataset_offset0.db`,
`main_dataset_undirected.db`, `main_dataset_combined.db`), keeping only rows with
`status = 'debate_completed'`. In `main_dataset_combined.db` the tables named
`*_offset0` hold runs with `commit_offset = 1`; they are exported here under their
actual offset (`offset1_*`).
