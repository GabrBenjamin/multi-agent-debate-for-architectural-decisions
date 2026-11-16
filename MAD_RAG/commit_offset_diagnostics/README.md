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

- `undirected` corresponds to **Free RAG**: the model autonomously formulates queries
  derived from the ADR text and retrieves what it deems relevant.
- `directed` corresponds to **Guided RAG**: retrieval is constrained toward
  organizational, cultural and individual factors identified in the literature as
  influential in architectural decision-making.

The database contains one table per run, named `offset<N>_<label>`, where `<N>` is the
commit offset (0, 1 or 3) and `<label>` is the retrieval strategy: `offset0_directed`,
`offset0_undirected`, `offset1_directed`, `offset1_undirected`, `offset3_directed` and
`offset3_undirected`.

The offset-0 runs cover a large share of the dataset. The offset-1 and offset-3 runs
were executed on the same random sample of ADRs, so that the effect of moving the
retrieval point back can be compared ADR by ADR. All runs used GPT-4o (temperature 0)
and `text-embedding-3-large`. ADRs whose repository could not be cloned or whose debate
did not complete are not included.

## Files

```
commit_offset_diagnostics/
├── README.md
├── commit_offset_diagnostics.db   # SQLite; one table per run (schema below), completed debates only,
│                                  # including the full debate trace of every ADR (message_history)
├── leak_check.py                  # flags traces in which the ADR's own outcome text was retrieved,
│                                  # and can export each trace as a plain-text file (--dump-traces)
└── summary/
    └── <run>.csv                  # one row per ADR: offset, files retrieved, debate answer, flags
```

To read a trace without SQLite tooling, export a run to text files:

```
python leak_check.py commit_offset_diagnostics.db offset0_directed --dump-traces traces/offset0_directed
```

Each exported file has a short header (ADR, commit, offset, files retrieved, debate
answer) followed by the complete trace: prompts, retrieved repository content, agent
turns and judge verdict.

Table schema (identical for all tables): `repo_name`, `github_address`, `adr_name`,
`commit_hash` and `first_commit_hash` (commits resolved by the pipeline for the ADR
file; the retrieval snapshot is the earliest ADR commit minus `commit_offset`),
`commit_offset`, `extraction_status`,
`context_considered_drivers` (the ADR sections given to the agents), `other_sections`
(the remaining ADR sections, including *Decision Outcome*; **withheld** from the agents),
`supported_side`, `debate_answer`, `reason`, `message_history` (full trace),
`timestamp`, `used_head_fallback`, `embedding_model`, `file_count`, `status`.

In a trace, retrieved repository content appears after the line
`Relevant information extracted from the repository:` inside each agent prompt.

## How leakage shows up

Leakage can take two forms. *Explicit* leakage occurs when the ADR's own text, or a
draft of it, is in the snapshot and is retrieved. *Implicit* leakage occurs when the
decision is not stated as such but the repository already reflects it, for example a
related document, README, guide or changelog that presupposes the chosen option because
the decision was taken or implemented before the ADR was recorded.

`leak_check.py` helps locate the explicit form: it flags the debates in which a verbatim
slice of the ADR's withheld *Decision Outcome* section appears in the trace, which can
only happen through repository retrieval. It
