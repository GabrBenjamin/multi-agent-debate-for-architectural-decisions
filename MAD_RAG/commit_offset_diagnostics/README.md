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
only happen through repository retrieval. It is a pointer to traces worth reading, not a
measure of how much leakage there is, since it cannot detect the implicit form. The
diagnostics were assessed by manually inspecting the retrieved content and the debate
traces (paper, Sections 3.4 and 6). The examples below illustrate what that inspection
found at each offset.

```
python leak_check.py commit_offset_diagnostics.db offset0_directed --examples 5
```

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

**Offset 1, explicit leakage (`offset1_directed`, `129digiraticouk_bfidiscovery / 0004-universal-viewer-customisations.md`).**
One commit before the ADR commit, the retrieved context still contains the full decision:

> ## Considered Options
> * Submodule the examples repository
> * Build components of UV in our solution […]
> ## Decision Outcome
> Chosen option: "Submodule the example repository", because it will keep the application largely agnostic of the viewers […]

Debate answer: *"Submodule the Examples Repository is the best option […]"*.

**Offset 1, implicit leakage (`offset1_undirected`, `89Sylius_Sylius / 2020_03_03_feature_coverage_in_api.md`).**
The ADR's outcome is "Using Behat for the feature coverage". The ADR file is not in the
snapshot and its text is not retrieved, so `leak_check.py` does not flag this case.
However, a related decision record committed earlier is retrieved, and it already
presupposes the outcome:

> # API - Coverage
> […] New Sylius API (based on API Platform), should cover in 100% business value provided in Admin panel (and already covered by `@ui` Behat scenarios). […]
> ### Decision and reasoning
> For the beginning, we focus on implementing only features already covered with Behat UI scenarios. In the future, it would result in bigger reliability of Behat API tests […]

Debate answer: *"The decision to initially implement features covered by Behat UI scenarios is the most suitable option […]"*.

Here the decision is revealed by neighbouring repository content rather than by the ADR
itself.

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
