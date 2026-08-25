# Multi-Agent Debate for Architectural Decisions

This repository contains experimental pipelines that evaluate whether language
models can reproduce architectural decisions recorded in Architecture Decision
Records (ADRs). The experiments compare debate-based approaches, structured
reasoning, single-agent baselines, and post-hoc analysis.

Most experiments are CSV-to-CSV pipelines: they read one ADR per row, generate
a decision, and save the result with a transcript. The RAG module indexes
repository documentation in a vector store and saves progress in SQLite.

## Repository Guide

| Directory | What to expect |
| --- | --- |
| `MAD_Regular+roles/` | Baseline four-agent debate: affirmative, negative, moderator, and judge. |
| `MAD_ATAM/` | Debate variant that generates ATAM-style scenarios and evaluates options before the debate. |
| `MAD_Guide_arch/` | Debate pipeline that turns debate output into a structured decision model and scores the options. |
| `MAD_More_agents/` | Three-debater variant: affirmative, negative, and challenger, supported by a moderator and judge. |
| `MAD_RAG/` | Repository-aware retrieval-augmented debate experiment. |
| `Single_Agent/` | Plain GPT, chain-of-thought, and few-shot non-debate baselines. |
| `Post_hoc/` | Grouping, comparison, and statistical analysis scripts for generated results. |

Each experiment has its own dependency and runtime requirements. There is no
repository-level `requirements.txt`.

## Replication Workflow

1. Change into the experiment directory.
2. Install dependencies using that module's README.
3. Set `OPENAI_API_KEY` for an OpenAI-based run.
4. Confirm the input file and columns described by that README.
5. Run the batch entry point and inspect the stated output.

For PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
```

For Linux and macOS shells:

```bash
export OPENAI_API_KEY="your-key"
```

Batch scripts use a local `Env.py` or `Utils/Env.py` helper. It preserves an
existing environment value or asks for the key at startup.

## How The Experiments Fit Together

1. Run a debate or single-agent experiment on an ADR CSV.
2. Save the generated answers to a CSV in that experiment directory.
3. Optionally use a module's `Comparison.py` as an aggregate review aid.
   Manual review remains the definitive evaluation.
4. Use `Post_hoc/` to group the results and run statistical analysis.

The modules do not all use the same input or output names. Read the README in
the directory you plan to run before starting it.

## Prerequisites

- Use Python 3.11 or 3.12. Python 3.14 is not supported by the pinned study
  dependencies.
- Create a virtual environment for the module you intend to run.
- Install that module's dependency file, for example:

```powershell
cd MAD_Regular+roles
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
```

On Linux and macOS:

```bash
cd MAD_Regular+roles
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r Requirements.txt
```

- Most batch scripts use OpenAI models and require `OPENAI_API_KEY`:

```powershell
$env:OPENAI_API_KEY="your-key"
```

```bash
export OPENAI_API_KEY="your-key"
```

Some single-run scripts instead assume a locally reachable Ollama instance or
an SSH-backed model. Those endpoint settings are hard-coded in the source and
must be reviewed before running the scripts.

## Dataset Expectations

Most batch pipelines expect a CSV with:

- `context_considered_drivers`: ADR context, decision drivers, and options.
- `other_sections`: the human decision used as a reference answer.

Input discovery varies by module. Some scripts look first in their own folder,
then search the repository, while others use a fixed local filename.

## Evaluation Categories

The generated decision is compared with the decision documented in the ADR.

### Match

**ADR decision:** choose separate URL creation for logging and error handling.

**Model decision:** separate URL creation into its own method for modularity,
reusability, and maintainability.

**Classification:** `Yes` - both decisions select the same architectural
strategy.

### Mismatch

**ADR decision:** build on top of Experimenter and invest in improvements,
rather than start a new application.

**Model decision:** use existing infrastructure while planning a gradual
transition to Nimbus.

**Classification:** `No` - the model proposes a different architectural
strategy from the recorded decision.

### Uncertainty

**ADR decision:** Option 1 is preferred, but further investigation is needed
before a conclusion.

**Model decision:** choose Option 2 as the pragmatic approach.

**Classification:** `Maybe` - the ADR does not record a final, directly
comparable decision.

## License

See [LICENSE](LICENSE).
