# Multi-Agent Debate for Architectural Decisions

This repository contains experimental pipelines that evaluate whether language
models can reproduce architectural decisions recorded in Architecture Decision
Records (ADRs). The experiments compare debate-based approaches, structured
reasoning, single-agent baselines, and post-hoc analysis.

## Repository Guide

| Directory | What to expect |
| --- | --- |
| `MAD_Regular+roles/` | Baseline four-agent debate: affirmative, negative, moderator, and judge. |
| `MAD_ATAM/` | Debate variant that generates ATAM-style scenarios and evaluates options before the debate. |
| `MAD_Guide_arch/` | Debate pipeline that turns debate output into a structured decision model and scores the options. |
| `MAD_More_agents/MAD-main/MAD_Framework/` | Five-agent variant with an additional neutral participant. |
| `MAD_RAG/` | Retrieval-augmented debate experiment. Its documentation is intentionally unchanged while that work continues. |
| `Single_Agent/` | Chain-of-thought and few-shot, non-debate baselines. |
| `Post_hoc/` | Grouping, comparison, and statistical analysis scripts for generated results. |

Each experiment is independently runnable and has its own dependency file or
environment requirements. There is no repository-level `requirements.txt`.

## How The Experiments Fit Together

1. Run a debate or single-agent experiment on an ADR CSV.
2. Save the generated answers to a CSV in that experiment directory.
3. Optionally use a module's `Comparison.py` as an aggregate review aid.
   Manual review remains the definitive evaluation.
4. Use `Post_hoc/` to group the results and run statistical analysis.

The modules do not all use the same input or output names. Read the README in
the directory you plan to run before starting it.

## Prerequisites

- Python 3.11 or later is recommended.
- Create a virtual environment for the module you intend to run.
- Install that module's dependency file, for example:

```powershell
cd MAD_Regular+roles
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
```

- Most batch scripts use OpenAI models and require `OPENAI_API_KEY`:

```powershell
$env:OPENAI_API_KEY="your-key"
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

## Experimental Notes

- These scripts are research artifacts, not a production application.
- Model names, remote endpoints, prompts, and file names are often configured
  directly in Python files.
- Generated CSVs, local SQLite databases, caches, and analysis outputs may be
  large or machine-specific. Review them before committing them to Git.

## License

See [LICENSE](LICENSE).
