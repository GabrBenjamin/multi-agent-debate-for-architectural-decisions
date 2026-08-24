# Multi-Agent Debate Framework

This experiment extends the regular debate with a fifth participant: a neutral
third debater. It is located below `MAD_More_agents/MAD-main/` and has its own
copy of the debate framework.

## Pipeline

For each ADR record, three debaters discuss candidate solutions, a moderator
evaluates the discussion, and a judge produces the final answer when required.
The batch output preserves the full debate transcript.

## Main Files

| File | Purpose |
| --- | --- |
| `Run_all.py` | Batch entry point using `gpt-4o` for the three debaters, moderator, and judge. |
| `Main.py` | Runs one hard-coded five-agent debate using `gpt-4o`. |
| `Utils/` | Agent abstractions, prompts, nodes, environment helper, and LangGraph workflow. |
| `debate_results_with_history.csv` | Example or generated batch result file. |

## Setup

This folder does not contain its own dependency file. Install the compatible
LangChain, LangGraph, pandas, and OpenAI dependencies used by the sibling
debate modules, then provide an API key before running either entry point:

```powershell
$env:OPENAI_API_KEY="your-key"
```

## Run The Batch Pipeline

From this directory:

```powershell
python Run_all.py
```

The script searches for `sample_with_all_three_extract.csv` in this directory,
then in `MAD_ATAM/`. It requires `context_considered_drivers` and writes
`debate_results_with_history.csv` with the source context, prompt, selected
side, final answer, reason, third-debater answer, and full message history.

`Comparison.py` is optional aggregate-review assistance and does not replace
manual review of the experiment results.

## Single Example

```powershell
python Main.py
```

`Main.py` and `Run_all.py` both use `gpt-4o` through `ChatOpenAI`.

## Notes

- The batch pipeline uses a maximum of two debate rounds.
- Prompts, model names, input locations, and output filenames are configured
  in source files rather than command-line options.
- Running the batch script makes API calls and may incur costs.
