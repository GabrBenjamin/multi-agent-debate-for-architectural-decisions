# More-Debater Debate Framework

This experiment extends the regular debate with a third debater that challenges
assumptions and introduces a distinct perspective.

## Pipeline

For each ADR record, three debaters discuss candidate solutions, a moderator
evaluates the discussion, and a judge produces the final answer when required.
The batch output preserves the full debate transcript.

## How It Works

For every ADR, `Run_all.py` creates `AffirmativeSide`, `NegativeSide`, and
`ThirdPart` with `gpt-4o`. The first two argue opposing positions and the third
challenges their assumptions or contributes a distinct perspective. `Moderator`
evaluates the exchange; `Judge` supplies the final answer.
`Utils/State_graph.py` coordinates the messages with LangGraph.

## Main Files

| File | Purpose |
| --- | --- |
| `Run_all.py` | Batch entry point using `gpt-4o` for the three debaters, moderator, and judge. |
| `Main.py` | Runs one hard-coded three-debater debate using `gpt-4o`. |
| `Comparison.py` | Adds an aggregate comparison result to the batch output. |
| `Utils/` | Agent abstractions, prompts, nodes, environment helper, and LangGraph workflow. |
| `debate_results_with_history.csv` | Example or generated batch result file. |
| `output_with_comparisons3a.csv` | Published comparison result file. |

## Setup

Use Python 3.11 or 3.12. Install the dependencies in `Requirements.txt`, then
provide an API key before running either entry point.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r Requirements.txt
$env:OPENAI_API_KEY="your-key"
```

On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r Requirements.txt
export OPENAI_API_KEY="your-key"
```

## Run The Batch Pipeline

From this directory:

```powershell
python Run_all.py
```

On Linux and macOS, run `python3 Run_all.py`.

The script searches for `sample_with_all_three_extract.csv` in this directory,
then in the repository's shared dataset location, `MAD_ATAM/`. This is the
same shared input used by the regular and ATAM batch experiments. It requires
`context_considered_drivers` and writes `debate_results_with_history.csv`.

| Output column | Contents |
| --- | --- |
| `context_considered_drivers` | Original ADR context, decision drivers, and options. |
| `full_prompt` | Complete prompt supplied to the debate. |
| `supported_side` | Side selected by the workflow. |
| `debate_answer` | Final model decision. |
| `reason` | Explanation returned by the workflow. |
| `third_answ` | Third debater's challenger perspective. |
| `message_history` | Complete debate transcript. |

`Comparison.py` reads `debate_results_with_history.csv` and writes
`output_with_comparisons3a.csv`, adding the `comparison_result` column.

## Single Example

```powershell
python Main.py
```

On Linux and macOS, run `python3 Main.py`.

`Main.py` and `Run_all.py` both use `gpt-4o` through `ChatOpenAI`.
