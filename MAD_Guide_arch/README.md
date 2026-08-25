# MAD GuideArch Debate Pipeline

This module combines a multi-agent debate with GuideArch-style option scoring.
The debaters discuss an ADR in free text; the moderator or judge then returns a
strict JSON decision model for `Scorer.py`.

## Pipeline

`ADR CSV row -> free-text debate -> structured decision JSON -> fuzzy scorer -> winning option`

The debate uses four roles: an affirmative debater, a negative debater, a
moderator, and a judge. The batch pipeline is the primary entry point.

For each ADR, `Run_all.py` asks the debaters to discuss drivers, option impacts,
constraints, and risk. The moderator and judge prompts require strict JSON with
the GuideArch decision model. The graph stores its `final` object in
`debate_answer`; the runner normalizes it with `prepare_struct_for_scoring()`
and calls `Scorer.compute_best_option()`. The scorer converts qualitative or
triangular impacts into comparable numeric values, applies priorities and
constraints, and returns the winning option.

## Main Files

| File | Purpose |
| --- | --- |
| `Run_all.py` | Runs the dataset, extracts structured debate output, scores options, and writes results. |
| `Main.py` | Runs one hard-coded debate topic for development or inspection. |
| `Scorer.py` | Implements the GuideArch-style fuzzy option ranking. |
| `Utils/` | Prompts, agents, state graph, nodes, and environment helpers. |
| `adrs_final_sample_58.csv` | Batch input dataset included with the module. |

## Setup

Use Python 3.11 or 3.12 for the pinned study dependencies.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
pip install langchain-together
$env:OPENAI_API_KEY="your-key"
```

On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r Requirements.txt
python3 -m pip install langchain-together
export OPENAI_API_KEY="your-key"
```

`langchain-together` is imported by `Utils/Agents.py` but is not listed in
`Requirements.txt`, so it must currently be installed separately.

## Run The Batch Pipeline

```powershell
python Run_all.py
```

On Linux and macOS, run `python3 Run_all.py`.

The script reads `adrs_final_sample_58.csv`. It expects
`context_considered_drivers` and `other_sections`, then writes
`debate_guideArchS_cleaned_scored.csv`.

The result includes the source topic, human decision, debate metadata,
structured decision JSON, full scoring JSON, and the scorer-selected winner.

`debate_struct_json` stores the structured decision model returned by the graph.
`scoring_json` stores the calculated option scores. `debate_answer` is the
winner returned by `Scorer.py`; `message_history` keeps the full discussion.

## Decision Model Used By The Scorer

The scorer uses drivers, options, impacts, constraints, and optional risk flags.
Drivers define an orientation and priority; impacts may use qualitative labels
or triangular fuzzy values. The scorer validates constraints and ranks valid
options.

## Single Debate

```powershell
python Main.py
```
