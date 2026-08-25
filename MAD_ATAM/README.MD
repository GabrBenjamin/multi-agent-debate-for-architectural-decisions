# MAD ATAM

This experiment extends a multi-agent architectural debate with an
ATAM-inspired preparation stage. For each ADR row, the pipeline asks a model to
derive scenarios, identify quality attributes and constraints, and list option
pros and cons before running the debate.

## How It Works

`Run_all.py` reads each ADR and sends `context_considered_drivers` to the local
`ChatOllama` client. Its three local-model calls prepare the debate topic.
The script then creates affirmative and negative debaters, a moderator, and a
judge with `gpt-4o`, invokes `Utils/State_graph.py`, and saves the final state.
The local client's `server_url` is configured in `Run_all.py`.

## Main Files

| File | Purpose |
| --- | --- |
| `Main.py` | Runs one hard-coded ADR example through the ATAM preparation and debate flow. |
| `Run_all.py` | Runs the same flow for every row in the input CSV. |
| `Utils/` | Agent classes, prompts, model access, debate nodes, and LangGraph wiring. |
| `sample_with_all_three_extract.csv` | Batch input included with this experiment. |

## Setup

Use Python 3.11 or 3.12 for the pinned study dependencies.

Install the local dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
```

On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r Requirements.txt
```

`Run_all.py` currently uses `gpt-4o` for the debate agents, but its call that
loads `OPENAI_API_KEY` is commented out. Set the key before running it:

```powershell
$env:OPENAI_API_KEY="your-key"
```

```bash
export OPENAI_API_KEY="your-key"
```

The ATAM preparation uses a locally configured model client. Configure the
model endpoint in the source to match your local setup before running it.

## Run

Run one hard-coded example:

```powershell
python Main.py
```

On Linux and macOS, run `python3 Main.py`.

Run the batch experiment:

```powershell
python Run_all.py
```

On Linux and macOS, run `python3 Run_all.py`.

The batch script reads `sample_with_all_three_extract.csv`. It requires
`context_considered_drivers` and uses `other_sections` as the human reference
decision.

## Outputs

`Run_all.py` writes `ATAM_debateGLGL.csv` with the source topic, human decision,
selected side, final answer, reason, and message history. That file is generated
when the script runs and may not be present in a fresh clone.
