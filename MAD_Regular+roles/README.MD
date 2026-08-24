# Regular Multi-Agent Debate

This is the baseline debate pipeline. Four agents discuss an architectural
decision: an affirmative debater, a negative debater, a moderator, and a judge.
The moderator may end the debate early; otherwise the judge produces the final
decision after the configured rounds.

## How It Works

For each ADR row, `Run_all.py` turns `context_considered_drivers` into a debate
topic and creates four `gpt-4o` agents. The affirmative and negative agents
argue competing positions. The moderator evaluates the discussion after every
round and the judge provides the final answer when required. `Utils/State_graph.py`
coordinates this exchange through LangGraph; its nodes preserve the selected
side, final answer, reason, and full message history.

## Main Files

| File | Purpose |
| --- | --- |
| `Main.py` | Runs one hard-coded debate using the `llama` model path. |
| `Run_all.py` | Runs the four-agent debate for each row in an ADR CSV using `gpt-4o`. |
| `Utils/` | Shared agents, prompts, model wrappers, graph nodes, and LangGraph workflow. |
| `Requirements.txt` | Python dependencies for this module. |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
$env:OPENAI_API_KEY="your-key"
```

`Run_all.py` reads `OPENAI_API_KEY` at startup and uses `gpt-4o` for all four
agents.

`Utils/Env.py` uses an existing key from the environment or requests one in the
terminal before the agents are created.

## Run The Batch Pipeline

```powershell
python Run_all.py
```

The script looks for `sample_with_all_three_extract.csv` in this directory and
then in `MAD_ATAM/`. The input must contain
`context_considered_drivers`; `other_sections` is saved as the reference human
decision.

The generated file is `debateC_1.csv`, containing the source topic, reference
decision, selected side, final answer, reason, and debate history.

| Output column | Contents |
| --- | --- |
| `topic` | Original ADR context, drivers, and options. |
| `human_decision` | Decision recorded in the ADR. |
| `supported_side` | Side selected by the workflow. |
| `debate_answer` | Final model decision. |
| `reason` | Explanation returned by the workflow. |
| `message_history` | Complete debate transcript. |

## Single Example

```powershell
python Main.py
```

`Main.py` uses the `llama` model path and therefore depends on the remote model
configuration in `Utils/LLM.py`. Review that configuration before running it;
it is not a drop-in local OpenAI workflow.
