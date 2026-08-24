# Regular Multi-Agent Debate

This is the baseline debate pipeline. Four agents discuss an architectural
decision: an affirmative debater, a negative debater, a moderator, and a judge.
The moderator may end the debate early; otherwise the judge produces the final
decision after the configured rounds.

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

## Single Example

```powershell
python Main.py
```

`Main.py` uses the `llama` model path and therefore depends on the remote model
configuration in `Utils/LLM.py`. Review that configuration before running it;
it is not a drop-in local OpenAI workflow.

## Notes

- Debate prompts and the round limit are defined in Python rather than through
  command-line options.
- Generated results can be costly and model-dependent.
- `Comparison.py` is optional aggregate-review assistance; manually reviewed
  results remain the definitive evaluation.
