# Single-Agent Baselines

This directory contains non-debate baselines for the architectural decision
task. Each script calls `gpt-4o` once per ADR record and writes a CSV of model
answers for later comparison.

## How It Works

All runners loop through `all_three_extract.csv` and call `gpt-4o` once per
ADR. `GPT.py` uses the base decision prompt. `cot.py` adds a reasoning
instruction to that prompt. `Few_shot.py` adds three examples selected using
seed `42`. Each output retains the source ADR fields beside the generated
decision for comparison with the human decision.

## Scripts

| File | Method | Output |
| --- | --- | --- |
| `GPT.py` | Plain `gpt-4o` base prompt. | `gpt_BIG.csv` |
| `cot.py` | Adds a chain-of-thought instruction to the base decision prompt. | `gpt_BIG_cot.csv` |
| `Few_shot.py` | Adds randomly selected ADR examples to each decision prompt. | `gpt_BIG_fewshot.csv` |

## Setup

Use Python 3.11 or 3.12 for the pinned study dependencies.

Install the dependencies listed in `Requirements.txt`, then provide an OpenAI
API key:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
$env:OPENAI_API_KEY="your-key"
```

On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r Requirements.txt
export OPENAI_API_KEY="your-key"
```

## Input

All scripts use `all_three_extract.csv`, included in this directory. They
expect `context_considered_drivers`; `other_sections` is optional and is copied
to the results as the human decision.

If only an Excel version of the input is found, the scripts convert it to CSV
next to that input file.

## Run

```powershell
python GPT.py
python cot.py
python Few_shot.py
```

On Linux and macOS:

```bash
python3 GPT.py
python3 cot.py
python3 Few_shot.py
```

The few-shot script uses three randomly chosen examples per target record with
a fixed random seed of `42`, so its prompt selection is reproducible for an
unchanged input CSV. Each example contains another record's
`context_considered_drivers` followed by the first line of its
`other_sections` field. The target record is excluded from its own example pool.
