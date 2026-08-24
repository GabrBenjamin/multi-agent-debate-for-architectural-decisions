# Single-Agent Baselines

This directory contains non-debate baselines for the architectural decision
task. Both scripts call `gpt-4o` once per ADR record and write a CSV of model
answers for later comparison.

## Scripts

| File | Method | Output |
| --- | --- | --- |
| `cot.py` | Adds a chain-of-thought instruction to the base decision prompt. | `gpt_BIG_cot.csv` |
| `Few_shot.py` | Adds randomly selected ADR examples to each decision prompt. | `gpt_BIG_fewshot.csv` |

## Setup

Install the dependencies listed in `Requirements.txt`, then provide an OpenAI
API key:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
$env:OPENAI_API_KEY="your-key"
```

## Input

Both scripts use `all_three_extract.csv`, included in this directory. They
expect `context_considered_drivers`; `other_sections` is optional and is copied
to the results as the human decision.

If only an Excel version of the input is found, the scripts convert it to CSV
next to that input file.

## Run

```powershell
python cot.py
python Few_shot.py
```

The few-shot script uses three randomly chosen examples per target record with
a fixed random seed of `42`, so its prompt selection is reproducible for an
unchanged input CSV.

## Notes

- These are batch scripts, not interactive tools.
- Each row makes an API request and may incur provider charges.
- The generated outputs are intended to be compared with the ADR reference
  decisions in a later evaluation step.
