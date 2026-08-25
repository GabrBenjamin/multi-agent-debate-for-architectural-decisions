# Post-Hoc Grouping and Analysis

This directory transforms experiment results into a grouped ADR dataset and
then tests relationships between decision characteristics and the outcome
classification (`Yes`, `No`, or `Maybe`).

## How It Works

`Grouping.py` combines raw debate results and comparison outcomes, extracts or
normalizes ADR characteristics, and writes progressively enriched CSV files.
`Group_analysis.py` tests associations between those characteristics and the
`Yes`/`No`/`Maybe` result. `Drivers_options_analysis.py` counts decision drivers
and options from ADR Markdown and performs additional analyses.

## Workflow

Run these scripts in order:

1. `Grouping.py` creates structured ADR variables and produces
   `Final_Grouping2.csv` through several intermediate CSV files.
2. `Group_analysis.py` reads `Final_Grouping2.csv` and runs categorical,
   bivariate, and multinomial analyses.
3. `Drivers_options_analysis.py` reads `Final_Grouping2.csv`, counts drivers
   and options from ADR Markdown, and runs additional models.

## Required Inputs

`Grouping.py` uses two files in this directory:

| File | Role |
| --- | --- |
| `debate_BIG_GPT.csv` | Raw debate results containing the ADR text. |
| `output_with_comparisonsBIG_GPT.csv` | Answer comparison results used to add `comparison_result`. |

The comparison results are aligned by row order, so do not reorder either input
independently before running the pipeline.

## Setup

Use Python 3.11 or 3.12. The analysis dependencies, including pandas, NumPy,
SciPy, scikit-learn, statsmodels, Patsy, and LangChain, are listed in
`Requirements.txt`.

On Windows PowerShell, create an environment, install the dependencies, and set
an API key before running `Grouping.py`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r Requirements.txt
$env:OPENAI_API_KEY="your-key"
python Grouping.py
```

On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r Requirements.txt
export OPENAI_API_KEY="your-key"
python3 Grouping.py
```

Then run the analyses:

```powershell
python Group_analysis.py
python Drivers_options_analysis.py
```

On Linux and macOS:

```bash
python3 Group_analysis.py
python3 Drivers_options_analysis.py
```

## Generated Files

`Grouping.py` writes the following progression:

`Grouped_decisions.csv` -> `Grouped_decisions_with_comparison.csv` ->
`Grouped_decisions_with_comparison_filled.csv` ->
`Grouped_decisions_with_concerns_split.csv` -> `Final_Grouping.csv` ->
`Final_Grouping2.csv`

`Drivers_options_analysis.py` additionally creates
`Final_Grouping2_with_counts.csv`.

## What Is Analysed

The grouping stage extracts or normalizes fields such as layer, family,
decision type, scope, lifecycle stage, ambiguity, risks, and concerns. The
analysis scripts test their association with the `Yes`/`No`/`Maybe` comparison
outcome and examine the counts of decision drivers and considered options.
