# fewshot_baseline.py
from langchain_openai import ChatOpenAI
from Env import set_env
import pandas as pd
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(BASE_DIR)


def resolve_input_path(filename: str) -> Path:
    candidates = [BASE_DIR / filename, REPO_ROOT / filename]
    candidates.extend(path for path in sorted(REPO_ROOT.glob(f"**/{filename}")) if path not in candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return REPO_ROOT / filename


def ensure_csv_input(stem: str) -> Path:
    csv_path = resolve_input_path(f"{stem}.csv")
    if csv_path.exists():
        return csv_path

    spreadsheet_candidates = [
        BASE_DIR / f"{stem}.xlsx",
        BASE_DIR / f"{stem}.xls",
        REPO_ROOT / f"{stem}.xlsx",
        REPO_ROOT / f"{stem}.xls",
    ]
    spreadsheet_candidates.extend(
        path
        for suffix in ("xlsx", "xls")
        for path in sorted(REPO_ROOT.glob(f"**/{stem}.{suffix}"))
        if path not in spreadsheet_candidates
    )

    spreadsheet_path = next((path for path in spreadsheet_candidates if path.exists()), None)
    if spreadsheet_path is None:
        return csv_path

    output_csv_path = spreadsheet_path.with_suffix(".csv")
    df_excel = pd.read_excel(spreadsheet_path)
    df_excel.to_csv(output_csv_path, index=False)
    return output_csv_path

set_env("OPENAI_API_KEY")

INPUT_CSV = ensure_csv_input("all_three_extract")
OUTPUT_CSV = BASE_DIR / "gpt_BIG_fewshot.csv"
K_EX       = 3          # number of random exemplars
RAND_SEED  = 42

random.seed(RAND_SEED)
df = pd.read_csv(INPUT_CSV)  # needs columns: context_considered_drivers, other_sections
if "context_considered_drivers" not in df.columns:
    raise ValueError("CSV must contain 'context_considered_drivers' column.")
if "other_sections" not in df.columns:
    df["other_sections"] = ""

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

def build_fewshot_prompt(df, idx, k, target_blob):
    n = len(df)
    pool = [i for i in range(n) if i != idx]
    ex_ids = random.sample(pool, min(k, len(pool))) if pool else []

    parts = []
    # Each exemplar: original prompt line, then its completion on the next line
    for j in ex_ids:
        ex_in = str(df.at[j, "context_considered_drivers"])
        ex_out = str(df.at[j, "other_sections"]) if not pd.isna(df.at[j, "other_sections"]) else ""
        ex_out = ex_out.strip().splitlines()[0] if ex_out else "Selected option"

        parts.append(f" Select the best option for the following:: {ex_in}\n{ex_out}\n\n")

    # Target: EXACT original prompt (no extra wording)
    parts.append(f" Select the best option for the following:: {target_blob}")
    return "".join(parts)

results = []
for idx, row in df.iterrows():
    target_blob = str(row["context_considered_drivers"])
    prompt = build_fewshot_prompt(df, idx, K_EX, target_blob)

    resposta = llm.invoke(prompt)
    content = getattr(resposta, "content", str(resposta))

    results.append({
        "topic": target_blob,
        "human_decision": row.get("other_sections", ""),
        "fewshot_answer": content,
    })

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Saved few-shot results to {OUTPUT_CSV}")
