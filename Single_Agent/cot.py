# cot_baseline.py
from langchain_openai import ChatOpenAI
from Env import set_env
import pandas as pd
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
OUTPUT_CSV = BASE_DIR / "gpt_BIG_cot.csv"

topics_df = pd.read_csv(INPUT_CSV)  # needs columns: context_considered_drivers, other_sections
if "context_considered_drivers" not in topics_df.columns:
    raise ValueError("CSV must contain 'context_considered_drivers' column.")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

results = []
for idx, row in topics_df.iterrows():
    base_prompt = f" Select the best option for the following:: {row['context_considered_drivers']}"
    # === CoT addition (only technique added) ===
    prompt = base_prompt + "\nThink step by step before giving the final answer."

    resposta = llm.invoke(prompt)
    content = getattr(resposta, "content", str(resposta))

    results.append({
        "topic": row["context_considered_drivers"],
        "human_decision": row.get("other_sections", ""),
        "cot_answer": content,
    })

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Saved CoT results to {OUTPUT_CSV}")
