# fewshot_baseline.py
from langchain_openai import ChatOpenAI
from Utils.Env import set_env
import pandas as pd
import random

set_env("OPENAI_API_KEY")

INPUT_CSV  = r'C:/Users/gabri/Downloads/all_three_extract.csv'
OUTPUT_CSV = "gpt_BIG_fewshot.csv"
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
