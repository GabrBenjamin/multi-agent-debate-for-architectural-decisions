# cot_baseline.py
from langchain_openai import ChatOpenAI
from Utils.Env import set_env
import pandas as pd

set_env("OPENAI_API_KEY")

INPUT_CSV  = r'C:/Users/gabri/Downloads/all_three_extract.csv'
OUTPUT_CSV = "gpt_BIG_cot.csv"

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
