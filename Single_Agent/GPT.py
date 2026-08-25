from langchain_openai import ChatOpenAI
from Env import set_env
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

set_env("OPENAI_API_KEY")

INPUT_CSV = BASE_DIR / "all_three_extract.csv"
OUTPUT_CSV = BASE_DIR / "gpt_BIG.csv"

topics_df = pd.read_csv(INPUT_CSV)
if "context_considered_drivers" not in topics_df.columns:
    raise ValueError("CSV must contain 'context_considered_drivers' column.")
if "other_sections" not in topics_df.columns:
    topics_df["other_sections"] = ""

llm = ChatOpenAI(model="gpt-4o", temperature=0)

results = []

for _, row in topics_df.iterrows():
    prompt = f" Select the best option for the following:: {row['context_considered_drivers']}"
    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))

    result = {
        "topic": row["context_considered_drivers"],
        "human_decision": row["other_sections"],
        "debate_answer": content,
    }
    results.append(result)

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Saved plain GPT results to {OUTPUT_CSV}")
