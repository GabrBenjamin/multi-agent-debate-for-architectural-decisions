import os
import pandas as pd
from Utils.Env import set_env

# ------- LangChain imports -------
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI




set_env("OPENAI_API_KEY")

# ========= 2. LOAD YOUR DATA =========
df = pd.read_csv('debate_guideArchS_cleaned_scored.csv')
# Make sure it has columns 'human_decision' and 'winner'.

# ========= 3. CREATE A LANGCHAIN PROMPT =========
# We'll inject {row_id}, {human_decision}, and {guidearch_winner} into this template.
template = """You are a helpful assistant. 
We have two decisions:
1) human_decision: {human_decision}
2) guidearch_winner: {guidearch_winner}

We want to see if both come to the same conclusion or choose the same option. 
Return a single line with "Yes" if they match or "No" if they do not match and maybe if one of them left it open but the discussion aligns, 
followed by "//" then a short explanation in a single line. 
We also have an ID for this row: {row_id}.

Return it in the exact format:
"<row_id> - <Yes/No/Maybe> // <short explanation>"

No extra lines or text.
"""

prompt = PromptTemplate(
    input_variables=["row_id", "human_decision", "guidearch_winner"],
    template=template
)

# ========= 4. SET UP THE LLM AND CHAIN =========
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")  # temperature=0 gives more deterministic output
chain = LLMChain(llm=llm, prompt=prompt)

# ========= 5. LOOP OVER THE DATA AND GET COMPARISONS =========
results = []

for idx, row in df.iterrows():
    row_id = idx + 1

    human_decision = row.get("human_decision", "")
    guidearch_winner = row.get("winner", "")
    if pd.isna(human_decision):
        human_decision = ""
    if pd.isna(guidearch_winner):
        guidearch_winner = "No GuideArch winner"

    response = chain.run({
        "row_id": row_id,
        "human_decision": human_decision,
        "guidearch_winner": guidearch_winner
    })

    results.append(response)

# ========= 6. STORE THE RESULTS BACK INTO THE DATAFRAME OR A NEW DF =========
df["comparison_result"] = results

# ========= 7. SAVE TO A NEW CSV FILE =========
output_path = 'winner_vs_human_clean_comparison.csv'
df.to_csv(output_path, index=False)

print("Done. The results have been saved to:", output_path)