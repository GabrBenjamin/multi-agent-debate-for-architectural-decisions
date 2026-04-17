import os
import pandas as pd

# ------- LangChain imports -------
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI




# OpenAI API key should be set via environment variable
# export OPENAI_API_KEY=your_key_here

# ========= 2. LOAD YOUR DATA =========
df = pd.read_csv(r'ATAM_debateGLGL.csv')  # <-- change to your actual CSV path
# Make sure it has columns 'human_decision' and 'debate_answer'.

# ========= 3. CREATE A LANGCHAIN PROMPT =========
# We'll inject {row_id}, {human_decision}, and {debate_answer} into this template.
template = """You are a helpful assistant. 
We have two decisions:
1) human_decision: {human_decision}
2) debate_answer: {debate_answer}

We want to see if both come to the same conclusion or choose the same option. 
Return a single line with "Yes" if they match or "No" if they do not match and maybe if one of them left it open but the discussion aligns, 
followed by "//" then a short explanation in a single line. 
We also have an ID for this row: {row_id}.

Return it in the exact format:
"<row_id> - <Yes/No/Maybe> // <short explanation>"

No extra lines or text.
"""

prompt = PromptTemplate(
    input_variables=["row_id", "human_decision", "debate_answer"],
    template=template
)

# ========= 4. SET UP THE LLM AND CHAIN =========
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")  # temperature=0 gives more deterministic output
chain = LLMChain(llm=llm, prompt=prompt)

# ========= 5. LOOP OVER THE DATA AND GET COMPARISONS =========
results = []

for idx, row in df.iterrows():
    # If you have a separate ID column, use row["id"]. Otherwise, just use idx+1
    row_id = idx + 1  
    
    human_decision = row["human_decision"]
    debate_answer = row["debate_answer"]

    # Run the chain
    response = chain.run({
        "row_id": row_id,
        "human_decision": human_decision,
        "debate_answer": debate_answer
    })

    # Example response could be: "3 - No // The first decision discusses..."
    results.append(response)

# ========= 6. STORE THE RESULTS BACK INTO THE DATAFRAME OR A NEW DF =========
df["comparison_result"] = results

# ========= 7. SAVE TO A NEW CSV FILE =========
output_path = "output_with_comparisonsATAM_GLGL_ALL.csv"
df.to_csv(output_path, index=False)

print("Done. The results have been saved to:", output_path)
