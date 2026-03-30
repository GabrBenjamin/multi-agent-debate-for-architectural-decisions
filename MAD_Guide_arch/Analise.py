import os
import json
import re
import pandas as pd
from Utils.Env import set_env

# ------- LangChain imports -------
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

# ========= ENV =========
set_env("OPENAI_API_KEY")

# ========= CONFIG =========
INPUT_CSV  = r"debateC_1.csv"               # <-- change if needed
OUTPUT_CSV = "adr_enriched_minimal.csv"
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0

# ========= LOAD =========
df = pd.read_csv(INPUT_CSV)
for col in ["topic", "human_decision"]:
    if col not in df.columns:
        raise ValueError(f"CSV must contain '{col}' column.")

def safe_get(row, name):
    val = row.get(name, "")
    return "" if pd.isna(val) else str(val)

# ========= PROMPT (STRICT JSON) =========
template = """You are extracting structured metadata from an ADR.

Inputs:
- TOPIC_BUNDLE: contains Context + Decision Drivers + Considered Options (merged)
- HUMAN_DECISION_BUNDLE: contains the final Decision and/or Rationale

Return ONLY ONE JSON object with EXACTLY these keys (no extra keys, no prose):

{
  "concerns": [string],                  // e.g., ["availability","security"]
  "layer": null or string,               // one of: infra|platform|backend|data|api|frontend|cross-cutting
  "family": null or string,              // e.g., database_choice|deployment_strategy|auth_method|caching|api_style|messaging|cloud_provider|framework_library|documentation_process
  "conflicting_drivers_count": integer,  // number of antagonistic driver pairs present
  "ambiguity_level": null or string,     // low|medium|high
  "risk_flags": [string],                // e.g., ["data_loss","prod_outage","compliance"]
  "migration_complexity": null or string // small|medium|large
}

Rules:
- Use ONLY what the text supports; if unclear, be conservative (null/0/[]).
- Concerns multi-label from: availability, performance, latency, security, cost, maintainability, operability, data_privacy, observability, usability.
- Layer must be one of: infra, platform, backend, data, api, frontend, cross-cutting (or null).
- Family use compact tags like: database_choice, deployment_strategy, auth_method, caching, api_style, messaging, cloud_provider, framework_library, documentation_process (or null).
- Conflicting drivers: count classic antagonistic pairs (e.g., latency vs cost, security vs velocity, availability vs cost). Return the COUNT only.
- Ambiguity reflects vagueness vs measurable thresholds.
- Output STRICT JSON only (no backticks/code fences).

ROW_ID: {row_id}

--- TOPIC_BUNDLE (Context + Drivers + Options) ---
{topic_text}

--- HUMAN_DECISION_BUNDLE (Decision / Rationale) ---
{human_decision_text}
"""

prompt = PromptTemplate(
    input_variables=["row_id", "topic_text", "human_decision_text"],
    template=template
)

llm = ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME)
chain = LLMChain(llm=llm, prompt=prompt)

# ========= JSON RECOVERY =========
def extract_json(s: str):
    s = (s or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m1, m2 = s.find("{"), s.rfind("}")
        if m1 != -1 and m2 != -1 and m2 > m1:
            cand = s[m1:m2+1]
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
        s2 = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            return None

# ========= PREP OUTPUT COLS =========
TARGET_COLS = [
    "concerns",
    "layer",
    "family",
    "conflicting_drivers_count",
    "ambiguity_level",
    "risk_flags",
    "migration_complexity",
]
for c in TARGET_COLS:
    if c not in df.columns:
        df[c] = None

# ========= RUN =========
parsed, failed = 0, 0

for idx, row in df.iterrows():
    row_id = idx + 1
    topic_text = safe_get(row, "topic")
    human_decision_text = safe_get(row, "human_decision")

    resp = chain.run({
        "row_id": row_id,
        "topic_text": topic_text,
        "human_decision_text": human_decision_text
    })
    data = extract_json(resp)

    # Retry once if needed
    if data is None:
        retry = LLMChain(llm=llm, prompt=PromptTemplate(
            input_variables=prompt.input_variables,
            template=prompt.template + "\n\nIMPORTANT: Retry and output STRICT JSON ONLY, no commentary."
        ))
        resp = retry.run({
            "row_id": row_id,
            "topic_text": topic_text,
            "human_decision_text": human_decision_text
        })
        data = extract_json(resp)

    if data is None:
        failed += 1
        continue

    # Write results (lists as JSON strings so CSV stays valid)
    df.at[idx, "concerns"]                  = json.dumps(data.get("concerns") or [])
    df.at[idx, "layer"]                     = data.get("layer")
    df.at[idx, "family"]                    = data.get("family")
    df.at[idx, "conflicting_drivers_count"] = data.get("conflicting_drivers_count")
    df.at[idx, "ambiguity_level"]           = data.get("ambiguity_level")
    df.at[idx, "risk_flags"]                = json.dumps(data.get("risk_flags") or [])
    df.at[idx, "migration_complexity"]      = data.get("migration_complexity")

    parsed += 1

# ========= SAVE =========
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Parsed: {parsed}, Failed: {failed}. Saved to: {OUTPUT_CSV}")
