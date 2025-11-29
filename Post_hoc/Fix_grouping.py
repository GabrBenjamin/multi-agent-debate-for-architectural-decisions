# Cell: LLM backfill for decision_type, scope, layer, family (one call per missing field)

import os, re, json, time
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from Utils.Env import set_env

INPUT_CSV  = "Grouped_decisions_with_comparison.csv"
OUTPUT_CSV = "Grouped_decisions_with_comparison_filled.csv"

MODEL_NAME  = "gpt-5"
TEMPERATURE = 0
MAX_RETRIES = 1
RATE_LIMIT_DELAY_S = 0.0  # set to e.g. 0.2 if needed

TARGET_FIELDS = ["decision_type", "scope", "layer", "family"]  # ambiguity_level excluded per your note

# Allowed labels (must match your analysis code)
DECISION_TYPE_ENUM = ["structural","technology","behavior_integration","cross_cutting","process"]
SCOPE_ENUM         = ["system_wide","bounded_context","component_local"]
LAYER_ENUM         = ["infra","platform","backend","data","api","frontend","cross-cutting"]
FAMILY_ENUM        = ["database_choice","deployment_strategy","auth_method","caching","api_style",
                      "messaging","cloud_provider","framework_library","documentation_process"]

set_env("OPENAI_API_KEY")

# --- Field-specific prompts (STRICT JSON, one key only) ---

PROMPTS = {
    "decision_type": PromptTemplate(
        input_variables=["topic_text", "decision_text"],
        template=(
            "Classify the DECISION TYPE of this Architecture Decision Record (ADR).\n"
            "Pick ONE label from:\n"
            "  structural | technology | behavior_integration | cross_cutting | process\n\n"
            "Guidance:\n"
            "- structural: topology/decomposition/style (monolith vs microservices, service boundaries, event-driven/choreography/orchestration, layering, hexagonal, CQRS, saga as style)\n"
            "- technology: concrete product/platform/framework/service selection (PostgreSQL vs MongoDB, Kafka/RabbitMQ, React, Kubernetes, AWS/GCP/Azure, REST vs gRPC as stack choice)\n"
            "- behavior_integration: interaction semantics/contracts (API style as rules, versioning, idempotency, schema evolution)\n"
            "- cross_cutting: policies/tactics applied across components (authN/Z, encryption, logging/tracing/observability, resilience, blue/green, canary, compliance)\n"
            "- process: procedures/standards (branching strategy, release cadence, ADR policy/template, documentation standards)\n\n"
            "If multiple apply, choose the PRIMARY focus (structural > technology > behavior_integration > cross_cutting > process).\n"
            "If insufficient evidence, return null.\n\n"
            "Return STRICT JSON only:\n"
            '{{"decision_type": "<one_of_labels_or_null>"}}\n\n'
            "--- TOPIC ---\n{topic_text}\n\n--- DECISION/RATIONALE ---\n{decision_text}"
        )
    ),
    "scope": PromptTemplate(
        input_variables=["topic_text", "decision_text"],
        template=(
            "Classify the SCOPE of this ADR. Pick ONE:\n"
            "  system_wide | bounded_context | component_local\n\n"
            "Definitions:\n"
            "- system_wide: platform/org policy, shared infra/cluster/VPC, CI/CD-wide\n"
            "- bounded_context: domain/subsystem/bounded context level\n"
            "- component_local: single service/component/repo\n\n"
            "If unclear, return null.\n\n"
            "Return STRICT JSON only:\n"
            '{{"scope": "<one_label_or_null>"}}\n\n'
            "--- TOPIC ---\n{topic_text}\n\n--- DECISION/RATIONALE ---\n{decision_text}"
        )
    ),
    "layer": PromptTemplate(
        input_variables=["topic_text", "decision_text"],
        template=(
            "Classify the ARCHITECTURE LAYER if possible. Pick ONE:\n"
            "  infra | platform | backend | data | api | frontend | cross-cutting\n\n"
            "Use cross-cutting for transversal policies (auth/observability/resilience).\n"
            "If unclear, return null.\n\n"
            "Return STRICT JSON only:\n"
            '{{"layer": "<one_label_or_null>"}}\n\n'
            "--- TOPIC ---\n{topic_text}\n\n--- DECISION/RATIONALE ---\n{decision_text}"
        )
    ),
    "family": PromptTemplate(
        input_variables=["topic_text", "decision_text"],
        template=(
            "Classify the decision FAMILY (compact tag). Pick ONE:\n"
            "  database_choice | deployment_strategy | auth_method | caching | api_style | messaging | cloud_provider | framework_library | documentation_process\n\n"
            "If unclear, return null.\n\n"
            "Return STRICT JSON only:\n"
            '{{"family": "<one_label_or_null>"}}\n\n'
            "--- TOPIC ---\n{topic_text}\n\n--- DECISION/RATIONALE ---\n{decision_text}"
        )
    ),
}

def extract_json(s: str):
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            m1, m2 = s.find("{"), s.rfind("}")
            if m1 != -1 and m2 != -1 and m2 > m1:
                try:
                    return json.loads(s[m1:m2+1])
                except json.JSONDecodeError:
                    return None
            return None

def normalize(val, allowed):
    if val is None:
        return None
    v = str(val).strip().lower()
    return v if v in allowed else None

def is_missing(x):
    return pd.isna(x) or (isinstance(x, str) and x.strip() == "")

df = pd.read_csv(INPUT_CSV)

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# track provenance
for fld in TARGET_FIELDS:
    flag = f"{fld}_filled_llm"
    if flag not in df.columns:
        df[flag] = False

attempt_counts = {f: 0 for f in TARGET_FIELDS}
filled_counts  = {f: 0 for f in TARGET_FIELDS}

for idx, row in df.iterrows():
    topic = row.get("topic", "") or ""
    decision = row.get("human_decision", "") or ""
    for field in TARGET_FIELDS:
        if not is_missing(row.get(field, None)):
            continue
        attempt_counts[field] += 1

        prompt = PROMPTS[field]
        # first try
        raw = llm.invoke(prompt.format(topic_text=topic, decision_text=decision)).content
        parsed = extract_json(raw)

        # retry once if needed
        if parsed is None and MAX_RETRIES > 0:
            raw = llm.invoke((prompt.template + "\n\nIMPORTANT: Return STRICT JSON only. No commentary.")
                             .format(topic_text=topic, decision_text=decision)).content
            parsed = extract_json(raw)

        # pick and normalize
        val = None
        if isinstance(parsed, dict) and field in parsed:
            if field == "decision_type":
                val = normalize(parsed[field], DECISION_TYPE_ENUM)
            elif field == "scope":
                val = normalize(parsed[field], SCOPE_ENUM)
            elif field == "layer":
                val = normalize(parsed[field], LAYER_ENUM)
            elif field == "family":
                val = normalize(parsed[field], FAMILY_ENUM)

        if val is not None:
            df.at[idx, field] = val
            df.at[idx, f"{field}_filled_llm"] = True
            filled_counts[field] += 1

        if RATE_LIMIT_DELAY_S > 0:
            time.sleep(RATE_LIMIT_DELAY_S)

print("LLM backfill attempts per field:", attempt_counts)
print("LLM successfully filled per field:", filled_counts)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
