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


import pandas as pd
import re

df = pd.read_csv("Grouped_decisions_with_concerns_split.csv")
df_new = df.copy()

def remap_technology_to_kruchten(text):
    t = (text or "")
    if re.search(r"\b(REST|gRPC|GraphQL|pub/?sub|event[- ](driven|sourcing)|orchestrat|choreograph|API style|contract|schema|versioning)\b", t, re.I):
        return "behavior_integration"
    if re.search(r"\b(auth|OIDC|OAuth|mTLS|JWT|WAF|encryp|telemetry|OpenTelemetry|logging|tracing|feature[- ]flag|CDN|cache policy)\b", t, re.I):
        return "cross_cutting"
    if re.search(r"\b(CI/CD|pipeline|GitHub Actions|branching|release cadence|ADR policy|governance|definition of done)\b", t, re.I):
        return "process"
    if re.search(r"\b(PostgreSQL|MySQL|MongoDB|Kafka|RabbitMQ|Redis|Elasticsearch|Kubernetes|Docker|AWS|GCP|Azure|framework|runtime|broker|database|datastore|queue|bus)\b", t, re.I):
        return "structural"
    return "structural"

mask = df_new.get("decision_type") == "technology"
if mask.any():
    bundle = (df_new.loc[mask, "topic"].fillna("") +
              "\n" +
              df_new.loc[mask, "human_decision"].fillna(""))
    df_new.loc[mask, "decision_type"] = [
        remap_technology_to_kruchten(txt) for txt in bundle.tolist()
    ]

df_new.to_csv("Final_Grouping.csv", index=False)




import re
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from Utils.Env import set_env

set_env("OPENAI_API_KEY")

INPUT = "Grouped_decisions_with_comparison_filled.csv"
OUTPUT = "Grouped_decisions_with_concerns_split.csv"
MODEL = "gpt-5"
TEMP = 0

# Controlled vocabulary for concern labels
VOCAB = [
    "availability", "performance", "security", "modifiability",
    "usability", "interoperability", "testability",
    "cost", "time_to_market", "compliance",
    "operability", "observability"
]
VOCAB_SET = {v.lower() for v in VOCAB}

ALIASES = {
    "maintainability": "modifiability",
    "privacy": "security",
    "data_privacy": "security",
    "latency": "performance",
    "throughput": "performance",
    "time_behavior": "performance",
    "time-to-market": "time_to_market",
}


def parse_json_block(text: str):
    """Attempt to load JSON even if fenced in markdown code blocks."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m1, m2 = text.find("{"), text.rfind("}")
        if m1 != -1 and m2 != -1 and m2 > m1:
            try:
                return json.loads(text[m1:m2 + 1])
            except json.JSONDecodeError:
                return None
        return None


def norm_label(raw: str):
    """Normalize aliases and keep only labels from the controlled vocabulary."""
    label = str(raw).strip().lower().replace(" ", "_")
    label = ALIASES.get(label, label)
    return label if label in VOCAB_SET else None


def clean_list(values):
    if not isinstance(values, list):
        return []
    normalized = {label for item in values if (label := norm_label(item))}
    return sorted(normalized)


# --- Load data

df = pd.read_csv(INPUT)
required_cols = {"topic", "human_decision"}
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")

# Ensure output columns exist
if "main_concerns" not in df.columns:
    df["main_concerns"] = None

llm = ChatOpenAI(model=MODEL, temperature=TEMP)

PROMPT = PromptTemplate(
    input_variables=["row_id", "drivers_source", "outcome_source", "vocab"],
    template=(
        "You analyze an Architecture Decision Record (ADR).\n\n"
        "Return STRICT JSON with exactly these keys (no prose):\n"
        "{{\n"
        '  "drivers_concerns": [string],\n'
        '  "outcome_primary_concerns": [string]\n'
        "}}\n\n"
        "Label set (use ONLY these): {vocab}\n\n"
        "Normalization rules:\n"
        "- maintainability -> modifiability\n"
        "- privacy / data protection -> security (unless clearly regulatory -> compliance)\n"
        "- latency / throughput / time behavior -> performance\n"
        "- time-to-market / schedule / speed to deliver -> time_to_market\n"
        "- run-ops / SRE / MTTR / on-call / deploy / backout -> operability\n"
        "- logs / metrics / traces / telemetry / SLOs -> observability\n"
        "- Include every concern that is clearly supported in the text.\n\n"
        "Source guidance:\n"
        "- drivers_concerns: use ONLY the **Context / Decision Drivers / Considered Options** block.\n"
        "- outcome_primary_concerns: use ONLY the **Decision Outcome / Rationale** block; capture the top 1-3 drivers for the chosen option.\n"
        "  Ignore concerns that apply solely to rejected alternatives.\n\n"
        "ROW_ID: {row_id}\n"
        "--- CONTEXT / DRIVERS / OPTIONS ---\n{drivers_source}\n\n"
        "--- DECISION OUTCOME / RATIONALE ---\n{outcome_source}\n"
    ),
)

rows_processed = len(df)
drivers_non_empty = 0
main_non_empty = 0
parse_failures = 0

for i, row in df.iterrows():
    drivers_source = "" if pd.isna(row["topic"]) else str(row["topic"])
    outcome_source = "" if pd.isna(row["human_decision"]) else str(row["human_decision"])

    if not drivers_source and not outcome_source:
        df.at[i, "concerns"] = json.dumps([])
        df.at[i, "main_concerns"] = json.dumps([])
        continue

    message = PROMPT.format(
        row_id=i + 1,
        drivers_source=drivers_source,
        outcome_source=outcome_source,
        vocab=", ".join(VOCAB),
    )

    response = llm.invoke(message).content
    payload = parse_json_block(response)

    if not (isinstance(payload, dict) and "drivers_concerns" in payload and "outcome_primary_concerns" in payload):
        retry_message = (
            message
            + "\n\nIMPORTANT: Return STRICT JSON only, with keys drivers_concerns and outcome_primary_concerns."
        )
        response = llm.invoke(retry_message).content
        payload = parse_json_block(response)
        if payload is None:
            parse_failures += 1

    drivers = clean_list(payload.get("drivers_concerns") if isinstance(payload, dict) else [])
    main = clean_list(payload.get("outcome_primary_concerns") if isinstance(payload, dict) else [])

    df.at[i, "concerns"] = json.dumps(drivers)
    df.at[i, "main_concerns"] = json.dumps(main)

    if drivers:
        drivers_non_empty += 1
    if main:
        main_non_empty += 1

print(
    "Rows processed: {rows} | concerns filled: {drivers} | main_concerns filled: {main} | parse failures: {fails}".format(
        rows=rows_processed,
        drivers=drivers_non_empty,
        main=main_non_empty,
        fails=parse_failures,
    )
)

df.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")


# Start from your current family_kind7
df["family_kind8"] = df["family_kind7"]

# 1) Split Application (Implementation) into Frontend vs Backend using `layer` first
mask_app = df["family_kind8"].eq("Application (Implementation)")
lay = df["layer"].astype(str).str.lower()

df.loc[mask_app & lay.eq("frontend"), "family_kind8"] = "Application — Frontend"
df.loc[mask_app & lay.eq("backend"),  "family_kind8"] = "Application — Backend"

# 2) For remaining app rows (no clear layer), use text cues
FE_RE = re.compile(r"\b(ui|frontend|react|angular|vue|svelte|next\.js|nuxt|tailwind|css|html|web client|spa|pwa|ios|android|flutter|react native)\b", re.I)
BE_RE = re.compile(r"\b(backend|service|microservice|spring|django|rails|quarkus|nest(js)?|ktor|express|fastapi|domain layer|hexagonal|clean architecture)\b", re.I)

text = (df["topic"].fillna("") + " " + df["human_decision"].fillna(""))

df.loc[mask_app & df["family_kind8"].eq("Application (Implementation)") & text.str.contains(FE_RE), "family_kind8"] = "Application — Frontend"
df.loc[mask_app & df["family_kind8"].eq("Application (Implementation)") & text.str.contains(BE_RE), "family_kind8"] = "Application — Backend"

# 3) Precision override: if an "Application" row is actually about APIs/contracts, relabel to Integration
API_RE = re.compile(r"\b(rest|grpc|graphql|openapi|contract|schema version(ing)?|api[-\s]?gateway|idl)\b", re.I)
df.loc[
    df["family_kind8"].eq("Application (Implementation)") & text.str.contains(API_RE),
    "family_kind8"
] = "Integration & Interfaces"

# 4) Any still-unsplit app rows → default to Backend (or leave as generic if you prefer)
df.loc[df["family_kind8"].eq("Application (Implementation)"), "family_kind8"] = "Application — Backend"

# 5) See distribution
print(df["family_kind8"].value_counts(dropna=False))
