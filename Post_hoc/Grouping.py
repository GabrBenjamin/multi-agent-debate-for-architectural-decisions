
"""
Single consolidated grouping pipeline based on the code stages shared in chat.

Stages preserved:
1) Initial ADR metadata extraction -> Grouped_decisions.csv
2) Row-order merge of comparison_result -> Grouped_decisions_with_comparison.csv
3) LLM backfill for missing decision_type/scope/layer/family -> Grouped_decisions_with_comparison_filled.csv
4) Concern split into concerns + main_concerns -> Grouped_decisions_with_concerns_split.csv
5) Remap decision_type == technology -> Final_Grouping.csv
6) Family regrouping to family_kind7 / family_kind8 -> Final_Grouping2.csv
"""

import os
import re
import json
import time
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate as CorePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

try:
    from Utils.Env import set_env
    set_env("OPENAI_API_KEY")
except Exception:
    pass


# =========================================================
# CONFIG
# =========================================================

RAW_INPUT_CSV = "debate_BIG_GPT.csv"
COMPARISON_SOURCE_CSV = "output_with_comparisonsBIG_GPT.csv"

STEP1_OUTPUT = "Grouped_decisions.csv"
STEP2_OUTPUT = "Grouped_decisions_with_comparison.csv"
STEP3_OUTPUT = "Grouped_decisions_with_comparison_filled.csv"
STEP4_OUTPUT = "Grouped_decisions_with_concerns_split.csv"
STEP5_OUTPUT = "Final_Grouping.csv"
FINAL_OUTPUT = "Final_Grouping2.csv"

MODEL_NAME = "gpt-5"
TEMPERATURE = 0
RATE_LIMIT_DELAY_S = 0.0


# =========================================================
# STAGE 1 — INITIAL EXTRACTOR
# Based on your first extractor script.
# =========================================================

CONCERNS = {
    "availability": [r"\bavailability\b", r"\buptime\b", r"\bSLA\b", r"\bSLO\b", r"MTBF", r"HA\b"],
    "performance": [r"\bperformance\b", r"\bthroughput\b", r"\bTPS\b"],
    "latency": [r"\blatency\b", r"\bP95\b", r"\bP99\b", r"\brespon(se|se[-\s]?time)\b"],
    "security": [r"\bsecurity\b", r"\bencrypt", r"\bauth", r"\bauthoriz", r"\bOWASP\b"],
    "cost": [r"\bcost\b", r"\bbudget\b", r"\bpricing\b", r"\bspend\b"],
    "maintainability": [r"\bmaintainab", r"\bmodifiab", r"\bchange effort\b", r"\btech debt\b"],
    "operability": [r"\boperab", r"\boperat(ions|ional)\b", r"\bSRE\b", r"\bMTTR\b"],
    "data_privacy": [r"\bprivacy\b", r"\bGDPR\b", r"\bPII\b", r"\bHIPAA\b"],
    "observability": [r"\bobservab", r"\blog(ging)?\b", r"\btracing\b", r"\bmetrics?\b"],
    "usability": [r"\busabil", r"\bUX\b", r"\buser[-\s]?friendly\b"],
}

LAYER_ENUM = {"infra", "platform", "backend", "data", "api", "frontend", "cross-cutting"}
FAMILY_ENUM = {
    "database_choice", "deployment_strategy", "auth_method", "caching", "api_style",
    "messaging", "cloud_provider", "framework_library", "documentation_process"
}
DECISION_TYPE_ENUM = {"structural", "technology", "behavior_integration", "cross_cutting", "process"}
SCOPE_ENUM = {"system_wide", "bounded_context", "component_local"}
LIFECYCLE_ENUM = {"inception", "design", "implementation", "evolution"}

STRUCTURAL_KW = [
    r"\bmonolith\b", r"\bmicroservices?\b", r"\bservice boundary\b", r"\bmodule\b", r"\blayer(ed|ing)\b",
    r"\bevent[-\s]?driven\b", r"\bchoreograph(y|ed)\b", r"\borchestrati(on|ng)\b", r"\btopolog"
]
TECHNOLOGY_KW = [
    r"\b(PostgreSQL|MySQL|MongoDB|Kafka|RabbitMQ|Redis|Elasticsearch|Kubernetes|Docker|gRPC|REST|GraphQL|React|Django|Spring|AWS|GCP|Azure)\b"
]
BEHAVIOR_KW = [
    r"\bcontract\b", r"\bschema\b", r"\bversioning\b", r"\bidempotenc", r"\bevent schema\b", r"\bAPI style\b",
    r"\bREST\b", r"\bGraphQL\b", r"\bgRPC\b"
]
CROSSCUT_KW = [
    r"\bauth(enticat|oriza)", r"\bencrypt", r"\blog(ging)?\b", r"\btracing\b", r"\bobservab",
    r"\bpolicy\b", r"\brollout\b", r"\bblue/?green\b", r"\bcanary\b", r"\bresilien(ce|t)\b", r"\bcircuit breaker\b",
]
PROCESS_KW = [
    r"\bbranch(ing)? strategy\b", r"\brelease cadence\b", r"\bADR policy\b", r"\bdocument(ation)? standard\b",
    r"\bdefinition of done\b", r"\bcontribution guidelin"
]

SYSTEM_WIDE_KW = [
    r"\borganization[-\s]?wide\b", r"\bplatform\b", r"\bcluster\b", r"\bVPC\b", r"\bglobal policy\b", r"\bCI/CD\b"
]
BOUNDED_CONTEXT_KW = [
    r"\bbounded context\b", r"\bcontext:\s*\w+", r"\bdomain\b", r"\bsubsystem\b", r"\bmodule\b"
]
COMPONENT_LOCAL_KW = [
    r"\b(service|component|module|repo)\b", r"\bmicroservice\b", r"\bpackage\b"
]

INCEPTION_KW = [r"\bspike\b", r"\bprototype\b", r"\bvision\b"]
DESIGN_KW = [r"\bwe will adopt\b", r"\bdecide to\b", r"\bdesign choice\b", r"\bbefore implementation\b"]
IMPLEMENTATION_KW = [r"\bimplement(ing|ation)\b", r"\bPR\b", r"\bbranch\b", r"\bcod(e|ing)\b", r"\bmerge\b"]
EVOLUTION_KW = [r"\bmigrat(e|ion)\b", r"\bdeprecat(e|ion)\b", r"\breplace\b", r"\bcutover\b", r"\brollback\b", r"\bblue/?green\b", r"\bcanary\b"]

PATTERNS = [
    "circuit breaker", "event sourcing", "saga", "CQRS", "bulkhead", "repository",
    "adapter", "proxy", "facade", "hexagonal", "retry", "backoff", "compensation"
]

TRADEOFF_PAIRS = [
    ("latency", "cost"),
    ("availability", "cost"),
    ("performance", "modifiability"),
    ("security", "velocity"),
    ("consistency", "availability"),
    ("privacy", "observability"),
    ("reliability", "time_to_market"),
]
TRADEOFF_SYNONYMS = {
    "latency": ["latency", "response time", "p95", "p99"],
    "cost": ["cost", "budget", "pricing", "spend"],
    "availability": ["availability", "uptime", "SLA", "SLO", "HA"],
    "performance": ["performance", "throughput", "tps", "qps"],
    "modifiability": ["modifiability", "maintainability", "change effort", "tech debt"],
    "security": ["security", "encryption", "auth", "authorization", "owasp"],
    "velocity": ["velocity", "delivery speed", "time to deliver", "cadence"],
    "consistency": ["consistency", "ACID", "strict consistency", "linearizability"],
    "privacy": ["privacy", "GDPR", "PII", "HIPAA"],
    "observability": ["observability", "logging", "tracing", "metrics", "telemetry"],
    "reliability": ["reliability", "error rate", "failure rate", "fault tolerance"],
    "time_to_market": ["time to market", "TTM", "release faster", "deliver quickly"],
}
CONTRAST_CUES_RE = re.compile(r"\b(but|however|trade[-\s]?off|at the expense of|versus|vs\.?|tension)\b", re.I)

VAGUE_WORDS = re.compile(r"\b(fast|scalable|robust|secure|simple|lightweight|flexible|resilient|reliable|easy)\b", re.I)
METRIC_RE = re.compile(r"(p9[59])|(\b\d{2,4}ms\b)|(\b\d{1,2}\.\d%|\b\d{1,3}%\b)|(\b\d+ (rps|qps|tps)\b)|(\b99\.\d{1,2}%\b)", re.I)
ACCEPT_RE = re.compile(r"\b(acceptance criteria|SLO|SLA|health checks?|1[-\s]?click rollback|error rate <|latency <)\b", re.I)
CONTRA_RE = re.compile(r"\b(on the other hand|conflict|contradict|inconsistent)\b", re.I)

RISK_FLAGS_ALLOW = {"data_loss", "prod_outage", "compliance", "security_breach", "downtime", "cost_overrun"}


def safe_get(row, name):
    val = row.get(name, "")
    return "" if pd.isna(val) else str(val)


def extract_json(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m1, m2 = s.find("{"), s.rfind("}")
        if m1 != -1 and m2 != -1 and m2 > m1:
            cand = s[m1:m2 + 1]
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
        s2 = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            return None


def any_match(patterns, text):
    return any(re.search(p, text, re.I) for p in patterns)


def classify_decision_type(text):
    if any_match(STRUCTURAL_KW, text):
        return "structural"
    if any_match(TECHNOLOGY_KW, text):
        return "technology"
    if any_match(BEHAVIOR_KW, text):
        return "behavior_integration"
    if any_match(CROSSCUT_KW, text):
        return "cross_cutting"
    if any_match(PROCESS_KW, text):
        return "process"
    return None


def classify_scope(text):
    if any_match(SYSTEM_WIDE_KW, text):
        return "system_wide"
    if any_match(BOUNDED_CONTEXT_KW, text):
        return "bounded_context"
    if any_match(COMPONENT_LOCAL_KW, text):
        return "component_local"
    return None


def classify_lifecycle(text):
    if any_match(EVOLUTION_KW, text):
        return "evolution"
    if any_match(IMPLEMENTATION_KW, text):
        return "implementation"
    if any_match(DESIGN_KW, text):
        return "design"
    if any_match(INCEPTION_KW, text):
        return "inception"
    return "design"


def normalize_enum(value, allowed_set):
    if value is None:
        return None
    v = str(value).strip().lower()
    return v if v in allowed_set else None


def normalize_list_strings(lst):
    if not lst:
        return []
    out = []
    for x in lst:
        if not x:
            continue
        out.append(str(x).strip())
    return out


def whitelist_risk_flags(flags):
    return [f for f in normalize_list_strings(flags) if f in RISK_FLAGS_ALLOW]


def find_patterns(text):
    found = []
    for p in PATTERNS:
        if re.search(rf"\b{re.escape(p)}\b", text, re.I):
            found.append(p)
    return found[0] if found else None


def detect_concerns(text):
    found = set()
    for k, pats in CONCERNS.items():
        if any(re.search(p, text, re.I) for p in pats):
            found.add(k)
    return sorted(found)


def count_conflicting_drivers(text):
    text_l = text.lower()
    total = 0
    for a, b in TRADEOFF_PAIRS:
        a_words = "|".join(re.escape(w) for w in TRADEOFF_SYNONYMS[a])
        b_words = "|".join(re.escape(w) for w in TRADEOFF_SYNONYMS[b])
        if re.search(a_words, text_l) and re.search(b_words, text_l):
            if re.search(CONTRAST_CUES_RE, text_l):
                total += 1
        if total >= 4:
            break
    return total


def score_ambiguity(text):
    score = 2
    if VAGUE_WORDS.search(text):
        score += 1
    if METRIC_RE.search(text):
        score -= 1
    if ACCEPT_RE.search(text):
        score -= 1
    if CONTRA_RE.search(text):
        score += 1

    if score <= 0:
        level = "low"
    elif score >= 3:
        level = "high"
    else:
        level = "medium"

    dist = abs(score - 2)
    conf = 0.5 + min(0.4, 0.2 * dist)
    return level, round(conf, 2)


EXTRACTOR_TEMPLATE = """You are extracting structured metadata from an Architecture Decision Record (ADR).

Inputs:
- TOPIC_BUNDLE: Context + Decision Drivers + Considered Options (merged)
- HUMAN_DECISION_BUNDLE: Decision and/or Rationale

Return ONE JSON object with EXACTLY these keys (no extra keys, no prose):

{{
  "concerns": [string],
  "layer": null or string,
  "family": null or string,
  "pattern_or_tactic": null or string,
  "risk_flags": [string],
  "forces_documented": boolean,
  "alternatives_listed": boolean,
  "consequences_listed": boolean,
  "decision_type": null or string,
  "scope": null or string,
  "lifecycle_stage": null or string,
  "conflicting_drivers_count": integer,
  "ambiguity_level": null or string
}}

Rules to stay objective:
- Only use information supported by Decision/Rationale; be conservative.
- concerns: choose from availability, performance, latency, security, cost, maintainability, operability, data_privacy, observability, usability.
- layer: one of {{infra, platform, backend, data, api, frontend, cross-cutting}}.
- family: compact tags from {{database_choice, deployment_strategy, auth_method, caching, api_style, messaging, cloud_provider, framework_library, documentation_process}}.
- pattern_or_tactic: set only if the Decision states adoption/usage; ignore patterns mentioned only as rejected options.
- risk_flags: pick compact flags like data_loss, prod_outage, compliance, security_breach, downtime, cost_overrun.
- decision_type (deterministic priority): if topology/boundaries/styles change -> structural; else if product/tool selection -> technology; else if interaction style/contracts/schemas -> behavior_integration; else if policy (security/observability/deploy) -> cross_cutting; else if procedures/standards -> process.
- scope: system_wide beats bounded_context beats component_local when multiple scopes appear.
- lifecycle_stage (keywords): evolution if migrate/deprecate/replace/cutover; else implementation if implementing/PR/branch/coding; else design; inception only for spike/prototype/vision.
- conflicting_drivers_count: count antagonistic pairs only when contrasted (e.g., "but", "however", "trade-off", "vs"). Use known pairs like latency↔cost, availability↔cost, performance↔modifiability, security↔velocity, consistency↔availability, privacy↔observability, reliability↔time to market. Return the COUNT.
- ambiguity_level: low if metrics/SLOs or explicit acceptance criteria are present; high if vague adjectives/modals dominate or contradictions; else medium.

Output STRICT JSON only (no backticks).
ROW_ID: {{row_id}}

--- TOPIC_BUNDLE ---
{{topic_text}}

--- HUMAN_DECISION_BUNDLE ---
{{human_decision_text}}
"""


# =========================================================
# STAGE 2 — BACKFILL
# Based on your backfill script.
# =========================================================

BACKFILL_TARGET_FIELDS = ["decision_type", "scope", "layer", "family"]

BACKFILL_DECISION_TYPE_ENUM = ["structural", "technology", "behavior_integration", "cross_cutting", "process"]
BACKFILL_SCOPE_ENUM = ["system_wide", "bounded_context", "component_local"]
BACKFILL_LAYER_ENUM = ["infra", "platform", "backend", "data", "api", "frontend", "cross-cutting"]
BACKFILL_FAMILY_ENUM = [
    "database_choice", "deployment_strategy", "auth_method", "caching", "api_style",
    "messaging", "cloud_provider", "framework_library", "documentation_process"
]

BACKFILL_PROMPTS = {
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


def backfill_extract_json(s: str):
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
                    return json.loads(s[m1:m2 + 1])
                except json.JSONDecodeError:
                    return None
            return None


def backfill_normalize(val, allowed):
    if val is None:
        return None
    v = str(val).strip().lower()
    return v if v in allowed else None


def is_missing(x):
    return pd.isna(x) or (isinstance(x, str) and x.strip() == "")


# =========================================================
# STAGE 3 — CONCERN SPLIT
# Based on your concern split script.
# =========================================================

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
    label = str(raw).strip().lower().replace(" ", "_")
    label = ALIASES.get(label, label)
    return label if label in VOCAB_SET else None


def clean_list(values):
    if not isinstance(values, list):
        return []
    normalized = {label for item in values if (label := norm_label(item))}
    return sorted(normalized)


CONCERN_SPLIT_PROMPT = PromptTemplate(
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


# =========================================================
# RUN PIPELINE
# =========================================================

print("Stage 1/6: Initial extractor")
df = pd.read_csv(RAW_INPUT_CSV)
for col in ["topic", "human_decision"]:
    if col not in df.columns:
        raise ValueError("CSV must contain 'topic' and 'human_decision' columns.")

extractor_prompt = CorePromptTemplate(
    input_variables=["row_id", "topic_text", "human_decision_text"],
    template=EXTRACTOR_TEMPLATE
)
llm_core = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, max_retries=6, timeout=120)
parser = StrOutputParser()
chain = extractor_prompt | llm_core | parser

TARGET_COLS = [
    "concerns",
    "layer",
    "family",
    "pattern_or_tactic",
    "risk_flags",
    "forces_documented",
    "alternatives_listed",
    "consequences_listed",
    "decision_type",
    "scope",
    "lifecycle_stage",
    "conflicting_drivers_count",
    "ambiguity_level",
    "ambiguity_confidence",
]
for c in TARGET_COLS:
    if c not in df.columns:
        df[c] = None

parsed_count, failed_count = 0, 0

for idx, row in df.iterrows():
    row_id = idx + 1
    topic_text = safe_get(row, "topic")
    human_decision_text = safe_get(row, "human_decision")
    bundle_text = f"{topic_text}\n\n{human_decision_text}"

    resp = chain.invoke({
        "row_id": row_id,
        "topic_text": topic_text,
        "human_decision_text": human_decision_text,
    })
    data = extract_json(resp)

    if data is None:
        retry_prompt = CorePromptTemplate(
            input_variables=extractor_prompt.input_variables,
            template=extractor_prompt.template + "\n\nIMPORTANT: Retry and output STRICT JSON ONLY, no commentary."
        )
        retry_chain = retry_prompt | llm_core | parser
        resp = retry_chain.invoke({
            "row_id": row_id,
            "topic_text": topic_text,
            "human_decision_text": human_decision_text,
        })
        data = extract_json(resp)

    if data is None:
        failed_count += 1
        continue

    llm_concerns = normalize_list_strings(data.get("concerns") or [])
    det_concerns = detect_concerns(bundle_text)
    concerns = sorted(set([c for c in llm_concerns if c in CONCERNS.keys()]) | set(det_concerns))

    layer = normalize_enum(data.get("layer"), LAYER_ENUM)

    family = data.get("family")
    family = family if (family and family in FAMILY_ENUM) else None

    pat = find_patterns(bundle_text) or data.get("pattern_or_tactic")
    pattern_or_tactic = None
    if pat:
        for p in PATTERNS:
            if p.lower() == str(pat).lower():
                pattern_or_tactic = p
                break

    risk_flags = whitelist_risk_flags(data.get("risk_flags") or [])

    def as_bool_stage1(x):
        return bool(x) if isinstance(x, bool) else False

    forces_documented = as_bool_stage1(data.get("forces_documented"))
    alternatives_listed = as_bool_stage1(data.get("alternatives_listed"))
    consequences_listed = as_bool_stage1(data.get("consequences_listed"))

    dt_llm = normalize_enum(data.get("decision_type"), DECISION_TYPE_ENUM)
    dt_det = classify_decision_type(bundle_text)
    decision_type = dt_det or dt_llm

    sc_llm = normalize_enum(data.get("scope"), SCOPE_ENUM)
    sc_det = classify_scope(bundle_text)
    scope_order = {"system_wide": 3, "bounded_context": 2, "component_local": 1, None: 0}
    scope = sc_llm if scope_order.get(sc_llm, 0) >= scope_order.get(sc_det, 0) else sc_det

    lc_llm = normalize_enum(data.get("lifecycle_stage"), LIFECYCLE_ENUM)
    lc_det = classify_lifecycle(bundle_text)
    lifecycle_stage = lc_llm or lc_det

    conflicting_drivers_count = count_conflicting_drivers(bundle_text)

    ambiguity_level, ambiguity_conf = score_ambiguity(bundle_text)

    df.at[idx, "concerns"] = json.dumps(concerns)
    df.at[idx, "layer"] = layer
    df.at[idx, "family"] = family
    df.at[idx, "pattern_or_tactic"] = pattern_or_tactic
    df.at[idx, "risk_flags"] = json.dumps(risk_flags)
    df.at[idx, "forces_documented"] = forces_documented
    df.at[idx, "alternatives_listed"] = alternatives_listed
    df.at[idx, "consequences_listed"] = consequences_listed
    df.at[idx, "decision_type"] = decision_type
    df.at[idx, "scope"] = scope
    df.at[idx, "lifecycle_stage"] = lifecycle_stage
    df.at[idx, "conflicting_drivers_count"] = int(conflicting_drivers_count)
    df.at[idx, "ambiguity_level"] = ambiguity_level
    df.at[idx, "ambiguity_confidence"] = ambiguity_conf

    parsed_count += 1

    if RATE_LIMIT_DELAY_S > 0:
        time.sleep(RATE_LIMIT_DELAY_S)

df.to_csv(STEP1_OUTPUT, index=False)
print(f"Done. Parsed: {parsed_count}, Failed: {failed_count}. Saved to: {STEP1_OUTPUT}")

print("Stage 2/6: Merge comparison_result by row order")
df_grouped = pd.read_csv(STEP1_OUTPUT)
df_comparisons = pd.read_csv(COMPARISON_SOURCE_CSV)

if len(df_grouped) != len(df_comparisons):
    raise ValueError(
        f"Row mismatch: grouped={len(df_grouped)}, comparisons={len(df_comparisons)}"
    )

df_grouped["comparison_result"] = df_comparisons["comparison_result"].values
df_grouped.to_csv(STEP2_OUTPUT, index=False)
print(f"Saved: {STEP2_OUTPUT}")

print("Stage 3/6: Backfill missing decision_type/scope/layer/family")
df = pd.read_csv(STEP2_OUTPUT)

llm_backfill = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

for fld in BACKFILL_TARGET_FIELDS:
    flag = f"{fld}_filled_llm"
    if flag not in df.columns:
        df[flag] = False

attempt_counts = {f: 0 for f in BACKFILL_TARGET_FIELDS}
filled_counts = {f: 0 for f in BACKFILL_TARGET_FIELDS}

for idx, row in df.iterrows():
    topic = row.get("topic", "") or ""
    decision = row.get("human_decision", "") or ""

    for field in BACKFILL_TARGET_FIELDS:
        if not is_missing(row.get(field, None)):
            continue

        attempt_counts[field] += 1
        prompt = BACKFILL_PROMPTS[field]

        raw = llm_backfill.invoke(prompt.format(topic_text=topic, decision_text=decision)).content
        parsed = backfill_extract_json(raw)

        if parsed is None:
            raw = llm_backfill.invoke(
                (prompt.template + "\n\nIMPORTANT: Return STRICT JSON only. No commentary.")
                .format(topic_text=topic, decision_text=decision)
            ).content
            parsed = backfill_extract_json(raw)

        val = None
        if isinstance(parsed, dict) and field in parsed:
            if field == "decision_type":
                val = backfill_normalize(parsed[field], BACKFILL_DECISION_TYPE_ENUM)
            elif field == "scope":
                val = backfill_normalize(parsed[field], BACKFILL_SCOPE_ENUM)
            elif field == "layer":
                val = backfill_normalize(parsed[field], BACKFILL_LAYER_ENUM)
            elif field == "family":
                val = backfill_normalize(parsed[field], BACKFILL_FAMILY_ENUM)

        if val is not None:
            df.at[idx, field] = val
            df.at[idx, f"{field}_filled_llm"] = True
            filled_counts[field] += 1

        if RATE_LIMIT_DELAY_S > 0:
            time.sleep(RATE_LIMIT_DELAY_S)

print("LLM backfill attempts per field:", attempt_counts)
print("LLM successfully filled per field:", filled_counts)

df.to_csv(STEP3_OUTPUT, index=False)
print(f"Saved: {STEP3_OUTPUT}")

print("Stage 4/6: Split concerns into concerns + main_concerns")
df = pd.read_csv(STEP3_OUTPUT)
required_cols = {"topic", "human_decision"}
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")

if "main_concerns" not in df.columns:
    df["main_concerns"] = None

llm_concerns = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

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

    message = CONCERN_SPLIT_PROMPT.format(
        row_id=i + 1,
        drivers_source=drivers_source,
        outcome_source=outcome_source,
        vocab=", ".join(VOCAB),
    )

    response = llm_concerns.invoke(message).content
    payload = parse_json_block(response)

    if not (isinstance(payload, dict) and "drivers_concerns" in payload and "outcome_primary_concerns" in payload):
        retry_message = (
            message
            + "\n\nIMPORTANT: Return STRICT JSON only, with keys drivers_concerns and outcome_primary_concerns."
        )
        response = llm_concerns.invoke(retry_message).content
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

    if RATE_LIMIT_DELAY_S > 0:
        time.sleep(RATE_LIMIT_DELAY_S)

print(
    "Rows processed: {rows} | concerns filled: {drivers} | main_concerns filled: {main} | parse failures: {fails}".format(
        rows=rows_processed,
        drivers=drivers_non_empty,
        main=main_non_empty,
        fails=parse_failures,
    )
)

df.to_csv(STEP4_OUTPUT, index=False)
print(f"Saved: {STEP4_OUTPUT}")

print("Stage 5/6: Remap decision_type == technology")
df = pd.read_csv(STEP4_OUTPUT)
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
    bundle = (
        df_new.loc[mask, "topic"].fillna("")
        + "\n"
        + df_new.loc[mask, "human_decision"].fillna("")
    )
    df_new.loc[mask, "decision_type"] = [
        remap_technology_to_kruchten(txt) for txt in bundle.tolist()
    ]

df_new.to_csv(STEP5_OUTPUT, index=False)
print(f"Saved: {STEP5_OUTPUT}")

print("Stage 6/6: Family regrouping to family_kind7 / family_kind8")
df = pd.read_csv(STEP5_OUTPUT)

to_kind7 = {
    "database_choice": "Data",
    "framework_library": "Application (Implementation)",
    "api_style": "Integration & Interfaces",
    "messaging": "Integration & Interfaces",
    "caching": "Integration & Interfaces",
    "deployment_strategy": "Platform & Delivery",
    "cloud_provider": "Platform & Delivery",
    "auth_method": "Security & Compliance",
    "documentation_process": "Governance & Documentation",
}

df["family_kind7"] = df["family"].map(to_kind7)

CUES = [
    ("Operations & Observability",
     r"(observab|telemetry|metric|prometheus|grafana|opentelemetry|trace|jaeger|logging|elk|splunk|datadog|new\s*relic|slo|sla|circuit breaker|retry|timeout|rate limit|backoff|runbook)"),
    ("Platform & Delivery",
     r"(kubernetes|k8s|cluster|namespace|vpc|subnet|peering|vpn|waf|alb|nlb|load balancer|ingress|egress|dns|terraform|ansible|cicd|pipeline|blue[- ]green|canary|feature flag|progressive delivery|traffic split|weighted routing|aws|gcp|azure|region|account|landing zone)"),
    ("Security & Compliance",
     r"(oidc|oauth|saml|jwt|rbac|abac|mtls|encrypt|kms|vault|secret(s)? manager|key rotation|privacy|gdpr|hipaa|sox|pci)"),
    ("Integration & Interfaces",
     r"(rest|grpc|graphql|openapi|contract|schema version|api gateway|event[- ](bus|driven)|kafka|rabbitmq|pub/?sub|topic|consumer group|idl)"),
    ("Application (Implementation)",
     r"(framework|runtime|spring|django|rails|quarkus|nestjs|monolith|microservice|service layer|domain layer|hexagonal|clean architecture)"),
    ("Data",
     r"(schema|database|postgres|mysql|mongodb|dynamodb|elasticsearch|index|shard|replica|warehouse|data lake|retention|lineage)"),
    ("Governance & Documentation",
     r"(adr(s)?\b|architecture (decision|record)|governance|standard(s)?|guideline(s)?|rfc\b|approval|review process|definition of done|template)"),
]


def backfill_kind7(row):
    if pd.notna(row.get("family_kind7")):
        return row["family_kind7"]
    text = f"{row.get('topic', '')} {row.get('human_decision', '')}".lower()
    for label, pat in CUES:
        if re.search(pat, text):
            return label
    return None


mask_na = df["family_kind7"].isna()
df.loc[mask_na, "family_kind7"] = df.loc[mask_na].apply(backfill_kind7, axis=1)

if "layer" in df.columns:
    m = df["family_kind7"].isna()
    layer = df.loc[m, "layer"].astype(str).str.lower()
    df.loc[m & layer.eq("data"), "family_kind7"] = "Data"
    df.loc[m & layer.eq("api"), "family_kind7"] = "Integration & Interfaces"
    df.loc[m & layer.eq("frontend"), "family_kind7"] = "Application (Implementation)"
    df.loc[m & layer.eq("backend"), "family_kind7"] = "Application (Implementation)"
    df.loc[m & layer.isin(["infra", "platform"]), "family_kind7"] = "Platform & Delivery"
    df.loc[m & layer.eq("cross-cutting"), "family_kind7"] = "Security & Compliance"

df["family_kind8"] = df["family_kind7"]

mask_app = df["family_kind8"].eq("Application (Implementation)")
lay = df["layer"].astype(str).str.lower()

df.loc[mask_app & lay.eq("frontend"), "family_kind8"] = "Application — Frontend"
df.loc[mask_app & lay.eq("backend"), "family_kind8"] = "Application — Backend"

FE_RE = re.compile(r"\b(ui|frontend|react|angular|vue|svelte|next\.js|nuxt|tailwind|css|html|web client|spa|pwa|ios|android|flutter|react native)\b", re.I)
BE_RE = re.compile(r"\b(backend|service|microservice|spring|django|rails|quarkus|nest(js)?|ktor|express|fastapi|domain layer|hexagonal|clean architecture)\b", re.I)
API_RE = re.compile(r"\b(rest|grpc|graphql|openapi|contract|schema version(ing)?|api[-\s]?gateway|idl)\b", re.I)

text = (df["topic"].fillna("") + " " + df["human_decision"].fillna(""))

df.loc[
    mask_app & df["family_kind8"].eq("Application (Implementation)") & text.str.contains(FE_RE, na=False),
    "family_kind8"
] = "Application — Frontend"

df.loc[
    mask_app & df["family_kind8"].eq("Application (Implementation)") & text.str.contains(BE_RE, na=False),
    "family_kind8"
] = "Application — Backend"

df.loc[
    df["family_kind8"].eq("Application (Implementation)") & text.str.contains(API_RE, na=False),
    "family_kind8"
] = "Integration & Interfaces"

df.loc[df["family_kind8"].eq("Application (Implementation)"), "family_kind8"] = "Application — Backend"

df.to_csv(FINAL_OUTPUT, index=False)
print(f"Saved: {FINAL_OUTPUT}")

print("\nFinal decision_type distribution:")
print(df["decision_type"].value_counts(dropna=False))

print("\nFinal family_kind8 distribution:")
print(df["family_kind8"].value_counts(dropna=False))