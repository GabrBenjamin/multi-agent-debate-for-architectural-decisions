# -*- coding: utf-8 -*-
"""
ADR metadata extractor with rule-backed post-processing.

Keeps:
- Tier A: concerns, layer, family, pattern_or_tactic, risk_flags,
          forces_documented, alternatives_listed, consequences_listed
- Tier B: decision_type, scope, lifecycle_stage, conflicting_drivers_count
- Tier C: ambiguity_level

Input CSV must have: topic, human_decision
"""

import os
import re
import json
import pandas as pd

# ------- LangChain (modern) imports -------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ======== ENV ========
try:
    # If you use your own set_env util, keep it; otherwise rely on OS env var.
    from Utils.Env import set_env
    set_env("OPENAI_API_KEY")
except Exception:
    pass  # Fallback to raw env var

# ======== CONFIG ========
INPUT_CSV   = r"debate_BIG_GPT.csv"
OUTPUT_CSV  = "Grouped_decisions.csv"
MODEL_NAME  = "gpt-5"
TEMPERATURE = 0

# ======== LEXICONS & RULE TABLES ========
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
    "usability": [r"\busabil", r"\bUX\b", r"\buser[-\s]?friendly\b"]
}

LAYER_ENUM = {"infra", "platform", "backend", "data", "api", "frontend", "cross-cutting"}
FAMILY_ENUM = {
    "database_choice","deployment_strategy","auth_method","caching","api_style",
    "messaging","cloud_provider","framework_library","documentation_process"
}
DECISION_TYPE_ENUM = {"structural","technology","behavior_integration","cross_cutting","process"}
SCOPE_ENUM = {"system_wide","bounded_context","component_local"}
LIFECYCLE_ENUM = {"inception","design","implementation","evolution"}

# Keywords for classification
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

# Patterns / tactics dictionary (short sample; extend as needed)
PATTERNS = [
    "circuit breaker","event sourcing","saga","CQRS","bulkhead","repository","adapter","proxy","facade","hexagonal",
    "retry","backoff","compensation"
]

# Trade-off pairs and synonyms
TRADEOFF_PAIRS = [
    ("latency","cost"),
    ("availability","cost"),
    ("performance","modifiability"),
    ("security","velocity"),
    ("consistency","availability"),
    ("privacy","observability"),
    ("reliability","time_to_market"),
]
TRADEOFF_SYNONYMS = {
    "latency": ["latency","response time","p95","p99"],
    "cost": ["cost","budget","pricing","spend"],
    "availability": ["availability","uptime","SLA","SLO","HA"],
    "performance": ["performance","throughput","tps","qps"],
    "modifiability": ["modifiability","maintainability","change effort","tech debt"],
    "security": ["security","encryption","auth","authorization","owasp"],
    "velocity": ["velocity","delivery speed","time to deliver","cadence"],
    "consistency": ["consistency","ACID","strict consistency","linearizability"],
    "privacy": ["privacy","GDPR","PII","HIPAA"],
    "observability": ["observability","logging","tracing","metrics","telemetry"],
    "reliability": ["reliability","error rate","failure rate","fault tolerance"],
    "time_to_market": ["time to market","TTM","release faster","deliver quickly"]
}
CONTRAST_CUES_RE = re.compile(r"\b(but|however|trade[-\s]?off|at the expense of|versus|vs\.?|tension)\b", re.I)

# Ambiguity scoring
VAGUE_WORDS = re.compile(r"\b(fast|scalable|robust|secure|simple|lightweight|flexible|resilient|reliable|easy)\b", re.I)
METRIC_RE   = re.compile(r"(p9[59])|(\b\d{2,4}ms\b)|(\b\d{1,2}\.\d%|\b\d{1,3}%\b)|(\b\d+ (rps|qps|tps)\b)|(\b99\.\d{1,2}%\b)", re.I)
ACCEPT_RE   = re.compile(r"\b(acceptance criteria|SLO|SLA|health checks?|1[-\s]?click rollback|error rate <|latency <)\b", re.I)
CONTRA_RE   = re.compile(r"\b(on the other hand|conflict|contradict|inconsistent)\b", re.I)

# Risk flags (free list; keep LLM-suggested but validate)
RISK_FLAGS_ALLOW = {"data_loss","prod_outage","compliance","security_breach","downtime","cost_overrun"}

# ======== HELPERS ========
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

def any_match(patterns, text):
    return any(re.search(p, text, re.I) for p in patterns)

def classify_decision_type(text):
    # Priority: structural > technology > behavior > cross_cutting > process
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
    # default bias to design if truly unknown
    return "design"

def normalize_enum(value, allowed_set):
    if value is None: return None
    v = str(value).strip().lower()
    return v if v in allowed_set else None

def normalize_list_strings(lst):
    if not lst: return []
    out = []
    for x in lst:
        if not x: continue
        out.append(str(x).strip())
    return out

def whitelist_risk_flags(flags):
    return [f for f in normalize_list_strings(flags) if f in RISK_FLAGS_ALLOW]

def find_patterns(text):
    found = []
    for p in PATTERNS:
        if re.search(rf"\b{re.escape(p)}\b", text, re.I):
            found.append(p)
    # prefer a single most central – pick shortest name hit first (heuristic)
    return found[0] if found else None

def detect_concerns(text):
    found = set()
    for k, pats in CONCERNS.items():
        if any(re.search(p, text, re.I) for p in pats):
            found.add(k)
    return sorted(found)

def count_conflicting_drivers(text):
    # Count each antagonistic pair if both sides appear in proximity with contrast cue
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
    # Start at 2 (medium), then adjust
    score = 2
    t = text
    if VAGUE_WORDS.search(t): score += 1
    if METRIC_RE.search(t):   score -= 1
    if ACCEPT_RE.search(t):   score -= 1
    if CONTRA_RE.search(t):   score += 1
    # Bucketize
    if score <= 0: level = "low"
    elif score >= 3: level = "high"
    else: level = "medium"
    # crude confidence: clamp to [0.4, 0.9] based on distance from 2
    dist = abs(score-2)
    conf = 0.5 + min(0.4, 0.2*dist)
    return level, round(conf, 2)

# ======== LLM PROMPT (STRICT JSON) ========
# IMPORTANT: Double braces {{ }} to escape literal JSON in PromptTemplate.
TEMPLATE = """You are extracting structured metadata from an Architecture Decision Record (ADR).

Inputs:
- TOPIC_BUNDLE: Context + Decision Drivers + Considered Options (merged)
- HUMAN_DECISION_BUNDLE: Decision and/or Rationale

Return ONE JSON object with EXACTLY these keys (no extra keys, no prose):

{{
  "concerns": [string],                  // e.g., ["availability","security"]
  "layer": null or string,               // one of: infra|platform|backend|data|api|frontend|cross-cutting
  "family": null or string,              // database_choice|deployment_strategy|auth_method|caching|api_style|messaging|cloud_provider|framework_library|documentation_process
  "pattern_or_tactic": null or string,   // e.g., circuit breaker|event sourcing|saga|CQRS|bulkhead|repository|adapter
  "risk_flags": [string],                // e.g., ["data_loss","prod_outage","compliance"]
  "forces_documented": boolean,          // is a Forces/Context section explicit?
  "alternatives_listed": boolean,        // are options enumerated?
  "consequences_listed": boolean,        // are consequences explicit?
  "decision_type": null or string,       // structural|technology|behavior_integration|cross_cutting|process
  "scope": null or string,               // system_wide|bounded_context|component_local
  "lifecycle_stage": null or string,     // inception|design|implementation|evolution
  "conflicting_drivers_count": integer,  // trade-off pairs present (see rules)
  "ambiguity_level": null or string      // low|medium|high
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

# ======== LOAD ========
df = pd.read_csv(INPUT_CSV)
for col in ["topic", "human_decision"]:
    if col not in df.columns:
        raise ValueError("CSV must contain 'topic' and 'human_decision' columns.")

prompt = PromptTemplate(
    input_variables=["row_id", "topic_text", "human_decision_text"],
    template=TEMPLATE
)
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, max_retries=6, timeout=120)
parser = StrOutputParser()
chain = prompt | llm | parser

# ======== OUTPUT COLS ========
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
    # optional helper confidences (from post-processor)
    "ambiguity_confidence"
]
for c in TARGET_COLS:
    if c not in df.columns:
        df[c] = None

# ======== RUN ========
parsed, failed = 0, 0
for idx, row in df.iterrows():
    row_id = idx + 1
    topic_text = safe_get(row, "topic")
    human_decision_text = safe_get(row, "human_decision")
    bundle_text = f"{topic_text}\n\n{human_decision_text}"

    resp = chain.invoke({
        "row_id": row_id,
        "topic_text": topic_text,
        "human_decision_text": human_decision_text
    })
    data = extract_json(resp)

    # Retry once if needed
    if data is None:
        retry_prompt = PromptTemplate(
            input_variables=prompt.input_variables,
            template=prompt.template + "\n\nIMPORTANT: Retry and output STRICT JSON ONLY, no commentary."
        )
        retry_chain = retry_prompt | llm | parser
        resp = retry_chain.invoke({
            "row_id": row_id,
            "topic_text": topic_text,
            "human_decision_text": human_decision_text
        })
        data = extract_json(resp)

    if data is None:
        failed += 1
        continue

    # -------- POST-PROCESS / ENFORCE RULES --------
    # Concerns: re-detect deterministically and union with LLM
    llm_concerns = normalize_list_strings(data.get("concerns") or [])
    det_concerns = detect_concerns(bundle_text)
    concerns = sorted(set([c for c in llm_concerns if c in CONCERNS.keys()]) | set(det_concerns))

    # Layer
    layer = normalize_enum(data.get("layer"), LAYER_ENUM)

    # Family
    family = data.get("family")
    family = family if (family and family in FAMILY_ENUM) else None

    # Pattern/tactic (prefer deterministic)
    pat = find_patterns(bundle_text) or data.get("pattern_or_tactic")
    pattern_or_tactic = None
    if pat:
        for p in PATTERNS:
            if p.lower() == str(pat).lower():
                pattern_or_tactic = p
                break

    # Risk flags
    risk_flags = whitelist_risk_flags(data.get("risk_flags") or [])

    # Forces / alternatives / consequences: booleans, default False
    def as_bool(x):
        return bool(x) if isinstance(x, bool) else False
    forces_documented   = as_bool(data.get("forces_documented"))
    alternatives_listed = as_bool(data.get("alternatives_listed"))
    consequences_listed = as_bool(data.get("consequences_listed"))

    # Decision type (prefer deterministic priority classifier)
    dt_llm = normalize_enum(data.get("decision_type"), DECISION_TYPE_ENUM)
    dt_det = classify_decision_type(bundle_text)
    decision_type = dt_det or dt_llm

    # Scope (prefer broader scope when conflict)
    sc_llm = normalize_enum(data.get("scope"), SCOPE_ENUM)
    sc_det = classify_scope(bundle_text)
    scope_order = {"system_wide":3, "bounded_context":2, "component_local":1, None:0}
    scope = sc_llm if scope_order.get(sc_llm,0) >= scope_order.get(sc_det,0) else sc_det

    # Lifecycle stage
    lc_llm = normalize_enum(data.get("lifecycle_stage"), LIFECYCLE_ENUM)
    lc_det = classify_lifecycle(bundle_text)
    lifecycle_stage = lc_llm or lc_det

    # Conflicting drivers (deterministic override)
    conflicting_drivers_count = count_conflicting_drivers(bundle_text)

    # Ambiguity level (deterministic with confidence)
    ambiguity_level, ambiguity_conf = score_ambiguity(bundle_text)

    # -------- WRITE RESULTS --------
    df.at[idx, "concerns"]                  = json.dumps(concerns)
    df.at[idx, "layer"]                     = layer
    df.at[idx, "family"]                    = family
    df.at[idx, "pattern_or_tactic"]         = pattern_or_tactic
    df.at[idx, "risk_flags"]                = json.dumps(risk_flags)
    df.at[idx, "forces_documented"]         = forces_documented
    df.at[idx, "alternatives_listed"]       = alternatives_listed
    df.at[idx, "consequences_listed"]       = consequences_listed
    df.at[idx, "decision_type"]             = decision_type
    df.at[idx, "scope"]                     = scope
    df.at[idx, "lifecycle_stage"]           = lifecycle_stage
    df.at[idx, "conflicting_drivers_count"] = int(conflicting_drivers_count)
    df.at[idx, "ambiguity_level"]           = ambiguity_level
    df.at[idx, "ambiguity_confidence"]      = ambiguity_conf

    parsed += 1

# ======== SAVE ========
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Parsed: {parsed}, Failed: {failed}. Saved to: {OUTPUT_CSV}")


