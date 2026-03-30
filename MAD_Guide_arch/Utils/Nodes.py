from langchain_core.messages import HumanMessage
from langgraph.graph import END
import json
from Utils.Config import config
from typing_extensions import TypedDict
from typing import List, Optional, Tuple
from langchain_core.messages import BaseMessage
from Utils.Agents import Debater, Moderator, Judge

# -----------------------------
# Debate state
# -----------------------------
class DebateState(TypedDict):
    round_number: int
    max_rounds: int
    topic: str
    positive_debater: 'Debater'
    negative_debater: 'Debater'
    moderator: 'Moderator'
    judge: 'Judge'
    debate_over: bool
    messages: List['BaseMessage']
    aff_ans: str
    neg_ans: str
    supported_side: str
    debate_answer: str            # GuideArch "final" JSON (as a JSON string) when finalized
    reason: str
    winning_text: str

# -----------------------------
# JSON schema hints (UPDATED to GuideArch schema with domain/compose and triangle-or-label)
# -----------------------------
MODERATOR_JSON_SCHEMA_HINT = (
    'Return a JSON object with these keys:\n'
    '  "Whether there is a preference": "Yes" or "No",\n'
    '  "Supported Side": "Affirmative" or "Negative" (required if preference is "Yes"),\n'
    '  "Reason": "<brief rationale>",\n'
    '  "debate_answer": "<short summary or final answer text>",\n'
    '  "final": {\n'
    '     "drivers": [ {"name":"<driver>", "orientation":"min|max", "priority":"H|M|L",\n'
    '                   "domain":{"min":<number>, "max":<number>}, "compose":"sum|product|min|max", "unit":"<optional>"} ],\n'
    '     "options": [ {"name":"<option>"} ],\n'
    '     "impacts": [\n'
    '         {"option":"<option>", "driver":"<driver>", "label":"L|M|H"}\n'
    '         OR {"option":"<option>", "driver":"<driver>", "triangle":[<opt>,<ant>,<pes>]}\n'
    '     ],\n'
    '     "constraints": [ {"driver":"<driver>", "type":"min|max", "value": <number>} ],\n'
    '     "risk_flags": ["<short_tag>"],\n'
    '     "weights": {"za":<0..1>, "zn":<0..1>, "zp":<0..1>}\n'
    '  },\n'
    '  "prob_correct": <number between 0 and 1>\n'
    '\n'
    'Rules:\n'
    '- If "Whether there is a preference" is "No", you may omit "Supported Side", "final", and "prob_correct".\n'
    '- If it is "Yes", you MUST include "Supported Side", and SHOULD include "final" and "prob_correct".\n'
)

JUDGE_JSON_SCHEMA_HINT = (
    'Return a JSON object with EXACTLY these keys: "final", "prob_correct", "Reason".\n'
    '"final" must contain:\n'
    '  "drivers": [ {"name":"<driver>", "orientation":"min|max", "priority":"H|M|L",\n'
    '                "domain":{"min":<number>, "max":<number>}, "compose":"sum|product|min|max", "unit":"<optional>"} ],\n'
    '  "options": [ {"name":"<option>"} ],\n'
    '  "impacts": [ {"option":"<option>", "driver":"<driver>", "label":"L|M|H"} OR {"option":"<option>", "driver":"<driver>", "triangle":[<opt>,<ant>,<pes>]} ],\n'
    '  "constraints": [ {"driver":"<driver>", "type":"min|max", "value": <number>} ],\n'
    '  "risk_flags": ["<short_tag>"],\n'
    '  "weights": {"za":<0..1>, "zn":<0..1>, "zp":<0..1>}\n'
)

# -----------------------------
# JSON helpers (no LLM repair)
# -----------------------------
def _coerce_json_text(raw: str) -> str:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if "\n" in cleaned:
            first_line, remainder = cleaned.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                cleaned = remainder
            else:
                cleaned = remainder if remainder.strip() else first_line
        cleaned = cleaned.rstrip("`").strip()
    # isolate outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        cleaned = cleaned[start:end + 1]
    return cleaned.strip()

def parse_or_repair_json(content: str, schema_hint: str, context: str) -> Tuple[Optional[dict], str]:
    """
    Try to parse JSON (no LLM calls). If it fails, return (None, original_content).
    """
    normalized = _coerce_json_text(content)
    try:
        obj = json.loads(normalized)
        return obj, json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        print(f"{context} is not valid JSON. Skipping auto-repair.")
        return None, content

# -----------------------------
# Debate turns
# -----------------------------
def affirmative_opening(state: DebateState):
    prompt = config['affirmative_prompt'].replace('##debate_topic##', state['topic'])
    state['positive_debater'].add_message(HumanMessage(content=prompt))
    msg = state['positive_debater'].generate_response()
    print(f"{state['positive_debater'].name}:\n{msg.content}\n{'-'*50}\n")

    state['messages'].append(msg)
    state['aff_ans'] = msg.content
    return state

def negative_opening(state: DebateState):
    prompt = config['negative_prompt'].replace('##aff_ans##', state['aff_ans'])
    state['negative_debater'].add_message(HumanMessage(content=prompt))
    msg = state['negative_debater'].generate_response()
    print(f"{state['negative_debater'].name}:\n{msg.content}\n{'-'*50}\n")

    state['messages'].append(msg)
    state['neg_ans'] = msg.content
    return state

def positive_debater_turn(state: DebateState):
    prompt = config['debate_prompt'].replace('##oppo_ans##', state['neg_ans'])
    state['positive_debater'].add_message(HumanMessage(content=prompt))
    msg = state['positive_debater'].generate_response()
    print(f"{state['positive_debater'].name}:\n{msg.content}\n{'-'*50}\n")

    state['messages'].append(msg)
    state['aff_ans'] = msg.content
    return state

def negative_debater_turn(state: DebateState):
    prompt = config['debate_prompt'].replace('##oppo_ans##', state['aff_ans'])
    state['negative_debater'].add_message(HumanMessage(content=prompt))
    msg = state['negative_debater'].generate_response()
    print(f"{state['negative_debater'].name}:\n{msg.content}\n{'-'*50}\n")

    state['messages'].append(msg)
    state['neg_ans'] = msg.content
    return state

def _normalize_pref(val: str) -> str:
    v = str(val).strip().lower()
    if v in {"yes", "y", "true", "1"}:
        return "yes"
    if v in {"no", "n", "false", "0"}:
        return "no"
    return v

# -----------------------------
# Moderator evaluation
# -----------------------------
def moderator_evaluation(state: DebateState):
    """
    Moderator evaluates the debate after the round.
    If there is a clear preference AND a GuideArch 'final' object is provided,
    we end the debate immediately and save the structured payload for the scorer.
    Otherwise, continue to the next round.
    """
    prompt = config['moderator_prompt']
    prompt = prompt.replace('##round##', str(max(state['round_number'], 1)))
    prompt = prompt.replace('##aff_ans##', state['aff_ans'])
    prompt = prompt.replace('##neg_ans##', state['neg_ans'])

    state['moderator'].add_message(HumanMessage(content=prompt))
    evaluation = state['moderator'].generate_response()
    print(f"{state['moderator'].name}:\n{evaluation.content}\n{'-' * 50}\n")
    state['messages'].append(evaluation)

    parsed, normalized = parse_or_repair_json(
        evaluation.content,
        MODERATOR_JSON_SCHEMA_HINT,
        "Moderator's evaluation",
    )
    if not isinstance(parsed, dict):
        print("Moderator's evaluation not valid JSON; continuing to next round.")
        state['round_number'] += 1
        return state

    evaluation.content = normalized
    preference = _normalize_pref(parsed.get("Whether there is a preference", ""))

    if preference == "yes":
        state['supported_side'] = str(parsed.get("Supported Side", ""))
        state['reason'] = str(parsed.get("Reason", ""))

        final_obj = parsed.get("final")
        if isinstance(final_obj, dict) and final_obj.get("drivers") and final_obj.get("options") and final_obj.get("impacts"):
            # Save GuideArch JSON payload as debate_answer (ready for scorer)
            state['debate_answer'] = json.dumps(final_obj, ensure_ascii=False)
            state['debate_over'] = True
            print("Moderator provided final GuideArch JSON. Ending debate now.")
            return state

        # Still end (preference decided) but only with summary text
        state['debate_answer'] = str(parsed.get("debate_answer", ""))
        state['debate_over'] = True
        print(f"The debate has concluded by moderator. Supported side: {state['supported_side']}.")
        return state

    # No preference -> continue
    state['round_number'] += 1
    return state

# -----------------------------
# Judge verdict
# -----------------------------
def judge_verdict(state: DebateState):
    """
    Judge provides the final verdict in GuideArch JSON format when max rounds are reached.
    """
    # Step 1 (candidates) — unchanged (free-form list if your judge does that)
    prompt1 = config['judge_prompt_last1']
    prompt1 = prompt1.replace('##aff_ans##', state['aff_ans'])
    prompt1 = prompt1.replace('##neg_ans##', state['neg_ans'])
    state['judge'].add_message(HumanMessage(content=prompt1))
    candidates = state['judge'].generate_response()
    print(f"{state['judge'].name}:\n{candidates.content}\n{'-' * 50}\n")

    # Step 2 (final GuideArch JSON)
    prompt2 = config['judge_prompt_last2']
    prompt2 = prompt2.replace('##debate_topic##', state['topic'])
    state['judge'].add_message(HumanMessage(content=prompt2))
    verdict = state['judge'].generate_response()
    print(f"{state['judge'].name}:\n{verdict.content}\n{'-' * 50}\n")

    parsed, normalized = parse_or_repair_json(
        verdict.content,
        JUDGE_JSON_SCHEMA_HINT,
        "Judge's verdict",
    )

    # Expect: {"final": {...GuideArch schema...}, "prob_correct": float, "Reason": "..."}
    if isinstance(parsed, dict) and isinstance(parsed.get("final"), dict):
        verdict.content = normalized
        state['debate_answer'] = json.dumps(parsed.get("final"), ensure_ascii=False)
        state['reason'] = str(parsed.get("Reason", ""))
    else:
        print("Judge's verdict is not valid GuideArch JSON; leaving debate_answer unchanged.")

    state['debate_over'] = True
    return state

# -----------------------------
# Control flow
# -----------------------------
def check_debate_over(state: DebateState):
    if state['debate_over']:
        return END
    # hand off to judge if we've hit max rounds
    if state['round_number'] >= state['max_rounds']:
        return 'judge_verdict'
    # continue debate
    return 'positive_debater_turn'
