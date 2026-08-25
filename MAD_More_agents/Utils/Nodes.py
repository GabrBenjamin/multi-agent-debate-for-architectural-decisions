import json
from typing import List, Optional, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from typing_extensions import TypedDict

from Utils.Config import config
from Utils.Agents import Debater, Moderator, Judge


class DebateState(TypedDict):
    round_number: int
    max_rounds: int
    topic: str
    positive_debater: Debater
    negative_debater: Debater
    third_part: Debater
    moderator: Moderator
    judge: Judge
    debate_over: bool
    messages: List[BaseMessage]
    aff_ans: str
    neg_ans: str
    third_answ: str
    supported_side: str
    debate_answer: str
    reason: str


MODERATOR_JSON_SCHEMA_HINT = (
    "Return a JSON object with the keys: \"Whether there is a preference\" (string, either \"yes\" or \"no\"), "
    "\"Supported Side\" (string naming the stance), \"debate_answer\" (string with the summary), "
    "and \"Reason\" (string explaining the rationale)."
)

JUDGE_JSON_SCHEMA_HINT = (
    "Return a JSON object with the keys: \"debate_answer\" (string describing the decision) "
    "and \"Reason\" (string explaining the decision)."
)


def _coerce_json_text(raw: str) -> str:
    """Strip code fences and isolate the outermost JSON object for parsing."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if "\n" in cleaned:
            first_line, remainder = cleaned.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                cleaned = remainder
            else:
                cleaned = remainder if remainder.strip() else first_line
        cleaned = cleaned.rstrip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        cleaned = cleaned[start:end + 1]
    return cleaned.strip()


def parse_or_repair_json(
    content: str,
    schema_hint: str,
    model_name: str,
    context: str,
) -> Tuple[Optional[dict], str]:
    """Parse JSON content, invoking an LLM to repair malformed payloads when needed."""
    normalized = _coerce_json_text(content)
    try:
        return json.loads(normalized), normalized
    except json.JSONDecodeError:
        print(f"{context} is not in the correct JSON format. Attempting to reformat.")

    try:
        llm = ChatOpenAI(model=model_name, temperature=0, response_format={"type": "json_object"})
    except TypeError:
        llm = ChatOpenAI(model=model_name, temperature=0)

    system_message = (
        "You convert text into a single JSON object that strictly matches the provided schema description. "
        "Return only raw JSON with double-quoted keys and values. Do not include markdown, commentary, or extra keys."
    )
    human_content = (
        "Schema description:\n"
        f"{schema_hint}\n\n"
        "Original content to fix:\n"
        f"{content}"
    )
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_content),
    ]

    try:
        reformatted = llm.invoke(messages).content
    except Exception as exc:
        print(f"Failed to call LLM to repair JSON for {context}: {exc}")
        return None, content

    normalized = _coerce_json_text(reformatted)
    try:
        return json.loads(normalized), normalized
    except json.JSONDecodeError:
        print(f"Reformatted {context.lower()} is still not valid JSON.")
        return None, content


def affirmative_opening(state: DebateState) -> DebateState:
    prompt = config['affirmative_prompt'].replace('##debate_topic##', state['topic'])
    state['positive_debater'].add_message(HumanMessage(content=prompt))
    response = state['positive_debater'].generate_response()
    print(f"{state['positive_debater'].name}:\n{response.content}\n{'-' * 50}\n")
    state['messages'].append(response)
    state['aff_ans'] = response.content
    return state


def negative_opening(state: DebateState) -> DebateState:
    prompt = config['negative_prompt'].replace('##aff_ans##', state['aff_ans'])
    state['negative_debater'].add_message(HumanMessage(content=prompt))
    response = state['negative_debater'].generate_response()
    print(f"{state['negative_debater'].name}:\n{response.content}\n{'-' * 50}\n")
    state['messages'].append(response)
    state['neg_ans'] = response.content
    return state


def third_part_opening(state: DebateState) -> DebateState:
    prompt = config['third_part_prompt']
    prompt = prompt.replace('##aff_ans##', state['aff_ans'])
    prompt = prompt.replace('##neg_ans##', state['neg_ans'])
    state['third_part'].add_message(HumanMessage(content=prompt))
    response = state['third_part'].generate_response()
    print(f"{state['third_part'].name}:\n{response.content}\n{'-' * 50}\n")
    state['messages'].append(response)
    state['third_answ'] = response.content
    return state


def positive_debater_turn(state: DebateState) -> DebateState:
    prompt = config['debate_prompt']
    prompt = prompt.replace('##oppo_ans##', state['neg_ans'])
    prompt = prompt.replace('##oppo_ans2##', state['third_answ'])
    prompt = prompt.replace('##Your_answer##', state['aff_ans'])
    state['positive_debater'].add_message(HumanMessage(content=prompt))
    response = state['positive_debater'].generate_response()
    print(f"{state['positive_debater'].name}:\n{response.content}\n{'-' * 50}\n")
    state['messages'].append(response)
    state['aff_ans'] = response.content
    return state


def negative_debater_turn(state: DebateState) -> DebateState:
    prompt = config['debate_prompt']
    prompt = prompt.replace('##oppo_ans##', state['aff_ans'])
    prompt = prompt.replace('##oppo_ans2##', state['third_answ'])
    prompt = prompt.replace('##Your_answer##', state['neg_ans'])
    state['negative_debater'].add_message(HumanMessage(content=prompt))
    response = state['negative_debater'].generate_response()
    print(f"{state['negative_debater'].name}:\n{response.content}\n{'-' * 50}\n")
    state['messages'].append(response)
    state['neg_ans'] = response.content
    return state


def third_debater_turn(state: DebateState) -> DebateState:
    prompt = config['debate_prompt']
    prompt = prompt.replace('##oppo_ans##', state['neg_ans'])
    prompt = prompt.replace('##oppo_ans2##', state['aff_ans'])
    prompt = prompt.replace('##Your_answer##', state['third_answ'])
    state['third_part'].add_message(HumanMessage(content=prompt))
    response = state['third_part'].generate_response()
    print(f"{state['third_part'].name}:\n{response.content}\n{'-' * 50}\n")
    state['messages'].append(response)
    state['third_answ'] = response.content
    return state


def moderator_evaluation(state: DebateState) -> DebateState:
    prompt = config['moderator_prompt']
    prompt = prompt.replace('##round##', str(max(state['round_number'], 1)))
    prompt = prompt.replace('##aff_ans##', state['aff_ans'])
    prompt = prompt.replace('##neg_ans##', state['neg_ans'])
    prompt = prompt.replace('##third_ans##', state['third_answ'])
    state['moderator'].add_message(HumanMessage(content=prompt))
    response = state['moderator'].generate_response()
    print(f"{state['moderator'].name}:\n{response.content}\n{'-' * 50}\n")
    state['messages'].append(response)
    parsed, normalized = parse_or_repair_json(
        response.content,
        MODERATOR_JSON_SCHEMA_HINT,
        state['moderator'].llm_name,
        "Moderator's evaluation",
    )
    if not isinstance(parsed, dict):
        state['round_number'] += 1
        return state
    response.content = normalized
    preference = str(parsed.get("Whether there is a preference", "")).strip().lower()
    if preference == "yes":
        state['debate_over'] = True
        state['supported_side'] = str(parsed.get("Supported Side", ""))
        state['debate_answer'] = str(parsed.get("debate_answer", ""))
        state['reason'] = str(parsed.get("Reason", ""))
        print(f"The debate has concluded. The moderator supports the {state['supported_side']} side.")
    else:
        state['round_number'] += 1
    return state


def judge_verdict(state: DebateState) -> DebateState:
    prompt1 = config['judge_prompt_last1']
    prompt1 = prompt1.replace('##aff_ans##', state['aff_ans'])
    prompt1 = prompt1.replace('##neg_ans##', state['neg_ans'])
    prompt1 = prompt1.replace('##third_ans##', state['third_answ'])
    state['judge'].add_message(HumanMessage(content=prompt1))
    candidates = state['judge'].generate_response()
    print(f"{state['judge'].name}:\n{candidates.content}\n{'-' * 50}\n")

    prompt2 = config['judge_prompt_last2']
    prompt2 = prompt2.replace('##debate_topic##', state['topic'])
    state['judge'].add_message(HumanMessage(content=prompt2))
    verdict = state['judge'].generate_response()
    print(f"{state['judge'].name}:\n{verdict.content}\n{'-' * 50}\n")
    parsed, normalized = parse_or_repair_json(
        verdict.content,
        JUDGE_JSON_SCHEMA_HINT,
        state['judge'].llm_name,
        "Judge's verdict",
    )
    if isinstance(parsed, dict):
        verdict.content = normalized
        state['debate_answer'] = str(parsed.get("debate_answer", ""))
        state['reason'] = str(parsed.get("Reason", ""))
    else:
        print("Judge's verdict is not valid JSON.")
    state['debate_over'] = True
    return state


def check_debate_over(state: DebateState) -> str:
    if state['debate_over']:
        return END
    if state['round_number'] >= state['max_rounds']:
        return 'judge_verdict'
    return 'positive_debater_turn'
