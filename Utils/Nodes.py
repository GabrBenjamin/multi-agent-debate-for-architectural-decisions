from langchain_core.messages import HumanMessage
from langgraph.graph import END
import json
from Utils.Config import config
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import List  # Import List from typing
from langchain_core.messages import BaseMessage  # Import BaseMessage
from Utils.Agents import Debater, Moderator, Judge  # Import agent classes

# Define DebateState TypedDict
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
    debate_answer: str
    reason: str
    
# Define node functions


def affirmative_opening(state: DebateState):
    """
    Affirmative debater presents the opening statement based on the debate topic.
    """
    # Use the debate topic as the prompt
    prompt = config['affirmative_prompt'].replace('##debate_topic##', state['topic'])
    state['positive_debater'].add_message(HumanMessage(content=prompt))
    positive_opening = state['positive_debater'].generate_response()
    print(f"{state['positive_debater'].name}:\n{positive_opening.content}\n{'-' * 50}\n")

    # Add to overall messages
    state['messages'].append(positive_opening)
    # Store the affirmative answer
    state['aff_ans'] = positive_opening.content
    return state


def negative_opening(state: DebateState):
    """
    Negative debater presents the opening statement.
    """
    # Use the negative prompt
    prompt = config['negative_prompt'].replace('##aff_ans##', state['aff_ans'])
    state['negative_debater'].add_message(HumanMessage(content=prompt))
    negative_opening = state['negative_debater'].generate_response()
    print(f"{state['negative_debater'].name}:\n{negative_opening.content}\n{'-' * 50}\n")

    # Add to messages
    state['messages'].append(negative_opening)
    # Store negative answer
    state['neg_ans'] = negative_opening.content
    return state

def positive_debater_turn(state: DebateState):
    """
    Positive debater responds to negative debater.
    """
    prompt = config['debate_prompt'].replace('##oppo_ans##', state['neg_ans'])
    state['positive_debater'].add_message(HumanMessage(content=prompt))
    positive_turn = state['positive_debater'].generate_response()
    print(f"{state['positive_debater'].name}:\n{positive_turn.content}\n{'-' * 50}\n")
    state['messages'].append(positive_turn)

    # Update affirmative answer
    state['aff_ans'] = positive_turn.content


    return state

def negative_debater_turn(state: DebateState):
    """
    Negative debater responds to positive debater.
    """
    prompt = config['debate_prompt'].replace('##oppo_ans##', state['aff_ans'])
    state['negative_debater'].add_message(HumanMessage(content=prompt))
    negative_turn = state['negative_debater'].generate_response()
    print(f"{state['negative_debater'].name}:\n{negative_turn.content}\n{'-' * 50}\n")
    state['messages'].append(negative_turn)

    # Update negative answer
    state['neg_ans'] = negative_turn.content


    return state

def moderator_evaluation(state: DebateState):
    """
    Moderator evaluates the debate after the round.
    """
    prompt = config['moderator_prompt']
    prompt = prompt.replace('##round##', str(state['round_number']))
    prompt = prompt.replace('##aff_ans##', state['aff_ans'])
    prompt = prompt.replace('##neg_ans##', state['neg_ans'])
    state['moderator'].add_message(HumanMessage(content=prompt))
    evaluation = state['moderator'].generate_response()
    print(f"{state['moderator'].name}:\n{evaluation.content}\n{'-' * 50}\n")
    state['messages'].append(evaluation)

    # Parse the evaluation to decide whether to continue
    try:
        import json
        eval_dict = json.loads(evaluation.content)
        preference = eval_dict.get("Whether there is a preference", "").lower()
        if preference == "yes":
            state['debate_over'] = True
            state['supported_side'] = eval_dict.get("Supported Side", "")
            state['debate_answer'] = eval_dict.get("debate_answer", "")
            state['reason'] = eval_dict.get("Reason", "")
            print(f"The debate has concluded. The moderator supports the {state['supported_side']} side.")
        else:
            state['round_number'] += 1
    except json.JSONDecodeError:
        print("Moderator's evaluation is not in the correct JSON format. Continuing debate.")
        state['round_number'] += 1
    return state

def judge_verdict(state: DebateState):
    """
    Judge provides the final verdict.
    """
    # First prompt
    prompt1 = config['judge_prompt_last1']
    prompt1 = prompt1.replace('##aff_ans##', state['aff_ans'])
    prompt1 = prompt1.replace('##neg_ans##', state['neg_ans'])
    state['judge'].add_message(HumanMessage(content=prompt1))
    candidates = state['judge'].generate_response()
    print(f"{state['judge'].name}:\n{candidates.content}\n{'-' * 50}\n")

    # Second prompt
    prompt2 = config['judge_prompt_last2']
    prompt2 = prompt2.replace('##debate_topic##', state['topic'])
    state['judge'].add_message(HumanMessage(content=prompt2))
    verdict = state['judge'].generate_response()
    print(f"{state['judge'].name}:\n{verdict.content}\n{'-' * 50}\n")

    # Parse the verdict
    try:
        import json
        verdict_dict = json.loads(verdict.content)
        state['debate_answer'] = verdict_dict.get("debate_answer", "")
        state['reason'] = verdict_dict.get("Reason", "")
        state['debate_over'] = True
    except json.JSONDecodeError:
        print("Judge's verdict is not in the correct JSON format.")
        state['debate_over'] = True
    return state

def check_debate_over(state: DebateState):
    """
    Check if the debate is over.
    """
    if state['debate_over']:
        return END
    elif state['round_number'] > state['max_rounds']:
        return 'judge_verdict'
    else:
        return 'positive_debater_turn'