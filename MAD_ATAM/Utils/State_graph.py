from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import List  # Import List from typing
from langchain_core.messages import BaseMessage  # Import BaseMessage
from Utils.Nodes import DebateState
from Utils.Nodes import (
    affirmative_opening,
    negative_opening,
    positive_debater_turn,
    negative_debater_turn,
    moderator_evaluation,
    judge_verdict,
    check_debate_over
)


def build_state_graph():
    builder = StateGraph(DebateState)
    #builder.add_node('moderator_introduction', moderator_introduction)
    builder.add_node('affirmative_opening', affirmative_opening)
    builder.add_node('negative_opening', negative_opening)
    builder.add_node('positive_debater_turn', positive_debater_turn)
    builder.add_node('negative_debater_turn', negative_debater_turn)
    builder.add_node('moderator_evaluation', moderator_evaluation)
    builder.add_node('judge_verdict', judge_verdict)

    #builder.add_edge(START, 'moderator_introduction')
    builder.add_edge(START, 'affirmative_opening')
    #builder.add_edge('moderator_introduction', 'affirmative_opening')
    builder.add_edge('affirmative_opening', 'negative_opening')
    builder.add_edge('negative_opening', 'moderator_evaluation')
    builder.add_conditional_edges('moderator_evaluation', check_debate_over, ['positive_debater_turn', 'judge_verdict', END])
    builder.add_edge('positive_debater_turn', 'negative_debater_turn')
    builder.add_edge('negative_debater_turn', 'moderator_evaluation')
    builder.add_edge('judge_verdict', END)

    # Compile the graph with memory
    memory = MemorySaver()
    debate_graph = builder.compile(checkpointer=memory)

    return debate_graph
