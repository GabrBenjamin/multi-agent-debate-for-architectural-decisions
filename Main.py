import os
import getpass
from Utils.Agents import Debater, Moderator, Judge, Agent
from Utils.Config import config
from Utils.State_graph import build_state_graph
from langchain_openai import ChatOpenAI


#from Utils.Env import set_env
#set_env("OPENAI_API_KEY")




# Initialize agents
def initialize_agents(config):
    #agent_models = {
        #"AffirmativeSide": "gpt-3.5-turbo",
        #"NegativeSide": "gpt-3.5-turbo",
        #"Moderator": "gpt-4",
        #"Judge": "gpt-3.5-turbo"
    #}
    Agent.agents_registry.clear()
    topic = config['debate_topic']

    # Create debaters with valid names
    positive_debater = Debater(
        name="AffirmativeSide",
        role="debater",
        llm_name="llama",
        stance="Affirmative",
        description="Supports the argument."
    )

    negative_debater = Debater(
        name="NegativeSide",
        role="debater",
        llm_name="llama",
        stance="Negative",
        description="Opposes the argument."
    )

    # Create moderator and judge with valid names
    moderator = Moderator(
        name="Moderator",
        role="moderator",
        llm_name="llama",
        description="Oversees the debate, ensures rules are followed, and evaluates progress."
    )

    judge = Judge(
        name="Judge",
        role="judge",
        llm_name="llama",
        description="Provides the final verdict based on the arguments presented."
    )

    # Replace placeholders in the prompts
    player_meta_prompt = config['player_meta_prompt'].replace('##debate_topic##', topic)
    moderator_meta_prompt = config['moderator_meta_prompt'].replace('##debate_topic##', topic)

    # Set system messages
    positive_debater.set_system_message(player_meta_prompt)
    negative_debater.set_system_message(player_meta_prompt)
    moderator.set_system_message(moderator_meta_prompt)
    judge.set_system_message(moderator_meta_prompt)  # Assuming judge uses same as moderator

    return positive_debater, negative_debater, moderator, judge

def main():
    # Define the debate topic
    topic = "Propose a solution for the following ::## Context and Problem Statement To deliver BFI's IIIF Universal Viewer auditing platform, custom deliverables must be produced to support the serving of the Universal Viewer, and an underlying API which records audit events and persists them into a database."
    config['debate_topic'] = topic

    # Initialize agents
    positive_debater, negative_debater, moderator, judge = initialize_agents(config)

    # Initialize state
    state = {
        'round_number': 0,
        'max_rounds': 3,
        'topic': topic,
        'positive_debater': positive_debater,
        'negative_debater': negative_debater,
        'moderator': moderator,
        'judge': judge,
        'debate_over': False,
        'messages': [],
        'aff_ans': '',
        'neg_ans': '',
        'supported_side': '',
        'debate_answer': '',
        'reason': ''
    }

    # Build the state graph
    debate_graph = build_state_graph()
    config_dict = {"configurable": {"thread_id": "1"}}
    

    # Run the debate graph
    final_state = debate_graph.invoke(state, config_dict)


if __name__ == "__main__":
    main()