import os
import csv
from Utils.Agents import Debater, Moderator, Judge, Agent
from Utils.Config import config
from Utils.State_graph import build_state_graph
from langchain_openai import ChatOpenAI
from Utils.Env import set_env
import pandas as pd

set_env("OPENAI_API_KEY")

# Initialize agents
def initialize_agents(config, topic):
    # Create debaters with valid names
    Agent.agents_registry.clear()
    positive_debater = Debater(
        name="AffirmativeSide",
        role="debater",
        llm_name="gpt-4o",
        stance="Affirmative",
        description="Supports the argument."
    )

    negative_debater = Debater(
        name="NegativeSide",
        role="debater",
        llm_name="gpt-4o",
        stance="Negative",
        description="Opposes the argument."
    )

    # Create moderator and judge with valid names
    moderator = Moderator(
        name="Moderator",
        role="moderator",
        llm_name="gpt-4o",
        description="Oversees the debate, ensures rules are followed, and evaluates progress."
    )

    judge = Judge(
        name="Judge",
        role="judge",
        llm_name="gpt-4o",
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
    # Read the dataset of topics
    topics_df = pd.read_csv('C:/Users/gabri/Downloads/sample_with_all_three_extract.csv')  # Ensure this CSV has a column named 'topic'

    # Prepare a list to collect results
    results = []

    # Loop over each topic in the dataset
    for idx, row in topics_df.iterrows():
        topic = f" Select the best option for the following:: {row['context_considered_drivers']}"
        config['debate_topic'] = topic

        # Initialize agents with the current topic
        positive_debater, negative_debater, moderator, judge = initialize_agents(config, topic)

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
            'messages': [],  # This will be populated by the debate graph
            'aff_ans': '',
            'neg_ans': '',
            'supported_side': '',
            'debate_answer': '',
            'reason': ''
        }

        # Build the state graph
        debate_graph = build_state_graph()
        config_dict = {"configurable": {"thread_id": str(idx)}}

        # Run the debate graph
        final_state = debate_graph.invoke(state, config_dict)

        # Extract all messages from the state['messages'] list
        message_history_str = '\n'.join(
            str(msg.content) for msg in state['messages']
        ) if state['messages'] else "No messages recorded"

        # Extract the output you want to save
        result = {
            'topic': row['context_considered_drivers'],  # Save the original topic for reference
            'human_decision' : row['other_sections'],
            'supported_side': final_state.get('supported_side', ''),
            'debate_answer': final_state.get('debate_answer', ''),
            'reason': final_state.get('reason', ''),
            'message_history': message_history_str
        }

        # Append the result to the list
        results.append(result)

    # Save all results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('debateC_1.csv', index=False)


if __name__ == "__main__":
    main()
