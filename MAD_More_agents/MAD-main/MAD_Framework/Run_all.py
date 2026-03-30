import pandas as pd

from Utils.Agents import Debater, Moderator, Judge, Agent
from Utils.Config import config
from Utils.State_graph import build_state_graph
from Utils.Env import set_env

set_env("OPENAI_API_KEY")


def initialize_agents(config, topic):
    """Create agents configured for the provided debate topic."""
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

    third_part = Debater(
        name="ThirdPart",
        role="debater",
        llm_name="gpt-4o",
        stance="neutral",
        description="Offer new point of view."
    )

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

    player_meta_prompt = config['player_meta_prompt'].replace('##debate_topic##', topic)
    moderator_meta_prompt = config['moderator_meta_prompt'].replace('##debate_topic##', topic)

    positive_debater.set_system_message(player_meta_prompt)
    negative_debater.set_system_message(player_meta_prompt)
    third_part.set_system_message(player_meta_prompt)
    moderator.set_system_message(moderator_meta_prompt)
    judge.set_system_message(moderator_meta_prompt)

    return positive_debater, negative_debater, third_part, moderator, judge


def main():
    topics_df = pd.read_csv('C:/Users/gabri/Downloads/sample_with_all_three_extract.csv')


    results = []

    for idx, row in topics_df.iterrows():
        original_context = row['context_considered_drivers']
        topic = (
            "Propose some solutions, list the pros and cons of each, and select the best one for the following:: "
            f"{original_context}"
        )
        config['debate_topic'] = topic

        (
            positive_debater,
            negative_debater,
            third_part,
            moderator,
            judge,
        ) = initialize_agents(config, topic)

        state = {
            'round_number': 0,
            'max_rounds': 2,
            'topic': topic,
            'positive_debater': positive_debater,
            'negative_debater': negative_debater,
            'third_part': third_part,
            'moderator': moderator,
            'judge': judge,
            'debate_over': False,
            'messages': [],
            'aff_ans': '',
            'neg_ans': '',
            'third_answ': '',
            'supported_side': '',
            'debate_answer': '',
            'reason': ''
        }

        debate_graph = build_state_graph()
        config_dict = {"configurable": {"thread_id": str(idx)}}

        final_state = debate_graph.invoke(state, config_dict)

        state_messages = final_state.get('messages', [])
        if not state_messages:
            print('Warning: No debate messages were recorded in the final state.')

        message_history = []
        for msg in state_messages:
            speaker = getattr(msg, 'name', None) or getattr(msg, 'type', 'Unknown')
            content = getattr(msg, 'content', '')
            message_history.append(f"{speaker}: {content}")

        message_history_str = '\n'.join(message_history) if message_history else 'No messages recorded'

        result = {
            'context_considered_drivers': original_context,
            'full_prompt': topic,
            'supported_side': final_state.get('supported_side', ''),
            'debate_answer': final_state.get('debate_answer', ''),
            'reason': final_state.get('reason', ''),
            'third_answ': final_state.get('third_answ', ''),
            'message_history': message_history_str
        }

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('debate_results_with_history.csv', index=False)


if __name__ == "__main__":
    main()
