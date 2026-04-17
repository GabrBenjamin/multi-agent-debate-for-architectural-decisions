import os
from Utils.Agents import Debater, Moderator, Judge, Agent
from Utils.Config import config
from Utils.State_graph import build_state_graph
from langchain_openai import ChatOpenAI

# Set OpenAI API key for this process
os.environ["OPENAI_API_KEY"] = "REDACTED_API_KEY"




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
        llm_name="gpt-4o",
        stance="Affirmative",
        description="Supports the argument.",
    )
    

    negative_debater = Debater(
        name="NegativeSide",
        role="debater",
        llm_name="gpt-4o",
        stance="Negative",
        description="Opposes the argument.",
    )

    # Create moderator and judge with valid names
    moderator = Moderator(
        name="Moderator",
        role="moderator",
        llm_name="gpt-4o",
        description="Oversees the debate, ensures rules are followed, and evaluates progress.",
    )

    judge = Judge(
        name="Judge",
        role="judge",
        llm_name="gpt-4o",
        description="Provides the final verdict based on the arguments presented.",
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

def main_debate(context, retriever_info):
    llm = ChatOpenAI(model='gpt-4o')
    init = context["context_considered_drivers"]
    promptone = f"""You are an expert in software architeture. Collect Scenarios[eliciting system usage scenarios and collecting requirements, constraints, and environmental details ] for the following : {init} """
    ri=llm.invoke(promptone)
    # print(ri)
    promptsecond = f"""You are an expert in software. In addition to the scenarios {ri},  the attribute-based require ments, constraints, and environment of the system must be identified. As a reminder here is the context,decision drivers and considered options {init} """
    ro=llm.invoke(promptsecond)
    # print(ro)
    promptthird = f"""Evaluate and list pros and cons from each of the options from :{init} .  based on the scenarios and requiriments from {ri} and {ro} """
    evaluatedscenarios = llm.invoke(promptthird)
    # print(evaluatedscenarios)
    topic = f"""Decide the best proposed solution from {evaluatedscenarios}"""  
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
        'reason': '',
        'retriever_mode': 'continuous', 
        'retriever_info': retriever_info,
    }
    # retriever modes:
    ### 'opening_only' - each debater receives repo information only once at their respective opening
    ### 'coninuous' - every time a debater responds, it receives new repository information based on the previous messages

    # Build the state graph
    debate_graph = build_state_graph()
    config_dict = {"configurable": {"thread_id": "1"}}
    final_state = debate_graph.invoke(state, config_dict)
    
    message_history_str = '\n------------\n'.join(
        str(msg.content) for msg in state['messages']
    ) if state['messages'] else "No messages recorded"
    
    result = {
        'context_considered_drivers': context['context_considered_drivers'], 
        'other_sections' : context['other_sections'],
        'supported_side': final_state.get('supported_side', ''),
        'debate_answer': final_state.get('debate_answer', ''),
        'reason': final_state.get('reason', ''),
        'message_history': message_history_str,
        'extraction_status': 'success'
    }
    return result


if __name__ == "__main__":
    main_debate()
