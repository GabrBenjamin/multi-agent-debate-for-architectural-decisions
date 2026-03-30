import os
import csv
from Utils.Agents import Debater, Moderator, Judge, Agent
from Utils.Config import config
from Utils.State_graph import build_state_graph
from langchain_openai import ChatOpenAI
from Utils.Env import set_env
import pandas as pd
from Scorer import compute_best_option
import json
set_env("OPENAI_API_KEY")


def prepare_struct_for_scoring(struct_obj):
    struct_obj = dict(struct_obj or {})

    struct_obj["impact_semantics"] = "native_value"
    struct_obj["require_complete_impacts"] = True
    struct_obj["missing_impact_policy"] = "neutral"
    struct_obj["include_invalid_totals"] = False

    if "risk_profile" not in struct_obj:
        flags = struct_obj.get("risk_flags", []) or []
        flags_lower = [str(x).strip().lower() for x in flags]

        if any("conservative" in x for x in flags_lower):
            struct_obj["risk_profile"] = "conservative"
        elif any("bold" in x for x in flags_lower):
            struct_obj["risk_profile"] = "bold"
        else:
            struct_obj["risk_profile"] = "balanced"

    return struct_obj
# ---------------------------
# Initialize agents (unchanged)
# ---------------------------
def initialize_agents(config_obj, topic):
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

    player_meta_prompt = config_obj['player_meta_prompt'].replace('##debate_topic##', topic)
    moderator_meta_prompt = config_obj['moderator_meta_prompt'].replace('##debate_topic##', topic)

    positive_debater.set_system_message(player_meta_prompt)
    negative_debater.set_system_message(player_meta_prompt)
    moderator.set_system_message(moderator_meta_prompt)
    judge.set_system_message(moderator_meta_prompt)

    return positive_debater, negative_debater, moderator, judge

# ---------------------------
# Main
# ---------------------------
def main():
    # Input CSV must have: context_considered_drivers, other_sections
    df = pd.read_csv('adrs_final_sample_58.csv')

    results = []

    for idx, row in df.iterrows():
        # Your debate prompt (unchanged except your earlier tweak)
        topic = (
            f"For the following context, drivers, and considered options:\n{row['context_considered_drivers']}\n\n"
            "Discuss ONLY the following:\n"
            "1) List the drivers, state whether each is minimize or maximize, and assign a priority (H/M/L).\n"
            "2) For each option, give its impact on each driver as L/M/H with a brief reason.\n"
            "3) Note any hard constraints (min/max thresholds) that apply to drivers.\n"
            "4) State the risk profile you recommend for this decision: choose one — Conservative (risk-averse), Balanced, or Bold (opportunity-seeking).\n"
            "End with a one-line summary. Do not output JSON."
        )
        config['debate_topic'] = topic

        # Initialize agents & state
        pos, neg, mod, judge = initialize_agents(config, topic)
        state = {
            'round_number': 0,
            'max_rounds': 3,
            'topic': topic,
            'positive_debater': pos,
            'negative_debater': neg,
            'moderator': mod,
            'judge': judge,
            'debate_over': False,
            'messages': [],
            'aff_ans': '',
            'neg_ans': '',
            'supported_side': '',
            # IMPORTANT: We now assume the debate graph will set 'debate_answer' to the FINAL JSON your scorer expects.
            # That JSON should contain: drivers (with orientation & priority), options, impacts (L/M/H), constraints, risk_flags.
            'debate_answer': '',   # will be a JSON string or dict
            'reason': '',
            # Optional: if your judge also returns the raw winning text separately
            'winning_text': ''
        }

        # Run debate workflow
        graph = build_state_graph()
        cfg = {"configurable": {"thread_id": str(idx)}}
        final_state = graph.invoke(state, cfg)

        # --- Get the structured JSON produced by the debate/judge ---
        debate_struct = final_state.get('debate_answer', {})

        # Accept both dicts and JSON strings
        if isinstance(debate_struct, str):
            try:
                debate_struct = json.loads(debate_struct)
            except Exception:
                debate_struct = {}

        # --- Call your external fuzzy scorer on the already-structured JSON ---
        # Signature expected: compute_best_option(final_struct: dict) -> dict with at least {"winner": "..."}
        scoring = {}
        try:
            prepared_struct = prepare_struct_for_scoring(debate_struct)
            scoring = compute_best_option(prepared_struct)
        except Exception as e:
            scoring = {"winner": None, "error": str(e)}

        computed_winner = scoring.get("winner")

        # What we save as the debate answer in the dataset
        debate_answer_for_dataset = (
            computed_winner if computed_winner
            else "GuideArch-fuzzy winner: None"
        )

        # Optional: collect full message history
        message_history_str = '\n'.join(
            str(msg.content) for msg in state.get('messages', [])
        ) if state.get('messages') else ""

        results.append({
            "topic": row["context_considered_drivers"],
            "human_decision": row.get("other_sections", ""),
            "supported_side": final_state.get("supported_side", ""),
            # Save the **scorer’s decision** as debate_answer, per your ask:
            "debate_answer": debate_answer_for_dataset,
            "reason": final_state.get("reason", ""),
            "message_history": message_history_str,
            # Optional artifacts for debugging:
            "debate_struct_json": json.dumps(debate_struct, ensure_ascii=False),
            "scoring_json": json.dumps(scoring, ensure_ascii=False),
        })

    pd.DataFrame(results).to_csv('debate_guideArchS_cleaned_scored.csv', index=False)
    print("Saved to debate_guideArchS_cleaned_scored.csv")

if __name__ == "__main__":
    main()
