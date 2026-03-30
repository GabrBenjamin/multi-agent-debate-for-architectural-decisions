config = {
    "debate_topic": "",
    "base_answer": "",
    "debate_answer": "",

    # Debaters stay FREE-FORM (no JSON). Keep as-is.
    "player_meta_prompt": (
        "You are a debater. Hello and welcome to the debate. It's not necessary to fully agree with each other's "
        "perspectives, as our objective is to find the correct answer.\n"
        "The debate topic is stated as follows:\n##debate_topic##.\n"
        "Debate in free text format."
    ),
    "affirmative_prompt": "##debate_topic##",
    "negative_prompt": "##aff_ans##\n\nYou disagree with my answer. Provide your answer and reasons.",
    "debate_prompt": (
        "##oppo_ans##\n\nDo you agree with my perspective? Additionally, you have access to the full debate history "
        "for extra context. Do not use JSON or structured formats in your response."
    ),

    # Moderator: STRICT JSON with GuideArch 'final' schema (domain/compose + triangle-or-label)
    "moderator_meta_prompt": (
        "You are a moderator. There will be two debaters involved in a debate. They will present their answers and "
        "discuss their perspectives on the following topic: \"##debate_topic##\".\n"
        "At the end of each round, you will evaluate answers and determine whether there is a clear preference. "
        "OUTPUT STRICT JSON ONLY (no code fences, no extra text). Use this format:\n"
        "{"
        "\"Whether there is a preference\": \"Yes\" or \"No\", "
        "\"Supported Side\": \"Affirmative\" or \"Negative\", "
        "\"Reason\": \"\", "
        "\"debate_answer\": \"\", "
        "\"final\": {"
            "\"drivers\": [ {\"name\":\"<driver>\", \"orientation\":\"min|max\", \"priority\":\"H|M|L\", "
                           "\"domain\":{\"min\":<number>, \"max\":<number>}, "
                           "\"compose\":\"sum|product|min|max\", \"unit\":\"<optional>\"} ], "
            "\"options\": [ {\"name\":\"<option>\"} ], "
            "\"impacts\": [ "
                "{"
                    "\"option\":\"<option>\", \"driver\":\"<driver>\", \"label\":\"L|M|H\""
                "} "
                "OR "
                "{"
                    "\"option\":\"<option>\", \"driver\":\"<driver>\", \"triangle\":[<opt>,<ant>,<pes>]"
                "} "
            "], "
            "\"constraints\": [ {\"driver\":\"<driver>\", \"type\":\"min|max\", \"value\": <number>} ], "
            "\"risk_flags\": [\"<short_tag>\"], "
            "\"weights\": {\"za\":<0..1>, \"zn\":<0..1>, \"zp\":<0..1>}"
        "}, "
        "\"prob_correct\": <number between 0 and 1>"
        "}\n"
        "Rules:\n"
        "- If \"Whether there is a preference\" is \"No\", you may omit 'Supported Side', 'final', and 'prob_correct'.\n"
        "- If it is \"Yes\", include 'Supported Side' and provide 'final' and 'prob_correct' whenever possible."
    ),

    "moderator_prompt": (
        "Now the ##round## round of debate for both sides has ended.\n\n"
        "Affirmative side arguing:\n##aff_ans##\n\n"
        "Negative side arguing:\n##neg_ans##\n\n"
        "You, as the moderator, will evaluate both sides' answers and determine if there is a clear preference for an "
        "answer candidate. If so, summarize your reasons and provide the final structured output. "
        "OUTPUT STRICT JSON ONLY (no code fences, no extra text). Use this format:\n"
        "{"
        "\"Whether there is a preference\": \"Yes\" or \"No\", "
        "\"Supported Side\": \"Affirmative\" or \"Negative\", "
        "\"Reason\": \"\", "
        "\"debate_answer\": \"\", "
        "\"final\": {"
            "\"drivers\": [ {\"name\":\"<driver>\", \"orientation\":\"min|max\", \"priority\":\"H|M|L\", "
                           "\"domain\":{\"min\":<number>, \"max\":<number>}, "
                           "\"compose\":\"sum|product|min|max\", \"unit\":\"<optional>\"} ], "
            "\"options\": [ {\"name\":\"<option>\"} ], "
            "\"impacts\": [ "
                "{"
                    "\"option\":\"<option>\", \"driver\":\"<driver>\", \"label\":\"L|M|H\""
                "} "
                "OR "
                "{"
                    "\"option\":\"<option>\", \"driver\":\"<driver>\", \"triangle\":[<opt>,<ant>,<pes>]"
                "} "
            "], "
            "\"constraints\": [ {\"driver\":\"<driver>\", \"type\":\"min|max\", \"value\": <number>} ], "
            "\"risk_flags\": [\"<short_tag>\"], "
            "\"weights\": {\"za\":<0..1>, \"zn\":<0..1>, \"zp\":<0..1>}"
        "}, "
        "\"prob_correct\": <number between 0 and 1>"
        "}\n"
        "If there is no clear preference, set \"Whether there is a preference\" to \"No\"."
    ),

    "judge_prompt_last1": (
        "Affirmative side arguing:\n##aff_ans##\n\n"
        "Negative side arguing:\n##neg_ans##\n\n"
        "Now, what answer candidates do we have? Present them briefly (free text, no JSON)."
    ),

    # Judge: STRICT JSON GuideArch 'final'
    "judge_prompt_last2": (
        "Therefore, ##debate_topic##\n"
        "Summarize your reasons and give the final answer you think is correct.\n"
        "OUTPUT STRICT JSON ONLY (no code fences, no extra text) with EXACTLY these keys:\n"
        "{"
          "\"Reason\": \"<brief rationale>\", "
          "\"final\": {"
            "\"drivers\": [ {\"name\":\"<driver>\", \"orientation\":\"min|max\", \"priority\":\"H|M|L\", "
                           "\"domain\":{\"min\":<number>, \"max\":<number>}, "
                           "\"compose\":\"sum|product|min|max\", \"unit\":\"<optional>\"} ], "
            "\"options\": [ {\"name\":\"<option>\"} ], "
            "\"impacts\": [ "
                "{"
                    "\"option\":\"<option>\", \"driver\":\"<driver>\", \"label\":\"L|M|H\""
                "} "
                "OR "
                "{"
                    "\"option\":\"<option>\", \"driver\":\"<driver>\", \"triangle\":[<opt>,<ant>,<pes>]"
                "} "
            "], "
            "\"constraints\": [ {\"driver\":\"<driver>\", \"type\":\"min|max\", \"value\": <number>} ], "
            "\"risk_flags\": [\"<short_tag>\"], "
            "\"weights\": {\"za\":<0..1>, \"zn\":<0..1>, \"zp\":<0..1>}"
          "}, "
          "\"prob_correct\": <number between 0 and 1>"
        "}"
    ),

    # OPTIONAL: post-hoc extractor (only if you decide to parse free text later)
    "extractor_prompt": (
        "SYSTEM:\nExtract ONLY structured fields needed for GuideArch scoring. Output strict JSON only.\n\n"
        "USER:\nWe need a strict JSON for scoring the decision.\n\n"
        "Inputs:\n- DEBATE_TOPIC:\n{debate_topic}\n\n"
        "- TOPIC_BUNDLE (Context + Decision Drivers + Considered Options):\n{topic_blob}\n\n"
        "- WINNING_TEXT (verbatim from the moderator/judge):\n{winning_text}\n\n"
        "Return EXACTLY this JSON schema (no extra keys, no code fences):\n"
        "{\n"
        "  \"drivers\": [ {\"name\":\"<driver>\", \"orientation\":\"min|max\", \"priority\":\"H|M|L\", "
        "                  \"domain\":{\"min\":<number>, \"max\":<number>}, "
        "                  \"compose\":\"sum|product|min|max\", \"unit\":\"<optional>\"} ],\n"
        "  \"options\": [ {\"name\":\"<option>\"} ],\n"
        "  \"impacts\": [ {\"option\":\"<option>\", \"driver\":\"<driver>\", \"label\":\"L|M|H\"} "
        "                 OR {\"option\":\"<option>\", \"driver\":\"<driver>\", \"triangle\":[<opt>,<ant>,<pes>]} ],\n"
        "  \"constraints\": [ {\"driver\":\"<driver>\", \"type\":\"min|max\", \"value\": <number> } ],\n"
        "  \"risk_flags\": [\"<short_tag>\"],\n"
        "  \"weights\": {\"za\":<0..1>, \"zn\":<0..1>, \"zp\":<0..1>}\n"
        "}\n\n"
        "Rules:\n"
        "- Use ONLY drivers/options that appear in TOPIC_BUNDLE or WINNING_TEXT.\n"
        "- Orientation must be explicit per driver (latency/cost → \"min\"; availability/reliability → \"max\").\n"
        "- If an impact is uncertain, prefer a 'triangle' in native units; otherwise use label 'M'.\n"
        "- Constraints apply to drivers; include only if supported by the inputs.\n"
        "- Output STRICT JSON only, nothing else."
    ),

    # OPTIONAL: validator/sanitizer for the extractor output
    "validator_prompt": (
        "SYSTEM:\nValidate and sanitize the candidate JSON to the required GuideArch schema. "
        "Fix only format-level issues; do not change substance beyond the rules. Output strict JSON only.\n\n"
        "USER:\nJSON_CANDIDATE:\n{extractor_json}\n\n"
        "Required schema (no extra keys):\n"
        "{\n"
        "  \"drivers\": [ {\"name\":\"<driver>\", \"orientation\":\"min|max\", \"priority\":\"H|M|L\", "
        "                  \"domain\":{\"min\":<number>, \"max\":<number>}, "
        "                  \"compose\":\"sum|product|min|max\", \"unit\":\"<optional>\"} ],\n"
        "  \"options\": [ {\"name\":\"<option>\"} ],\n"
        "  \"impacts\": [ {\"option\":\"<option>\", \"driver\":\"<driver>\", \"label\":\"L|M|H\"} "
        "                 OR {\"option\":\"<option>\", \"driver\":\"<driver>\", \"triangle\":[<opt>,<ant>,<pes>]} ],\n"
        "  \"constraints\": [ {\"driver\":\"<driver>\", \"type\":\"min|max\", \"value\": <number> } ],\n"
        "  \"risk_flags\": [\"<short_tag>\"],\n"
        "  \"weights\": {\"za\":<0..1>, \"zn\":<0..1>, \"zp\":<0..1>}\n"
        "}\n\n"
        "Sanitization rules:\n"
        "- Remove any impact that references a non-existent driver or option.\n"
        "- Ensure every driver has 'orientation' in {\"min\",\"max\"}; if missing but the name clearly implies it "
        "(latency/cost -> min, availability/reliability -> max), fill it; otherwise DROP that driver and related impacts.\n"
        "- Ensure every driver has 'priority' in {\"H\",\"M\",\"L\"}; if missing, set 'M'.\n"
        "- If 'domain' is missing, infer a plausible domain from units/context if present; otherwise set {\"min\":0, \"max\":1}.\n"
        "- Ensure every impact uses either a valid 'label' in {\"L\",\"M\",\"H\"} OR a numeric 'triangle' of three values.\n"
        "- Ensure 'constraints' only reference existing drivers; drop invalid ones.\n"
        "- Return ONLY the corrected JSON (no comments, no code fences)."
    ),
}


## Mudar o moderador de human pra system prompt
##timestamp no prompt 