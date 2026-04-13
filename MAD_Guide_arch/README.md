# MAD GuideArch Debate Pipeline

This repository contains a debate-driven decision analysis workflow built with LangChain and LangGraph. Two LLM debaters argue over an architectural decision, a moderator tries to extract a structured decision model, and a scorer ranks the candidate options using GuideArch-style fuzzy scoring.

This README documents the code in:

- `Main.py`
- `Run_all.py`
- `Scorer.py`
- `Utils/`

It does not attempt to document the notebooks, comparison scripts, or result-analysis files outside that scope.

## Overview

The pipeline has two distinct stages:

1. A debate stage generates arguments and attempts to convert them into a structured JSON decision model.
2. A scoring stage evaluates the structured model and selects the best option.

At a high level, the flow is:

`topic or CSV row -> debaters -> moderator/judge -> structured decision JSON -> scorer -> winning option`

## Repository Scope

### `Main.py`

Runs a single debate for one hardcoded topic. It:

- creates the affirmative debater, negative debater, moderator, and judge
- builds the debate state object
- compiles the LangGraph state machine
- invokes the debate workflow once

This file is best understood as a minimal entry point for a single interactive run.

### `Run_all.py`

Runs the full batch pipeline over a CSV dataset. It:

- reads `adrs_final_sample_58.csv`
- builds one debate topic per row from `context_considered_drivers`
- runs the debate workflow for each row
- extracts the structured decision JSON returned by the moderator or judge
- normalizes that JSON for the scorer
- calls `compute_best_option()` from `Scorer.py`
- writes the final dataset to `debate_guideArchS_cleaned_scored.csv`

This is the main operational script in the project.

### `Scorer.py`

Implements the GuideArch-style fuzzy scorer. It converts the structured debate output into comparable option scores, filters invalid options, and returns a ranked winner.

### `Utils/`

Contains the reusable infrastructure for:

- prompt configuration
- agent abstractions
- environment variable loading
- optional SSH-based Llama access
- debate node logic
- LangGraph wiring

## Architecture

### 1. Debate Agents

The system uses four agent roles:

- `AffirmativeSide`: proposes and defends an answer
- `NegativeSide`: challenges the affirmative answer and proposes an alternative
- `Moderator`: evaluates each round and can end the debate early if there is a clear preference
- `Judge`: produces the final answer when the moderator does not end the debate first

All of these inherit from `Agent` in `Utils/Agents.py`.

### 2. Shared Conversation State

The debate is driven by a typed state object defined in `Utils/Nodes.py`. Important fields include:

- `round_number`
- `max_rounds`
- `topic`
- `messages`
- `aff_ans`
- `neg_ans`
- `supported_side`
- `debate_answer`
- `reason`
- `debate_over`

`debate_answer` is especially important. In the current pipeline it is expected to hold the structured GuideArch JSON payload that will be passed into the scorer.

### 3. LangGraph Workflow

`Utils/State_graph.py` defines the debate graph:

1. `affirmative_opening`
2. `negative_opening`
3. `moderator_evaluation`
4. If needed, alternating rebuttal rounds
5. `judge_verdict` once `max_rounds` is reached

The moderator can terminate the debate early when it returns valid JSON indicating a preference and includes a usable `final` object.

## File-by-File Details

### `Main.py`

`Main.py` is a single-run driver script.

### What it does

- Defines a hardcoded debate topic
- Stores that topic in `config["debate_topic"]`
- Instantiates the four agents through `initialize_agents()`
- Creates the initial graph state
- Compiles and invokes the state graph

### Initialization behavior

`initialize_agents(config)`:

- clears the shared `Agent.agents_registry`
- creates two `Debater` instances, one `Moderator`, and one `Judge`
- injects the topic into the prompt templates from `Utils/Config.py`
- sets each agent's system prompt

### Important note

In the current code, `Main.py` assigns `llm_name="llama"` to all agents, but `Utils/Agents.py` only routes explicitly to:

- `codeqwen` via `ChatTogether`
- `deepseek` via `ChatTogether`
- any other value via `ChatOpenAI`

That means `llama` is not currently connected to `Utils/LLM.py` in this execution path. If you want `Main.py` to run successfully, you will likely need to either:

- change `llm_name` to a valid OpenAI model name, or
- update `Agent.generate_response()` to call `LlamaSSHLLM` when `llm_name == "llama"`

### `Run_all.py`

`Run_all.py` is the batch execution script and the most important script in the repository.

### Expected input

It reads `adrs_final_sample_58.csv` and expects at least these columns:

- `context_considered_drivers`
- `other_sections`

### Per-row processing

For each row, the script:

1. Builds a debate prompt from `context_considered_drivers`
2. Asks the debaters to discuss only:
   - decision drivers
   - whether each driver should be minimized or maximized
   - priority levels
   - per-option impacts
   - hard constraints
   - the recommended risk profile
3. Runs the LangGraph debate
4. Extracts `final_state["debate_answer"]`
5. Parses that value as JSON if needed
6. Calls `prepare_struct_for_scoring()`
7. Sends the normalized structure to `compute_best_option()`

### `prepare_struct_for_scoring()`

This helper modifies the structured debate output before scoring by forcing:

- `impact_semantics = "native_value"`
- `require_complete_impacts = True`
- `missing_impact_policy = "neutral"`
- `include_invalid_totals = False`

It also infers `risk_profile` from `risk_flags` when no explicit profile exists:

- contains `"conservative"` -> `conservative`
- contains `"bold"` -> `bold`
- otherwise -> `balanced`

### Output file

The batch results are saved to:

- `debate_guideArchS_cleaned_scored.csv`

The saved columns include:

- `topic`
- `human_decision`
- `supported_side`
- `debate_answer`
- `reason`
- `message_history`
- `debate_struct_json`
- `scoring_json`

`debate_answer` in the exported dataset is the winner chosen by the scorer, not the raw text answer from the debate.

### `Scorer.py`

`Scorer.py` converts the structured decision model into a ranked set of options.

### Expected input schema

The main entry point is:

`compute_best_option(final_struct: dict) -> dict`

The expected structure is:

```json
{
  "drivers": [
    {
      "name": "latency",
      "orientation": "min",
      "priority": "H",
      "domain": { "min": 0, "max": 1000 },
      "compose": "sum"
    }
  ],
  "options": [
    { "name": "Option A" },
    { "name": "Option B" }
  ],
  "impacts": [
    { "option": "Option A", "driver": "latency", "label": "M" },
    { "option": "Option B", "driver": "latency", "triangle": [300, 450, 700] }
  ],
  "constraints": [
    { "driver": "latency", "type": "max", "value": 700 }
  ],
  "risk_flags": ["balanced"]
}
```

### Scoring logic

The scorer performs the following steps.

#### Driver normalization

Each driver is defined by:

- `orientation`: whether lower is better (`min`) or higher is better (`max`)
- `priority`: `H`, `M`, `L`, or a numeric value
- `domain`: numeric range used to normalize values
- `compose`: how multiple triangles for the same driver are combined

Priority labels map to numeric values:

- `L = 1`
- `M = 5`
- `H = 9`

These values are normalized into driver weights.

#### Impact interpretation

Impacts can be represented in two ways:

- qualitative labels: `L`, `M`, `H`
- explicit triangular fuzzy values: `[optimistic, anticipated, pessimistic]`

If a label is used, the scorer converts it into a triangular value inside the driver's domain.

#### Dependency and conflict handling

The scorer also supports:

- `dependencies`: one option can require other options
- `conflicts`: incompatible options can invalidate a candidate closure

Dependencies are expanded through a closure step before the final score is computed.

#### Missing impact policy

When `require_complete_impacts` is enabled, each valid option must have impacts for all drivers. The behavior then depends on `missing_impact_policy`:

- `invalidate`: option becomes invalid
- `neutral`: missing drivers are filled with midpoint values

`Run_all.py` uses the `neutral` policy.

#### Constraint validation

Constraints are checked against the pessimistic end of each option's triangle:

- `type = "max"` means the pessimistic value must stay below the threshold
- `type = "min"` means the pessimistic value must stay above the threshold

If a constraint fails, the option is invalidated.

#### Total score construction

Each option's per-driver triangles are converted into normalized cost-space triangles. The scorer then computes:

- `za`: center of the total triangle
- `zn`: risk spread
- `zp`: opportunity spread

Those values are normalized across valid options and combined with one of these weight profiles:

- `balanced`
- `conservative`
- `bold`

The final ranking is built from `phi`, which is the weighted worst normalized component for each option. The option with the smallest `phi` is selected as the winner.

### Return value

`compute_best_option()` returns a dictionary with:

- `winner`
- `ranking`
- `details`

`details` includes rich debugging information such as:

- validity flags
- missing impacts
- dependency closure
- normalized driver triangles
- invalidation reasons
- rejected constraints
- `mu` and `phi` values

### Helper function

`rank_criticality(scored, top_t=3, decay=0.7)` provides a rough criticality ranking for drivers based on high-ranking options and uncertainty spread.

### `Utils/Config.py`

This file defines the global `config` dictionary used across the debate workflow.

### Main responsibilities

- stores prompt templates
- stores mutable runtime values such as `debate_topic`
- defines the moderator and judge JSON output contracts

### Prompt strategy

The project intentionally splits prompt behavior:

- debaters respond in free text
- moderator and judge are instructed to return strict JSON

This separation is important because the scorer depends on structured output, but the debate itself still benefits from unconstrained argumentation.

### `Utils/Agents.py`

This file defines the agent model hierarchy.

### `Agent`

Base class with:

- identity fields such as `name`, `role`, and `llm_name`
- `messages` history
- `system_message`
- a shared class-level `agents_registry`

### Shared-message behavior

When one agent adds a message through `add_message()`, that message is also appended to all other registered agents. This creates a shared global conversation history across all participants.

### Model routing

`generate_response()` currently routes as follows:

- `codeqwen` -> `ChatTogether(model="Qwen/Qwen2.5-Coder-32B-Instruct")`
- `deepseek` -> `ChatTogether(model="deepseek-ai/DeepSeek-V3")`
- anything else -> `ChatOpenAI(model=self.llm_name)`

The file imports `LlamaSSHLLM`, but the current implementation does not use it.

### `Utils/Env.py`

Contains a small helper:

`set_env(var: str)`

If the environment variable is missing, it prompts for the value using `getpass`, which prevents the key from being echoed to the terminal.

`Run_all.py` uses this for `OPENAI_API_KEY`.

### `Utils/LLM.py`

Provides an SSH-backed custom LangChain LLM wrapper:

- `SSHConnection`
- `LlamaSSHLLM`

This implementation:

- opens an SSH connection
- sends a `curl` request to a remote Ollama-style API endpoint
- parses the returned JSON
- exposes the result through a LangChain-compatible LLM class

At the moment, this path is not wired into `Agent.generate_response()`, so it is effectively dormant unless you integrate it yourself.

### `Utils/Nodes.py`

Contains the executable node functions used by the debate graph.

### Main nodes

- `affirmative_opening()`
- `negative_opening()`
- `positive_debater_turn()`
- `negative_debater_turn()`
- `moderator_evaluation()`
- `judge_verdict()`
- `check_debate_over()`

### Moderator behavior

`moderator_evaluation()`:

- builds a round summary prompt
- asks the moderator for strict JSON
- attempts to parse the reply
- ends the debate early if the moderator returns:
  - `"Whether there is a preference": "Yes"`
  - a usable `final` structure with drivers, options, and impacts

If the moderator cannot be parsed as JSON, the system simply continues to the next round.

### Judge behavior

`judge_verdict()` is used when the debate reaches `max_rounds`. It:

1. asks for candidate answers in free text
2. asks again for final strict JSON
3. saves `parsed["final"]` into `state["debate_answer"]`

### `Utils/State_graph.py`

Builds the LangGraph state machine with `StateGraph` and `MemorySaver`.

The graph edges are:

- `START -> affirmative_opening`
- `affirmative_opening -> negative_opening`
- `negative_opening -> moderator_evaluation`
- `moderator_evaluation -> positive_debater_turn | judge_verdict | END`
- `positive_debater_turn -> negative_debater_turn`
- `negative_debater_turn -> moderator_evaluation`
- `judge_verdict -> END`

## Setup

### Requirements

Recommended:

- Python 3.11+
- access to an OpenAI-compatible model if running `Run_all.py`

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
pip install langchain-together
```

`langchain-together` is imported in `Utils/Agents.py` but is not listed in `Requirements.txt`, so it may need to be installed separately.

### Credentials

Before running `Run_all.py`, ensure `OPENAI_API_KEY` is available in the environment. If it is not, the script will prompt for it at runtime.

Example:

```powershell
$env:OPENAI_API_KEY="your-key"
python Run_all.py
```

## How To Run

### Single debate

```powershell
python Main.py
```

Use this when you want to test the debate graph on one manually defined topic.

### Batch debate plus scoring

```powershell
python Run_all.py
```

Use this when you want to process the ADR dataset and generate scored results.

## Data Inputs And Outputs

### Input dataset used by `Run_all.py`

- file: `adrs_final_sample_58.csv`
- required columns:
  - `context_considered_drivers`
  - `other_sections`

### Main batch output

- file: `debate_guideArchS_cleaned_scored.csv`

### Output meaning

- `topic`: source prompt used for the debate
- `human_decision`: reference human decision text from the input row
- `supported_side`: moderator-selected side, when available
- `debate_answer`: scorer-selected winning option
- `reason`: explanation returned by the moderator or judge
- `message_history`: concatenated debate transcript
- `debate_struct_json`: structured decision model extracted from the debate
- `scoring_json`: full scorer output as JSON text

