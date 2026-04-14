# MAD Debate Framework

This repository contains a multi-agent debate workflow built with LangChain and LangGraph. Two debaters argue over a prompt, a moderator evaluates whether one side is already preferable, and a judge produces a final answer if the debate does not converge early.

This README is intentionally scoped to:

- `Main.py`
- `Run_all.py`
- `Utils/`

It does not document the analysis notebooks or post-processing scripts outside that scope.

## What The Code Does

The framework simulates a structured debate between four agents:

- `AffirmativeSide`: proposes or supports an answer
- `NegativeSide`: challenges the affirmative answer and proposes an alternative
- `Moderator`: evaluates the current round and decides whether a winner is already clear
- `Judge`: produces the final verdict when the moderator does not end the debate early

The orchestration is implemented as a LangGraph state machine. Each node represents a debate step, and the graph loops through debate rounds until one of these conditions is met:

- the moderator says there is a preference and returns a final answer
- the number of rounds exceeds the configured limit, in which case the judge is invoked

## Repository Area Covered

```text
Main.py
Run_all.py
Utils/
  Agents.py
  Config.py
  Env.py
  LLM.py
  Nodes.py
  State_graph.py
  nb_summarize.py
```

## High-Level Architecture

The runtime flow is:

1. A debate topic is defined.
2. Four agents are created and given role-specific system prompts.
3. A debate state dictionary is initialized.
4. `build_state_graph()` compiles the LangGraph workflow.
5. The graph executes debate turns and updates the shared state.
6. The final supported side, answer, and reasoning are read from the returned state.

The important design decision in this codebase is that all agent messages are shared across all agents through a global registry. That means each new message is appended to every agent's history, so each participant sees the full debate context.

## Dependencies

The project depends mainly on:

- `langchain`
- `langgraph`
- `langchain-openai`
- `langchain_ollama`
- `pandas`
- `numpy`
- `scikit-learn`
- `paramiko`

The pinned dependencies are listed in `Requirements.txt`.

## Installation

Create and activate a Python environment, then install the requirements:

```bash
pip install -r Requirements.txt
```

## Model Backends

The project currently supports two model access patterns:

### 1. Remote Llama via SSH

`Main.py` uses `llm_name="llama"` for every agent. In `Utils/Agents.py`, that name maps to `LlamaSSHLLM` from `Utils/LLM.py`.

`LlamaSSHLLM`:

- opens an SSH connection with `paramiko`
- executes a `curl` request on the remote machine against `http://localhost:xxxxx/api/chat`
- expects an Ollama-compatible chat endpoint on the remote host
- uses model name `llama3.3`

### 2. OpenAI Chat Models

`Run_all.py` uses `llm_name="gpt-4o"` for all agents. In that path, `Utils/Agents.py` instantiates `ChatOpenAI(model=self.llm_name, temperature=0)`.

`Run_all.py` also calls:

```python
set_env("OPENAI_API_KEY")
```

so the script will prompt for the API key if it is not already available in the environment.

## Important Setup Notes

Before running the code, check these assumptions:

- `Main.py` assumes the SSH/Ollama configuration in `Utils/LLM.py` is valid.
- `Run_all.py` assumes an OpenAI API key is available.
- `Run_all.py` reads a CSV from a hard-coded absolute path:

```text
C:/Users/gabri/Downloads/all_three_extract.csv
```

- `Run_all.py` writes results to:

```text
debate_BIG_GPT.csv
```

If your machine or dataset path differs, update the script before running it.

## Main.py

`Main.py` is the single-topic entrypoint. It runs one debate for one hard-coded prompt.

### What It Does

- imports the agent classes, config, and state graph builder
- defines `initialize_agents(config)`
- creates four agents
- injects the debate topic into the system prompts
- initializes the debate state
- builds the LangGraph workflow
- invokes the graph

### Debate Topic

The topic is hard-coded inside `main()`:

```python
topic = "Propose a solution for the following ::## Context and Problem Statement ..."
```

To use another debate prompt, replace that string directly in the script.

### Agent Initialization

`initialize_agents(config)` performs these steps:

- clears `Agent.agents_registry`
- creates two `Debater` instances
- creates one `Moderator`
- creates one `Judge`
- fills the `##debate_topic##` placeholders in the configured prompts
- assigns each system prompt to its agent

All four agents in `Main.py` use `llm_name="llama"`.

### State Object

The debate state passed into LangGraph contains:

- `round_number`
- `max_rounds`
- `topic`
- `positive_debater`
- `negative_debater`
- `moderator`
- `judge`
- `debate_over`
- `messages`
- `aff_ans`
- `neg_ans`
- `supported_side`
- `debate_answer`
- `reason`

These fields are updated across the graph execution.

### How To Run

```bash
python Main.py
```

### Expected Behavior

When executed, the script prints each speaker's output to the console as the graph advances. It does not currently save the final result to disk.

## Run_all.py

`Run_all.py` is the batch-processing entrypoint. It applies the same debate process to every row of an input CSV.

### What It Does

- loads a CSV into a pandas DataFrame
- loops through all rows
- creates a topic from `context_considered_drivers`
- runs a debate for each row
- stores the debate output plus metadata
- writes the aggregated results to `debate_BIG_GPT.csv`

### Input Expectations

The CSV is expected to contain at least these columns:

- `context_considered_drivers`
- `other_sections`

Those are used as:

- `context_considered_drivers`: the generated debate prompt
- `other_sections`: a reference value saved as `human_decision`

### Topic Construction

For each row, the topic is constructed as:

```python
topic = f" Select the best option for the following:: {row['context_considered_drivers']}"
```

### Output Columns

The output CSV contains:

- `topic`
- `human_decision`
- `supported_side`
- `debate_answer`
- `reason`
- `message_history`

### How To Run

```bash
python Run_all.py
```

### Notes About The Batch Script

- all agents use `gpt-4o`
- each row is assigned a `thread_id` equal to its DataFrame index
- the graph is rebuilt for each topic
- results are accumulated in memory and saved once at the end

## Utils Module Documentation

## Utils/Config.py

`Config.py` defines a single `config` dictionary used by the rest of the system.

It stores:

- the current `debate_topic`
- placeholders for `base_answer` and `debate_answer`
- the system prompt for debaters
- the system prompt for moderator and judge
- the prompt templates used during each debate step

### Key Prompt Templates

`player_meta_prompt`

- base system prompt for debaters
- receives `##debate_topic##`

`moderator_meta_prompt`

- base system prompt for moderator and judge
- instructs the model to output strict JSON

`affirmative_prompt`

- opening prompt for the affirmative side

`negative_prompt`

- tells the negative side to respond to the affirmative answer

`moderator_prompt`

- asks the moderator to evaluate both answers after a round

`judge_prompt_last1`

- asks the judge to list the candidate answers without reasons

`judge_prompt_last2`

- asks the judge for a final JSON verdict

`debate_prompt`

- used for back-and-forth turns after the openings

## Utils/Agents.py

`Agents.py` defines the core agent classes.

### Agent

`Agent` is a `pydantic.BaseModel` with:

- `name`
- `role`
- `llm_name`
- `messages`
- `system_message`
- `description`

It also maintains:

- `agents_registry`: a class-level list of every created agent

### Important Behavior

When an agent receives a message through `add_message()`:

- the message is appended to that agent
- the same message is also appended to every other registered agent

This creates a shared global conversation history across all agents.

### Response Generation

`generate_response()`:

- chooses the backend based on `llm_name`
- uses `LlamaSSHLLM()` when `llm_name == "llama"`
- otherwise uses `ChatOpenAI`
- prepends the agent's `system_message`
- sends the full message history to the model
- stores the generated `AIMessage`

### Debater

`Debater` extends `Agent` and adds:

- `stance`

It also exposes a `persona` property combining name, stance, and description.

### Moderator

`Moderator` extends `Agent` and provides:

- `evaluate_debate()`
- `decide_to_continue()`

In practice, the graph uses `generate_response()` through the node functions in `Utils/Nodes.py`.

### Judge

`Judge` extends `Agent` and provides:

- `provide_verdict()`

## Utils/Env.py

`Env.py` contains a small helper:

```python
set_env(var: str)
```

If the environment variable is missing, it securely prompts for it using `getpass.getpass()`.

This is used by `Run_all.py` for `OPENAI_API_KEY`.

## Utils/LLM.py

`LLM.py` implements a custom LangChain-compatible LLM wrapper for a remote Llama model.

### SSHConnection

`SSHConnection`:

- connects to a remote host with `paramiko`
- sends a JSON payload to an Ollama chat endpoint
- returns the parsed JSON response
- exposes `close()`

### LlamaSSHLLM

`LlamaSSHLLM` extends `BaseLLM` and implements:

- `_call()`
- `_generate()`
- `_llm_type`
- `close()`

Its `_call()` method sends a single chat request with:

- model: `llama3.3`
- temperature: `0`
- `stream=False`

This class is what allows `Agent.generate_response()` to treat the SSH-backed Llama model similarly to a standard LangChain LLM.

## Utils/Nodes.py

`Nodes.py` contains the debate state definition and all node functions used by the graph.

### DebateState

`DebateState` is a `TypedDict` defining the full graph state shape.

### Node Functions

`affirmative_opening(state)`

- sends the initial topic prompt to the affirmative debater
- stores the answer in `aff_ans`
- appends the answer to `messages`

`negative_opening(state)`

- asks the negative debater to respond to the affirmative answer
- stores the answer in `neg_ans`
- appends the answer to `messages`

`positive_debater_turn(state)`

- asks the affirmative debater to respond to the latest negative answer
- updates `aff_ans`

`negative_debater_turn(state)`

- asks the negative debater to respond to the latest affirmative answer
- updates `neg_ans`

`moderator_evaluation(state)`

- builds the moderator prompt using the current round and both answers
- requests a strict JSON evaluation
- parses:
  - `Whether there is a preference`
  - `Supported Side`
  - `Reason`
  - `debate_answer`
- ends the debate if preference is `yes`
- otherwise increments `round_number`

If the moderator response is not valid JSON, the code prints an error and continues the debate.

`judge_verdict(state)`

- prompts the judge twice
- first to enumerate answer candidates
- then to return the final JSON verdict
- stores `debate_answer` and `reason`
- marks the debate as over

`check_debate_over(state)`

- returns `END` if the debate is finished
- returns `judge_verdict` if `round_number > max_rounds`
- otherwise returns `positive_debater_turn`

### Round Logic Detail

The stopping condition is:

```python
elif state['round_number'] > state['max_rounds']
```

This means the judge is triggered only after the round counter exceeds `max_rounds`, not when it is equal. With `max_rounds = 3`, the debate can continue through multiple moderator evaluations before falling through to the judge.

## Utils/State_graph.py

`State_graph.py` assembles the LangGraph workflow.

### Graph Structure

The graph is built with these nodes:

- `affirmative_opening`
- `negative_opening`
- `positive_debater_turn`
- `negative_debater_turn`
- `moderator_evaluation`
- `judge_verdict`

### Edge Flow

The graph flow is:

```text
START
  -> affirmative_opening
  -> negative_opening
  -> moderator_evaluation
     -> END                  if moderator ends debate
     -> positive_debater_turn
        -> negative_debater_turn
        -> moderator_evaluation
     -> judge_verdict        if rounds exceeded
        -> END
```

The graph is compiled with `MemorySaver()` as its checkpointer.


## End-To-End Debate Flow

At runtime, a typical debate proceeds like this:

1. The affirmative side answers the topic.
2. The negative side disagrees and proposes an alternative.
3. The moderator checks whether one answer is already clearly better.
4. If not, the two debaters continue responding to each other.
5. After each exchange, the moderator checks again.
6. If the moderator still does not choose a side after the round limit is exceeded, the judge produces the final answer.

## Inputs And Outputs Summary

### Inputs

- a debate topic string in `Main.py`
- a CSV file in `Run_all.py`
- API credentials or remote model connectivity
- prompt templates in `Utils/Config.py`

### Outputs

- console logs for each debate step
- final state values returned by the LangGraph invocation
- `debate_BIG_GPT.csv` for batch execution in `Run_all.py`

## Known Limitations And Risks

The current code works as a prototype, but there are several implementation constraints worth documenting:

- `Run_all.py` uses a hard-coded absolute input CSV path.
- `Main.py` and `Run_all.py` do not expose command-line arguments.
- The moderator and judge rely on the model returning perfectly valid JSON.
- The remote Llama connection is hard-coded in `Utils/LLM.py`.
- `Main.py` and `Run_all.py` use different model backends, so results are not directly comparable unless that is intentional.
- The debate prompts and stopping logic are fixed in code rather than externally configured.

## Minimal Quick Start

If you only want the shortest path to run the code:

1. Install dependencies with `pip install -r Requirements.txt`.
2. Decide whether you want the SSH-backed Llama flow (`Main.py`) or the OpenAI batch flow (`Run_all.py`).
3. Update any hard-coded paths or credentials as needed.
4. Run either:

```bash
python Main.py
```

or:

```bash
python Run_all.py
```
