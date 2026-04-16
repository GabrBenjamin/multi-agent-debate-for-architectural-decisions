# 🧠 Multi-Agent Debate (MAD) Framework for Architectural Decision-Making

## 📌 Overview

This repository provides an experimental framework to evaluate the use of **Large Language Models (LLMs)** for **software architecture decision-making**, with a particular focus on comparing:

- Multi-Agent Debate (MAD)
- Single-Agent baselines
- Structured reasoning approaches (e.g., GuideArch, ATAM-inspired scenarios)
- Retrieval-Augmented Generation (RAG)

The objective is to assess whether LLM-based approaches can **replicate or approximate architectural decisions documented in Architecture Decision Records (ADRs)**.

---

## 🏗️ Repository Structure

```

├── MAD_Guide_arch/        # MAD with GuideArch-inspired reasoning
├── MAD_ATAM/             # MAD with ATAM-style scenario generation (
├── MAD_RAG/              # MAD with retrieval-augmented generation 
├── MAD_More_agents/      # Experiments with different agent configurations
├── Single_Agent/         # Baseline approaches (no debate)
├── Post_hoc/             # Post-hoc analysis and evaluation scripts
├── MAD_Regular_roles/    # Base MAD framework (supports role-based prompting)
├── README.md             # This file
└── LICENSE
```

## ⚙️ Core Concepts

### 🔹 Multi-Agent Debate (MAD)
Multiple LLM agents debate architectural decisions:
- Typically 2 agents (affirmative vs negative)
- Multi-round interaction
- A moderator (and optional judge) determines the final outcome

---

### 🔹 Single-Agent Baselines
Used for comparison:
- Zero-shot prompting
- Chain-of-thought (CoT)
- Few-shot prompting

---

### 🔹 Variants

Each folder explores a different enhancement:

- **GuideArch** → structured reasoning guidance
- **ATAM** → scenario-based evaluation
- **RAG** → repository-informed decision-making
- **Roles** → stakeholder simulation (e.g., architect, SRE)

---

## 🚀 Getting Started

### 1. Clone the repository

git clone <your-repo-url>
cd <repo-name>

### 2. Install dependencies


pip install -r requirements.txt
### 3. Set environment variables
export OPENAI_API_KEY=your_key_here
### ▶️ Running Experiments

Each module has its own execution pipeline.

👉 Navigate to the desired folder and follow its instructions:

MAD_Guide_arch/README.md
Single_Agent/README.md
MAD_RAG/README.md
MAD_ATAM/README.md
### 📊 Evaluation

The framework evaluates model outputs against ADR decisions using three categories:

Match (Yes): The model decision aligns with the ADR decision.
Mismatch (No): The model decision differs from the ADR decision.
Uncertainty (Maybe): The ADR or the model does not provide a definitive or directly comparable decision.
✅ Match Example

ADR:

## Decision Outcome

Chosen option: "Separate URL creation", because comes out best (see below).

MAD Output:

The negative side provides a compelling argument for separating URL creation into its own method, emphasizing modularity, reusability, and maintainability...
Therefore, Option 1 is the more robust and flexible solution.

Result:

1 - Yes // Both decisions favor separating URL creation for better logging and error handling.

✔️ Explanation: Both the ADR and the model select the same architectural strategy.

❌ Mismatch Example

ADR:

# Decision Outcome

Chosen option: build on top of Experimenter and invest in improvements...
We determined that the risk of starting a new application was too significant.

MAD Output:

The negative side provides a more balanced approach... leveraging existing infrastructure while planning for future development... including transition to Nimbus.

Result:

5 - No // The human decision chose to build on top of Experimenter, while the debate answer suggests a parallel strategy with gradual development of a new system.

❗ Explanation: The model proposes a strategy that differs from the human decision.

⚠️ Uncertainty Example

ADR:

## Decision Outcome

Ideally I would say Option 1, but we will need to investigate further to conclude properly.

MAD Output:

The negative side presents a compelling argument... Option 2 offers a pragmatic solution...

Result:

34 - Maybe // The human decision leans towards Option 1 but requires further investigation, while the debate answer selects Option 2.

⚠️ Explanation: The ADR does not provide a finalized decision, making comparison inconclusive.

### 🧪 Notes on Evaluation
Decisions are normalized to account for paraphrasing and equivalent alternatives.
Manual verification is used to ensure correctness.
“Uncertainty” cases are tracked separately to avoid bias in evaluation.
###  📁 Data

The experiments are based on:

ADR datasets extracted from open-source repositories
Preprocessed inputs (e.g., context, decision drivers, options)

⚠️ Some datasets may not be included due to size or licensing constraints.

###  🔬 Research Context

This repository supports research in:

LLMs for Software Engineering
Architectural decision-making
Multi-agent systems
Human-AI alignment in design decisions
📌 Notes
Some modules contain experimental or legacy implementations.
Results may vary depending on model versions and API behavior.
Deterministic settings (e.g., temperature = 0) are recommended.
📜 License

See the LICENSE file for details.


---

