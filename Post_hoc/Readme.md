# Supplementary Material: Grouping and Analysis Pipeline

## Overview

This repository contains the full pipeline used to transform raw Architecture Decision Records (ADRs) into a structured analytical dataset and subsequently evaluate statistical relationships between decision characteristics and decision outcomes.

The implementation is intentionally organized into multiple scripts reflecting distinct processing stages rather than a monolithic transformation. This separation mirrors the operational execution of the study, where intermediate artifacts were produced, validated, and refined before downstream analysis.

The pipeline is composed of three primary components:

* `Grouping.py`: construction of structured variables from ADR data
* `Group_analysis.py`: statistical analysis over grouped categorical features
* `Drivers_options_analysis.py`: extraction and evaluation of structural complexity proxies (number of drivers and options)

---

## 1. Grouping Pipeline (`Grouping.py`)

### 1.1 Purpose

This stage transforms semi-structured ADR content into a tabular representation composed of categorical, ordinal, and multi-label variables.

The process integrates:

* language-model-assisted extraction (initial structuring)
* deterministic heuristics (post-processing and normalization)

---

### 1.2 Extracted Variables

The following variables are generated:

* `layer`
* `family`
* `decision_type`
* `scope`
* `lifecycle_stage`
* `ambiguity_level`
* `pattern_or_tactic`
* `risk_flags`
* `concerns`
* `main_concerns`
* `conflicting_drivers_count`
* `forces_documented`
* `alternatives_listed`
* `consequences_listed`

---

### 1.3 Data Integration

The dataset is enriched with evaluation outputs:

* `comparison_result` is merged into the grouped dataset
* alignment is performed positionally (row-wise consistency required)

---

### 1.4 Backfilling Missing Values

Missing entries in key categorical variables are completed using constrained inference:

* `decision_type`
* `scope`
* `layer`
* `family`

This step is applied selectively and tracked implicitly through the transformation pipeline.

---

### 1.5 Concern Refinement

Two concern representations are maintained:

* `concerns`: broader ADR-level extraction
* `main_concerns`: extracted from decision outcome rationale

Normalization includes:

* whitespace cleanup
* capitalization standardization
* alias consolidation

---

### 1.6 Decision Type Remapping

Entries labeled as `technology` are reclassified using keyword-based rules into:

* `behavior_integration`
* `cross_cutting`
* `process`
* `structural`

This avoids overly generic categorization.

---

### 1.7 Family Aggregation

Fine-grained `family` labels are mapped into:

#### `family_kind7`

* Data
* Application (Implementation)
* Integration & Interfaces
* Platform & Delivery
* Security & Compliance
* Governance & Documentation
* Operations & Observability

#### `family_kind8`

Further refinement splits application-related decisions into:

* Application — Frontend
* Application — Backend

Additional heuristics are applied where layer information is insufficient.

---

### 1.8 Final Output

* `Final_Grouping2.csv`

This dataset is the canonical input for all downstream analysis.

---

## 2. Group-Based Analysis (`Group_analysis.py`)

### 2.1 Outcome Definition

The target variable is derived from:

* `comparison_result`

Extraction is performed via regex:

* primary extraction from prefix (before `//`)
* fallback extraction from full string

Final variable:

* `Target ∈ {Yes, No, Maybe}`

---

### 2.2 Preprocessing

* parsing of list-like fields (`concerns`, `main_concerns`)
* one-hot encoding of categorical and multi-label variables
* handling of missing values
* optional filtering of low-variance features

---

### 2.3 Bivariate Analysis

#### Concern-Level Tests

For each concern:

* contingency table (present vs absent × outcome)
* chi-square test
* Cramér’s V
* support counts

Multiple testing correction:

* Benjamini–Hochberg (FDR)

---

#### Categorical Variable Tests

Applied to:

* `layer`
* `family_kind8`
* `decision_type`
* `scope`
* `ambiguity_level`

Procedure:

* collapse rare levels into `Other`
* chi-square test
* standardized residuals
* row-wise proportions
* optional binary reduction (`Yes` vs `Not-Yes`)

---

### 2.4 Multinomial Models

Models are fitted using:

* `statsmodels.MNLogit`
* `scikit-learn` multinomial logistic regression

Outputs include:

* coefficients
* odds ratios / relative risk ratios
* statistical significance
* model fit indicators

---

### 2.5 Robustness and Sensitivity

* removal of sparse predictors
* handling of separation-prone variables
* re-estimation of models under reduced feature sets

---

## 3. Drivers and Options Analysis (`Drivers_options_analysis.py`)

### 3.1 Purpose

This component introduces structural complexity proxies derived directly from ADR content:

* number of decision drivers (`n_drivers`)
* number of considered options (`n_options`)

Unlike previous variables, these are obtained via deterministic parsing of ADR markdown structure.

---

### 3.2 Extraction Method

Counts are extracted from:

* `topic` field (raw ADR text)

Procedure:

1. Identify relevant sections using Markdown headings:

   * `Decision Drivers`
   * `Options`, `Alternatives`, or equivalent variants

2. Extract section body

3. Count top-level list items:

   * bullet lists (`-`, `*`, `+`)
   * numbered lists (`1.`, `2)`)
   * checkbox lists

Nested lists are intentionally ignored.

---

### 3.3 Generated Variables

* `n_drivers`
* `n_options`

---

### 3.4 Target Construction

Same procedure as other analyses:

* extract `Yes`, `No`, or `Maybe` from `comparison_result`
* prioritize prefix before `//`
* fallback to full string if needed

---

### 3.5 Multinomial Model (Counts)

A multinomial logistic regression is fitted:

* outcome: `Target`
* predictors:

  * `n_drivers`
  * `n_options`
  * optional controls:

    * `layer`
    * `family_kind8`
    * `decision_type`
    * `scope`
    * `lifecycle_stage`
    * `ambiguity_level`

Outputs:

* coefficients
* relative risk ratios

---

### 3.6 Centering

For robustness:

* centered variables:

  * `n_drivers_c`
  * `n_options_c`

These are used in predictive models and cross-validation.

---

### 3.7 Binary Model

Additional analysis:

* `Yes` vs `Not-Yes`

Used to evaluate separability of successful decisions.

---

### 3.8 Cross-Validation

Models evaluated using:

* 5-fold stratified cross-validation

#### Binary metrics:

* accuracy
* F1 score
* log loss
* Brier score

#### Multinomial metrics:

* accuracy
* macro-F1
* log loss

---

## 4. Generated Artifacts

### Grouping Outputs

* `Grouped_decisions.csv`
* `Grouped_decisions_with_comparison.csv`
* `Grouped_decisions_with_comparison_filled.csv`
* `Grouped_decisions_with_concerns_split.csv`
* `Final_Grouping.csv`
* `Final_Grouping2.csv`

### Analysis Scripts

* `Grouping.py`
* `Group_analysis.py`
* `Drivers_options_analysis.py`

---

## 5. Remarks

The pipeline reflects an iterative construction process where structured variables were progressively refined through a combination of automated extraction and heuristic adjustments.

The grouping stage prioritizes coverage and normalization, while the analysis stage emphasizes statistical validity and robustness.

The drivers/options analysis extends the representation by incorporating structural indicators derived directly from ADR content, complementing the categorical grouping variables.

No single script is intended to fully reproduce the pipeline in isolation; instead, the provided components represent the operational decomposition used during the study.

