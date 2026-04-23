# DATA DESCRIPTION

This document describes the main data files, variables, and output structures used in the TRACE-ED simulation workflow.

## 1. Overview

The TRACE-ED workflow uses synthetic short-answer responses and rubric-linked keyword definitions as protocol inputs.  
The simulation then generates run-level outputs and an aggregated results file (`records.json`), which is used for post-run analysis.

The repository is organized so that:
- protocol inputs are stored in `00_protocol/`;
- simulation outputs are stored in `03_runs/` or `03_runs_sample/`;
- analysis outputs are stored in the `analysis/` subfolder of a run folder.

## 2. Main Input Files

### 2.1 `00_protocol/pilot_answers_v1.json`

This file contains the main synthetic response dataset used in the simulation.

It provides:
- `question_text`  
  the question prompt used in the experiment
- `rubric`  
  the rubric structure used for scoring
- `answers`  
  the synthetic short-answer responses evaluated in the run

#### Expected structure
The file is expected to contain:
- one question definition
- one rubric structure
- a list of answers

#### Main content roles
- `question_text` is passed to the model during execution
- `rubric` is used in scoring and explanation generation
- `answers` are iterated across temperatures, repetitions, and architectures

---

### 2.2 `00_protocol/rubric_keywords_v1.json`

This file contains rubric-linked keyword definitions used in the TRACE-ED grounding procedure.

These keywords are matched to the configured `question_id` and support:
- semantic grounding checks
- absence-claim handling
- expected concept presence checks

#### Main content role
This file is used during grounding evaluation to determine whether explanation claims are sufficiently supported by the student answer text.

---

## 3. Configuration File

### `config.json`

This file stores the main simulation settings.

#### Main fields
- `experiment_name`  
  label used to generate the run folder name
- `model`  
  LLM model used for the simulation
- `question_id`  
  identifier used to select the relevant keyword set
- `temperatures`  
  list of temperature settings tested
- `n_repeats`  
  number of repetitions per condition
- `tau_grounding`  
  grounding threshold used in claim-evidence matching
- `run_root`  
  root directory where run folders are created

---

## 4. Main Aggregated Output File

### `records.json`

This is the central aggregated output file produced by the simulation script.

It contains one record per run condition.

#### Main variables in each record

- `answer_id`  
  identifier of the synthetic answer being evaluated

- `arch`  
  architecture type used in the run  
  expected values:
  - `single`
  - `multi`

- `temperature`  
  temperature value used for that run condition

- `repeat`  
  repetition index of the run condition

- `total_score`  
  total score assigned in the run

- `grounding_rate`  
  proportion of explanation claims that are considered grounded in the answer text

- `mean_coherence`  
  average coherence between score polarity and explanation polarity

- `contradiction_rate`  
  proportion of explanation claims showing contradiction between score direction and language polarity

#### Role of `records.json`
This file is used as the main dataset for the analysis script.  
All summary tables and architecture comparisons are derived from this file.

---

## 5. Run-Level Output Files

Within a run folder, the `runs/` subdirectory contains one folder per run condition.

Example structure:

```text
runs/
  single_A1_temp_0.0_rep_1/
    scoring.json
    explanation.txt
    full_output.txt
  multi_A1_temp_0.0_rep_1/
    scoring.json
    explanation.txt

### 5.1 `scoring.json`
Contains the structured scoring output produced by the model.

Typical content includes:
- `question_id`
- `indicators`
- `total_score`
- `max_total_score`

### 5.2 `explanation.txt`
Contains the explanation output produced for the run condition.

This file is used in the computation of:
- grounding rate
- coherence
- contradiction rate

### 5.3 `full_output.txt`
May appear in single-agent runs where the raw combined output is also stored.

---

## 6. Analysis Output Files

After running `analyze_monte_carlo.py`, an `analysis/` subfolder is created in the selected run directory.

This folder contains the main summary outputs.

### 6.1 `analysis_summary_by_architecture.csv`
Summary statistics for each metric across all runs within each architecture.

### 6.2 `analysis_summary_by_architecture_and_temperature.csv`
Summary statistics for each metric by architecture and temperature.

### 6.3 `analysis_icc_by_architecture.csv`
ICC(1,k) summary for total score by architecture.

### 6.4 `analysis_architecture_comparison.csv`
Overall comparison between single-agent and multi-agent conditions, including:
- mean
- standard deviation
- Welch’s t
- p-value
- Cohen’s d

### 6.5 `analysis_summary_full.json`
Machine-readable export of the complete analysis summary.

### 6.6 `analysis_report.txt`
Human-readable text summary of the analysis results.

---

## 7. Metric Definitions

### `total_score`
The total score assigned to an answer in a given run.

### `grounding_rate`
The proportion of explanation claims judged to be grounded in the student answer text.

### `mean_coherence`
The average coherence between score polarity and explanation polarity.

### `contradiction_rate`
The proportion of claims showing contradiction between scoring direction and evaluative language.

---

## 8. Data Scope

The TRACE-ED workflow uses **synthetic short-answer responses only**.  
No human participant data or identifiable personal data are included.

---

## 9. Reproducibility Note

The files in this repository are intended to preserve a reproducible chain from:
- protocol inputs,
- to simulation configuration,
- to run-level execution,
- to aggregated outputs,
- to structured analysis summaries.
