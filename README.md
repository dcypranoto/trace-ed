# TRACE-ED

**TRACE-ED** is a reproducibility repository for the study of architectural transparency in LLM-based cognitive assessment.  
It contains the protocol files, simulation scripts, configuration, sample Monte Carlo outputs, and analysis materials used to support the TRACE-ED workflow.

## Repository Purpose

This repository is intended to support:
- reproducibility of the TRACE-ED simulation workflow;
- inspection of the synthetic dataset and rubric-related protocol files;
- verification of the Monte Carlo execution structure;
- review of the post-run analysis outputs.

The repository is organized to separate:
- protocol inputs,
- execution code,
- run outputs,
- and derived analysis summaries.

## Project Structure

```text
trace-ed/
├── README.md
├── LICENSE
├── config.json
├── requirements.txt
├── DATA_DESCRIPTION.md
├── TRACE_ED_Simulation_Guide.pdf
├── .gitignore
│
├── 00_protocol/
│   ├── pilot_answers_v1.json
│   └── rubric_keywords_v1.json
│
├── 01_data/
├── 02_prompts/
├── 03_runs_sample/
│   └── pilot_montecarlo_v1_COMPARE_20260421_200317/
│       ├── records.json
│       ├── runs/
│       └── analysis/
│
├── 04_code/
│   ├── run_compare_single_vs_multi.py
│   ├── run_monte_carlo.py
│   └── analyze_monte_carlo.py
│
├── 05_results/
└── docs/

Main Components
1. Protocol files
The 00_protocol/ directory contains the main experimental inputs:
a. pilot_answers_v1.json
   synthetic short-answer responses, question text, and rubric structure
b. rubric_keywords_v1.json
   rubric-linked keyword definitions used in grounding evaluation
2. Configuration
The file config.json contains the main simulation settings, including:
a. experiment name
b. model name
c. question identifier
d. temperature values
e. number of repetitions
f. grounding threshold
g. run output root

3. Simulation code
The 04_code/ directory contains the main execution and analysis scripts:
a. run_compare_single_vs_multi.py
   runs the TRACE-ED comparative simulation
b. run_monte_carlo.py
   supports Monte Carlo execution workflow
c. analyze_monte_carlo.py
   generates architecture-level and temperature-level summaries from records.json

4. Sample outputs
The 03_runs_sample/ directory contains a representative experiment output folder, including:
a. records.json
b. run-level outputs in runs/
c. post-run summaries in analysis/

Requirements
The workflow requires Python 3.10 or later.
Install dependencies with:
pip install -r requirements.txt

Environment Setup
Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate
Set the OpenAI API key in the active terminal session:
set OPENAI_API_KEY=your_api_key_here

Running the Simulation
Run the main TRACE-ED simulation from the project root:
python 04_code/run_compare_single_vs_multi.py
This will generate a timestamped run folder inside 03_runs/ containing:
a. records.json
b. runs/
c. run-level scoring and explanation outputs

Running the Analysis
After the simulation has completed successfully, run:
python 04_code/analyze_monte_carlo.py
This will generate an analysis/ subfolder inside the selected run directory containing:
a. analysis_summary_by_architecture.csv
b. analysis_summary_by_architecture_and_temperature.csv
c. analysis_icc_by_architecture.csv
d. analysis_architecture_comparison.csv
e. analysis_summary_full.json
f. analysis_report.txt

Recommended Workflow
1. Review and edit config.json
2. Confirm that protocol files are present in 00_protocol/
3. Run a dry run by temporarily setting n_repeats = 2
4. Run the full simulation with n_repeats = 15
5. Verify that records.json and runs/ were generated
6. Run analyze_monte_carlo.py
7. Inspect the outputs in the analysis/ folder

Reproducibility Notes
This repository is structured to preserve a clear chain from:
a. protocol inputs,
b. to configuration,
c. to execution,
d. to aggregated outputs,
e. to structured analysis summaries.
The included TRACE_ED_Simulation_Guide.pdf provides a detailed step-by-step explanation of the full workflow.

Data and Scope
This project uses synthetic short-answer responses prepared for methodological validation.
No human participant data or identifiable personal data are included in the repository.

License
This repository is released under the MIT License.
