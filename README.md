# Mechanistic Interpretability of Error Representation

## Project Overview
This project investigates how language models internally deal with input token errors. We analyze activation patterns, attention heads, and pathways to understand error detection mechanisms.

## Structure Plan

├── README.md
├── environment.yml
├── setup.py
│
├── data/
│   ├── raw/
│   │   ├── number-sentences_clean.csv
│   │   ├── number-sentences_off-by-one.csv
│   │   ├── text-sentences_clean.json
│   │   └── text-sentences_syntax-error.json
│   │   └── text-sentences_typo.json
│   │
│   ├── processed/
│   │   ├── number-sentences_clean.csv
│   │   ├── number-sentences_off-by-one.csv
│   │   ├── text-sentences_clean.json
│   │   └── text-sentences_syntax-error.json
│   │   └── text-sentences_typo.json
│
├── scripts/
│   ├── extract_prompts.py
│   ├── preprocess_data.py
│   ├── run_experiments.py
│   ├── probing_tasks.py
│   ├── causal_interventions.py
│   ├── visualize_results.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── activation_analysis.ipynb
│   └── attention_analysis.ipynb
│
├── hpc_jobs/
│   ├── run_experiments.pbs
│   ├── probing_tasks.pbs
│   └── causal_interventions.pbs
│
├── results/
│   ├── activations/
│   ├── attention_weights/
│   ├── probing/
│   └── causal_analysis/
│
└── logs/
