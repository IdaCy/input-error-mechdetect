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
│   │   ├── clean_sentences.json
│   │   ├── typos.json
│   │   ├── syntax_errors.json
│   │   ├── numerical_prompts.json
│   │   └── off_by_one_errors.json
│   │
│   ├── processed/
│   │   ├── tokenized_clean.pt
│   │   ├── tokenized_typos.pt
│   │   ├── tokenized_syntax_errors.pt
│   │   ├── tokenized_numerical.pt
│   │   └── tokenized_off_by_one.pt
│   │
│   └── prompts_generator.py
│
├── experiments/
│   ├── experiment_01_activation_analysis/
│   │   ├── activation_results_layer_1-12.csv
│   │   ├── activation_results_layer_13-24.csv
│   │   ├── visualize_activations.py
│   │   └── activation_analysis.ipynb
│   │
│   ├── experiment_02_attention_analysis/
│   │   ├── attention_results.json
│   │   ├── visualize_attention.py
│   │   └── attention_analysis.ipynb
│   │
│   ├── experiment_03_probing_tasks/
│   │   ├── probe_classifier.py
│   │   ├── probing_results.json
│   │   └── probing_analysis.ipynb
│   │
│   ├── experiment_04_causal_interventions/
│   │   ├── ablation_results.csv
│   │   ├── causal_paths.json
│   │   └── causal_intervention_analysis.ipynb
│
├── models/
│   ├── llama/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── model_weights.pt
│   │
│   ├── llama2/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── model_weights.pt
│
├── scripts/
│   ├── run_prompt_inference.py
│   ├── extract_activations.py
│   ├── analyze_attention.py
│   ├── train_probe.py
│   ├── run_ablation_study.py
│   └── utils.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tokenization_analysis.ipynb
│   ├── 03_activation_visualization.ipynb
│   └── 04_attention_analysis.ipynb
│
├── results/
│   ├── activation_maps/
│   │   ├── clean_vs_typo/
│   │   │   ├── layer_1_activation.png
│   │   │   ├── layer_2_activation.png
│   │   │   └── ...
│   │   ├── syntax_errors/
│   │   │   ├── layer_1_activation.png
│   │   │   └── ...
│   │
│   ├── attention_maps/
│   │   ├── clean_vs_typo/
│   │   │   ├── head_1_attention.png
│   │   │   └── ...
│   │   └── syntax_errors/
│   │       ├── head_1_attention.png
│   │       └── ...
│   │
│   ├── probing/
│   │   ├── accuracy_scores.csv
│   │   └── layer_probing_summary.json
│   │
│   └── causal_interventions/
│       ├── ablation_summary.csv
│       └── mediation_paths.json
│
└── logs/
    ├── run_logs/
    │   ├── prompt_inference_2025-01-11.log
    │   ├── activation_analysis_2025-01-12.log
    │   └── ...
    │
    └── error_logs/
        ├── inference_errors.log
        └── data_processing_errors.log
