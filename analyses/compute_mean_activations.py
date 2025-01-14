import os
import pandas as pd
import numpy as np


# Paths
RESULTS_NUMBERS_PATH = "results_numbers/"
RESULTS_TEXT_PATH = "results_text/"
OUTPUT_PATH = "data/activations/"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Groups of files to process
files_to_average = {
    "activations_mean_number-sentences_clean.csv": [
        "activations_run1_number-sentences_clean.csv",
        "activations_run2_number-sentences_clean.csv",
        "activations_run3_number-sentences_clean.csv",
        "activations_run4_number-sentences_clean.csv",
        "activations_run5_number-sentences_clean.csv"
    ],
    "activations_mean_number-sentences_off-by-one.csv": [
        "activations_run1_number-sentences_off-by-one.csv",
        "activations_run2_number-sentences_off-by-one.csv",
        "activations_run3_number-sentences_off-by-one.csv",
        "activations_run4_number-sentences_off-by-one.csv",
        "activations_run5_number-sentences_off-by-one.csv"
    ],
    "activations_mean_text-sentences_clean.csv": [
        "activations_run1_text-sentences_clean.csv",
        "activations_run2_text-sentences_clean.csv",
        "activations_run3_text-sentences_clean.csv"
    ],
    "activations_mean_text-sentences_syntax-error.csv": [
        "activations_run1_text-sentences_syntax-error.csv",
        "activations_run2_text-sentences_syntax-error.csv",
        "activations_run3_text-sentences_syntax-error.csv"
    ],
    "activations_mean_text-sentences_typo.csv": [
        "activations_run1_text-sentences_typo.csv",
        "activations_run2_text-sentences_typo.csv",
        "activations_run3_text-sentences_typo.csv"
    ]
}


def compute_mean_activation(input_files, output_file, base_path):
    # Read and stack all activation files
    activations = [pd.read_csv(os.path.join(base_path, file), header=None) for file in input_files]
    # Compute the mean row-wise
    mean_activations = pd.concat(activations).groupby(level=0).mean()
    # Save to output file
    mean_activations.to_csv(output_file, index=False, header=False)


# Process numerical files
for output_file, input_files in files_to_average.items():
    base_path = RESULTS_NUMBERS_PATH if "number" in output_file else RESULTS_TEXT_PATH
    output_file_path = os.path.join(OUTPUT_PATH, output_file)
    compute_mean_activation(input_files, output_file_path, base_path)

print("Mean activations computed and saved.")
