import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
ACTIVATIONS_PATH = "data/activations/"
OUTPUT_PATH = "analyses/calculations/"

# Define files
number_files = [
    "activations_mean_number-sentences_clean.csv",
    "activations_mean_number-sentences_off-by-one.csv"
]

text_files = [
    "activations_mean_text-sentences_clean.csv",
    "activations_mean_text-sentences_syntax-error.csv",
    "activations_mean_text-sentences_typo.csv"
]


# Variance Calculation Script
def calculate_variance(file_list, output_file):
    variances = {}
    for file in file_list:
        activations = pd.read_csv(os.path.join(ACTIVATIONS_PATH, file), header=None)
        variances[file] = activations.var(axis=0).mean()  # Mean variance across neurons

    # Save variance results to a file
    with open(output_file, 'w') as f:
        for file, variance in variances.items():
            f.write(f"{file}: {variance}\n")
        print(f"Variance results saved to {output_file}")


# Calculate variance for number sentences
calculate_variance(number_files, OUTPUT_PATH + "variance_numbers.txt")

# Calculate variance for text sentences
calculate_variance(text_files, OUTPUT_PATH + "variance_text.txt")

print("Calculations saved!")
