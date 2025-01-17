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


def find_neurons_with_largest_mean_differences(clean_file, corrupted_file, output_file):
    clean_activations = pd.read_csv(os.path.join(ACTIVATIONS_PATH, clean_file), header=None)
    corrupted_activations = pd.read_csv(os.path.join(ACTIVATIONS_PATH, corrupted_file), header=None)
    
    clean_mean = clean_activations.mean(axis=0)
    corrupted_mean = corrupted_activations.mean(axis=0)
    mean_differences = abs(clean_mean - corrupted_mean)
    
    # Find top 10 neurons with largest differences
    largest_diff_neurons = mean_differences.nlargest(10).index.tolist()
    with open(output_file, 'w') as f:
        f.write("Neuron Index, Mean Difference\n")
        for idx in largest_diff_neurons:
            f.write(f"{idx}, {mean_differences[idx]}\n")
    print(f"Largest mean differences saved to {output_file}")


# Compare number sentences
find_neurons_with_largest_mean_differences(
    "activations_mean_number-sentences_clean.csv",
    "activations_mean_number-sentences_off-by-one.csv",
    OUTPUT_PATH + "largest_diff_neurons_numbers.txt"
)

# Compare text sentences (clean vs syntax-error)
find_neurons_with_largest_mean_differences(
    "activations_mean_text-sentences_clean.csv",
    "activations_mean_text-sentences_syntax-error.csv",
    OUTPUT_PATH + "largest_diff_neurons_text_syntax_error.txt"
)

# Compare text sentences (clean vs typo)
find_neurons_with_largest_mean_differences(
    "activations_mean_text-sentences_clean.csv",
    "activations_mean_text-sentences_typo.csv",
    OUTPUT_PATH + "largest_diff_neurons_text_typo.txt"
)

print("Calculations saved!")
