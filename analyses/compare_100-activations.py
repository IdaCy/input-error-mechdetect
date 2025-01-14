import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
ACTIVATIONS_PATH = "data/activations/"
OUTPUT_PATH = "analyses/plots/activationpacks/"

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


# Function to plot 100 neurons at a time
def plot_activations_in_batches(file_list, title_prefix, output_prefix):
    for file in file_list:
        activations = pd.read_csv(os.path.join(ACTIVATIONS_PATH, file), header=None)
        total_neurons = activations.shape[1]
        batch_size = 100
        num_batches = total_neurons // batch_size
        
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            plt.figure(figsize=(10, 6))
            plt.plot(activations.iloc[:, start:end].mean(axis=0))
            plt.title(f"{title_prefix} | Neurons {start} to {end}")
            plt.xlabel("Neuron Index")
            plt.ylabel("Activation Value")
            plt.tight_layout()
            output_file = f"{output_prefix}_neurons_{start}_to_{end}.png"
            plt.savefig(os.path.join(OUTPUT_PATH, output_file))
            plt.close()


# Plot activations for numbers
plot_activations_in_batches(
    number_files, "Number Sentences", "number_activations")

# Plot activations for text
plot_activations_in_batches(text_files, "Text Sentences", "text_activations")

print("Neuron activation graphs saved!")
