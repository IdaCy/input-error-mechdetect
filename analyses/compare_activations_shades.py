import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Paths to data
RESULTS_NUMBERS = "results_numbers/"
RESULTS_TEXT = "results_text/"
PLOTS_DIR = "analyses/plots/shades-activations/"

# Ensure the plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# File structure
number_files = [
    f"activations_run{i}_number-sentences_clean.csv" for i in range(1, 6)
] + [
    f"activations_run{i}_number-sentences_off-by-one.csv" for i in range(1, 6)
]

text_files = [
    f"activations_run{i}_text-sentences_clean.csv" for i in range(1, 4)
] + [
    f"activations_run{i}_text-sentences_syntax-error.csv" for i in range(1, 4)
] + [
    f"activations_run{i}_text-sentences_typo.csv" for i in range(1, 4)
]


# Function to plot activation data for a single 100-neuron pack
def plot_100_neurons_combined(files, data_dir, title, output_file, color_groups):
    plt.figure(figsize=(12, 6))
    
    for group_idx, (file_group, cmap_name) in enumerate(zip(files, color_groups)):
        cmap = colormaps[cmap_name]
        shades = np.linspace(0.4, 1, len(file_group))  # Shades for runs
        
        for i, file in enumerate(file_group):
            try:
                data = pd.read_csv(os.path.join(data_dir, file), header=None, encoding='utf-8').iloc[:, start_neuron:end_neuron]
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
            
            mean_activation = data.mean(axis=0)
            plt.plot(mean_activation, label=f"{file.split('_run')[0]} Run {i+1}", color=cmap(shades[i]))
    
    plt.title(title)
    plt.xlabel("Neuron Index")
    plt.ylabel("Mean Activation")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# Plot all 100-neuron packs for both number and text data
for start_neuron in range(0, 800, 100):  # Iterate over 100-neuron ranges
    end_neuron = start_neuron + 100

    # Number data: Clean and Off-by-One
    plot_100_neurons_combined(
        [number_files[:5], number_files[5:]],  # Clean, Off-by-One
        RESULTS_NUMBERS,
        f"Number Sentences (Neurons {start_neuron}-{end_neuron})",
        os.path.join(PLOTS_DIR, f"number_sentences_neurons_{start_neuron}_to_{end_neuron}.png"),
        ["Blues", "Oranges"]
    )

    # Text data: Clean, Syntax Error, Typo
    plot_100_neurons_combined(
        [text_files[:3], text_files[3:6], text_files[6:]],  # Clean, Syntax Error, Typo
        RESULTS_TEXT,
        f"Text Sentences (Neurons {start_neuron}-{end_neuron})",
        os.path.join(PLOTS_DIR, f"text_sentences_neurons_{start_neuron}_to_{end_neuron}.png"),
        ["Greens", "Purples", "Reds"]
    )

print("All plots saved in analyses/plots/shades-activations/")
