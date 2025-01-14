import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Paths to data
RESULTS_NUMBERS = "results_numbers/"
RESULTS_TEXT = "results_text/"
PLOTS_DIR = "analyses/plots/neuron-ranges/"

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

# Neuron ranges to analyze
neuron_ranges = [
    (78, 83), (260, 273), (305, 310), (600, 620), 
    (740, 755), (630, 654), (645, 655)
]

# Function to plot specific neuron ranges
def plot_neuron_range(files, data_dir, title, output_file, cmap_names, neuron_range):
    plt.figure(figsize=(12, 6))
    
    for group_idx, (file_group, cmap_name) in enumerate(zip(files, cmap_names)):
        cmap = colormaps[cmap_name]
        shades = np.linspace(0.4, 1, len(file_group))  # Shades for runs

        for i, file in enumerate(file_group):
            try:
                data = pd.read_csv(os.path.join(data_dir, file), header=None, encoding='utf-8')
                selected_data = data.iloc[:, neuron_range[0]:neuron_range[1]]
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue

            mean_activation = selected_data.mean(axis=0)
            plt.plot(mean_activation, label=f"{file.split('_run')[0]} Run {i+1}", color=cmap(shades[i]))

    plt.title(f"{title} (Neurons {neuron_range[0]}-{neuron_range[1]})")
    plt.xlabel("Neuron Index")
    plt.ylabel("Mean Activation")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Plotting logic
for neuron_range in neuron_ranges:
    # Number sentences
    plot_neuron_range(
        [number_files[:5], number_files[5:]],  # Clean, Off-by-One
        RESULTS_NUMBERS,
        f"Number Sentences (Neurons {neuron_range[0]}-{neuron_range[1]})",
        os.path.join(PLOTS_DIR, f"number_sentences_neurons_{neuron_range[0]}_to_{neuron_range[1]}.png"),
        ["Blues", "Oranges"],
        neuron_range
    )

    # Text sentences
    plot_neuron_range(
        [text_files[:3], text_files[3:6], text_files[6:]],  # Clean, Syntax Error, Typo
        RESULTS_TEXT,
        f"Text Sentences (Neurons {neuron_range[0]}-{neuron_range[1]})",
        os.path.join(PLOTS_DIR, f"text_sentences_neurons_{neuron_range[0]}_to_{neuron_range[1]}.png"),
        ["Greens", "Purples", "Reds"],
        neuron_range
    )

print("All plots saved in analyses/plots/shades-activations/")
