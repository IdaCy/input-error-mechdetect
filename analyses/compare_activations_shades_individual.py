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
number_files_clean = [
    f"activations_run{i}_number-sentences_clean.csv" for i in range(1, 6)
]
number_files_off_by_one = [
    f"activations_run{i}_number-sentences_off-by-one.csv" for i in range(1, 6)
]

text_files_clean = [
    f"activations_run{i}_text-sentences_clean.csv" for i in range(1, 4)
]
text_files_syntax_error = [
    f"activations_run{i}_text-sentences_syntax-error.csv" for i in range(1, 4)
]
text_files_typo = [
    f"activations_run{i}_text-sentences_typo.csv" for i in range(1, 4)
]


# Function to plot 100-neuron packs
def plot_100_neurons(files, data_dir, title, output_file, cmap_name):
    plt.figure(figsize=(12, 6))
    cmap = colormaps[cmap_name]
    shades = np.linspace(0.4, 1, len(files))  # Create shades for the runs

    for i, file in enumerate(files):
        try:
            data = pd.read_csv(os.path.join(data_dir, file), header=None, encoding='utf-8').iloc[:, :100]
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        mean_activation = data.mean(axis=0)
        plt.plot(mean_activation, label=f"{file.split('_run')[0]} Run {i+1}", color=cmap(shades[i]))

    plt.title(title)
    plt.xlabel("Neuron Index (First 100 Neurons)")
    plt.ylabel("Mean Activation")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# Plotting function for all 100-neuron packs
def plot_all_packs(files, data_dir, title_base, output_prefix, cmap_name):
    for start_neuron in range(0, 800, 100):  # Assuming 800 neurons in total
        end_neuron = start_neuron + 100
        title = f"{title_base} (Neurons {start_neuron}-{end_neuron})"
        output_file = os.path.join(
            PLOTS_DIR, f"{output_prefix}_neurons_{start_neuron}_to_{end_neuron}.png"
        )
        plot_100_neurons(
            [f for f in files], data_dir, title, output_file, cmap_name
        )


# Plot results_numbers
plot_all_packs(number_files_clean, RESULTS_NUMBERS, "Clean Number Sentences", "number_clean", "Blues")
plot_all_packs(number_files_off_by_one, RESULTS_NUMBERS, "Off-by-One Number Sentences", "number_off_by_one", "Oranges")

# Plot results_text
plot_all_packs(text_files_clean, RESULTS_TEXT, "Clean Text Sentences", "text_clean", "Greens")
plot_all_packs(text_files_syntax_error, RESULTS_TEXT, "Syntax Error Text Sentences", "text_syntax_error", "Purples")
plot_all_packs(text_files_typo, RESULTS_TEXT, "Typo Text Sentences", "text_typo", "Reds")

print("All plots saved in analyses/plots/shades-activations/")
