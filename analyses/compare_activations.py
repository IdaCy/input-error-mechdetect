import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths to mean activation files
ACTIVATIONS_PATH = "data/activations/"
OUTPUT_PATH = "analyses/plots/"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

number_files = [
    "activations_mean_number-sentences_clean.csv",
    "activations_mean_number-sentences_off-by-one.csv"
]

text_files = [
    "activations_mean_text-sentences_clean.csv",
    "activations_mean_text-sentences_syntax-error.csv",
    "activations_mean_text-sentences_typo.csv"
]


# Load activations
def load_activation(file_path):
    try:
        return pd.read_csv(file_path, header=None)
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        raise e


# Plot mean activation profiles
def plot_mean_activation_profiles(file_list, title, output_file):
    plt.figure(figsize=(10, 6))
    for file in file_list:
        activations = load_activation(os.path.join(ACTIVATIONS_PATH, file))
        mean_activation = activations.mean(axis=0)
        plt.plot(mean_activation, label=file.split('_mean_')[-1].split('.')[0])
    plt.title(title)
    plt.xlabel("Activation Index")
    plt.ylabel("Mean Activation Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, output_file))
    plt.close()


# Plot activation heatmaps
def plot_activation_heatmap(file_list, title, output_file):
    plt.figure(figsize=(12, 6), constrained_layout=True)
    for i, file in enumerate(file_list, 1):
        activations = load_activation(os.path.join(ACTIVATIONS_PATH, file))
        plt.subplot(1, len(file_list), i)
        plt.imshow(activations, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(file.split('_mean_')[-1].split('.')[0])
        plt.xlabel("Activation Index")
        plt.ylabel("Sentence Index")
    plt.suptitle(title)
    plt.savefig(os.path.join(OUTPUT_PATH, output_file))
    plt.close()

# Compare number activations
plot_mean_activation_profiles(
    number_files, 
    "Mean Activation Profiles: Number Sentences", 
    "mean_activation_profiles_numbers.png"
)
plot_activation_heatmap(
    number_files, 
    "Activation Heatmaps: Number Sentences", 
    "activation_heatmaps_numbers.png"
)

# Compare text activations
plot_mean_activation_profiles(
    text_files, 
    "Mean Activation Profiles: Text Sentences", 
    "mean_activation_profiles_text.png"
)
plot_activation_heatmap(
    text_files, 
    "Activation Heatmaps: Text Sentences", 
    "activation_heatmaps_text.png"
)

print("Plots saved to analyses/plots.")
