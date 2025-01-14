import os
import pandas as pd

# Paths for input directories and output directory
RESULTS_NUMBERS_PATH = "results_numbers/"
RESULTS_TEXT_PATH = "results_text/"
OUTPUT_PATH = "data/activations/"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)


def combine_results_numbers():
    clean_files = sorted([f for f in os.listdir(RESULTS_NUMBERS_PATH) if "number-sentences_clean" in f])
    off_by_one_files = sorted([f for f in os.listdir(RESULTS_NUMBERS_PATH) if "number-sentences_off-by-one" in f])

    clean_dataframes = [pd.read_csv(os.path.join(RESULTS_NUMBERS_PATH, file), header=None) for file in clean_files]
    off_by_one_dataframes = [pd.read_csv(os.path.join(RESULTS_NUMBERS_PATH, file), header=None) for file in off_by_one_files]

    combined_clean = pd.concat(clean_dataframes, axis=1)
    combined_off_by_one = pd.concat(off_by_one_dataframes, axis=1)

    combined = pd.concat([combined_clean, combined_off_by_one], axis=1)
    combined.to_csv(os.path.join(OUTPUT_PATH, "activations_numbers_combined.csv"), index=False, header=False)


def combine_results_text():
    clean_files = sorted([f for f in os.listdir(RESULTS_TEXT_PATH) if "text-sentences_clean" in f])
    syntax_error_files = sorted([f for f in os.listdir(RESULTS_TEXT_PATH) if "text-sentences_syntax-error" in f])
    typo_files = sorted([f for f in os.listdir(RESULTS_TEXT_PATH) if "text-sentences_typo" in f])

    clean_dataframes = [pd.read_csv(os.path.join(RESULTS_TEXT_PATH, file), header=None) for file in clean_files]
    syntax_error_dataframes = [pd.read_csv(os.path.join(RESULTS_TEXT_PATH, file), header=None) for file in syntax_error_files]
    typo_dataframes = [pd.read_csv(os.path.join(RESULTS_TEXT_PATH, file), header=None) for file in typo_files]

    combined_clean = pd.concat(clean_dataframes, axis=1)
    combined_syntax_error = pd.concat(syntax_error_dataframes, axis=1)
    combined_typo = pd.concat(typo_dataframes, axis=1)

    combined = pd.concat([combined_clean, combined_syntax_error, combined_typo], axis=1)
    combined.to_csv(os.path.join(OUTPUT_PATH, "activations_text_combined.csv"), index=False, header=False)


def main():
    combine_results_numbers()
    combine_results_text()
    print("Combination complete. Files saved in data/activations.")


if __name__ == "__main__":
    main()
