import pandas as pd

# Load mean activations for numbers
clean_numbers = pd.read_csv('analyses/output/mean_activations_number-sentences_clean.csv', header=None)
off_by_one_numbers = pd.read_csv('analyses/output/mean_activations_number-sentences_off-by-one.csv', header=None)

# Compute differences
differences_numbers = off_by_one_numbers - clean_numbers

# Save results
differences_numbers.to_csv('analyses/output/differences_numbers.csv',
                           index=False, header=False)
print("Differences for numbers (clean vs. off-by-one) saved!")
