import pandas as pd
import os

input_dir = 'results/activations/'
output_dir = 'analyses/output/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)

        # Load the activation data
        activations = pd.read_csv(file_path)

        # Compute mean activations
        means = activations.mean(axis=0)

        # Save the results
        output_file = os.path.join(output_dir, f'mean_{file_name}')
        means.to_csv(output_file, index=False)

        print(f"Mean activations for {file_name} " +
              f"computed and saved to {output_file}!")

print("All mean activations computed and saved.")
