import pandas as pd

# Load the CSV file
input_file = 'unique_logical_truths.csv'  # Update the file path if necessary
output_file = 'non-duplicated-unique_logical_truths.csv'

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Drop duplicate rows (keep the first occurrence)
df_unique = df.drop_duplicates()

# Save the cleaned DataFrame to a new CSV file
df_unique.to_csv(output_file, index=False)

print(f"Duplicates removed. Unique prompts saved to {output_file}.")
