# Filter the clean prompts for optimal conditions
import pandas as pd

# Load clean prompts
clean_prompts = pd.read_csv("clean_prompts.csv", header=None)[0].tolist()

# Function to filter prompts
def is_valid_prompt(sentence):
    words = sentence.split()
    word_count = len(words)
    return 10 <= word_count <= 20 and sentence[0].isupper() and sentence[-1] in ".!?"

# Apply the filter
filtered_prompts = [sentence for sentence in clean_prompts if is_valid_prompt(sentence)]

# Save filtered prompts
output_path = "filtered_clean_prompts.csv"
pd.DataFrame(filtered_prompts, columns=["sentence"]).to_csv(output_path, index=False, header=False)

print(f"Filtered clean prompts saved to {output_path}")
