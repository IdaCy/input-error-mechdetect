import os
import pandas as pd
import json
from transformers import AutoTokenizer

# Set the paths for raw and processed data
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

# Specify the model tokenizer to use
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to preprocess CSV files
def preprocess_csv(file_name):
    raw_file_path = os.path.join(RAW_DATA_PATH, file_name)
    processed_file_path = os.path.join(PROCESSED_DATA_PATH, file_name)

    # Read the CSV file
    df = pd.read_csv(raw_file_path)

    # Tokenize each sentence and add tokenized data to the DataFrame
    df['tokenized'] = df['sentence'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=128))

    # Save the processed file
    df.to_csv(processed_file_path, index=False)
    print(f"Processed and saved: {processed_file_path}")

# Function to preprocess JSON files
def preprocess_json(file_name):
    raw_file_path = os.path.join(RAW_DATA_PATH, file_name)
    processed_file_path = os.path.join(PROCESSED_DATA_PATH, file_name)

    # Read the JSON file
    with open(raw_file_path, 'r') as f:
        data = json.load(f)

    # Tokenize each sentence
    processed_data = []
    for entry in data:
        tokenized_entry = {
            'original': entry['sentence'],
            'tokenized': tokenizer.encode(entry['sentence'], truncation=True, padding='max_length', max_length=128)
        }
        processed_data.append(tokenized_entry)

    # Save the processed data to a JSON file
    with open(processed_file_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    print(f"Processed and saved: {processed_file_path}")

# Main function to iterate through raw files and preprocess them
def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    for file_name in os.listdir(RAW_DATA_PATH):
        if file_name.endswith('.csv'):
            preprocess_csv(file_name)
        elif file_name.endswith('.json'):
            preprocess_json(file_name)

if __name__ == "__main__":
    main()
