#!/usr/bin/env python

import os
import pandas as pd
import json
from transformers import AutoTokenizer

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
LOG_FILE = "logs/preprocess_cpu.log"

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def log(message):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(message + "\n")
    print(message)


def preprocess_csv(file_name):
    raw_file_path = os.path.join(RAW_DATA_PATH, file_name)
    processed_file_path = os.path.join(PROCESSED_DATA_PATH, file_name)

    try:
        log(f"Processing CSV file: {file_name}")
        df = pd.read_csv(raw_file_path, header=None, names=['sentence'])
        df['tokenized'] = df['sentence'].apply(
            lambda x: tokenizer.encode(x, truncation=True,
                                       padding='max_length', max_length=128))
        df.to_csv(processed_file_path, index=False, header=False)
        log(f"Processed and saved: {processed_file_path}")
    except Exception as e:
        log(f"Error processing file {file_name}: {str(e)}")


def preprocess_json(file_name):
    raw_file_path = os.path.join(RAW_DATA_PATH, file_name)
    processed_file_path = os.path.join(PROCESSED_DATA_PATH, file_name)

    try:
        log(f"Processing JSON file: {file_name}")
        with open(raw_file_path, 'r') as f:
            data = json.load(f)
        processed_data = [
            {'original': entry['sentence'],
             'tokenized': tokenizer.encode(entry['sentence'], truncation=True,
                                           padding='max_length',
                                           max_length=128)}
            for entry in data
        ]
        with open(processed_file_path, 'w') as f:
            json.dump(processed_data, f, indent=4)
        log(f"Processed and saved: {processed_file_path}")
    except Exception as e:
        log(f"Error processing file {file_name}: {str(e)}")


def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log("Starting preprocessing...")
    for file_name in os.listdir(RAW_DATA_PATH):
        if file_name.endswith('.csv'):
            preprocess_csv(file_name)
        elif file_name.endswith('.json'):
            preprocess_json(file_name)
        else:
            log(f"Skipped unsupported file: {file_name}")
    log("Preprocessing completed.")


if __name__ == "__main__":
    main()
